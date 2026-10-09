from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import pytest
from grpo import parse_args
import training
from helper.autoregressive import autoregressive_generate
from helper.target_checkpoint import save_checkpoint,load_checkpoint
from helper.checkpointing import capture_rng_state,restore_rng_state
from evaluation import evaluate
from fixtures import Target,Tokenizer,Cache


def args_for(root):
    return parse_args(['--model_dir','unused-in-dependency-injected-test','--dataset_path','unused-in-test',
        '--log_file',str(root/'logs/metrics.jsonl'),'--timing_file',str(root/'logs/timing.csv'),
        '--summary_file',str(root/'summary.json'),'--saved_model_dir',str(root/'checkpoints/target'),
        '--saved_statistics_dir',str(root/'statistics'),'--checkpoint_dir',str(root/'checkpoints/resume'),
        '--batch_size','2','--accumulation_steps','2','--repeated_generate_nums','4',
        '--num_epochs','1','--num_workers','0','--persistent_workers','false',
        '--max_length','7','--max_prompt_length','3','--max_training_token','32',
        '--logps_chunk_size','2','--save_checkpoint_steps','1'])


@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
def test_collect_grpo_checkpoint_evaluate_and_export_without_spec_imports(tmp_path,monkeypatch,device):
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.startswith(('specforge','sglang','helper.eagle3_specforge','helper.modeling_draft',
                                    'helper.specualtive_generate','helper.fast_lk_reflex','helper.tree_verification')):
                raise AssertionError('forbidden component import '+fullname)
    guard=Guard(); sys.meta_path.insert(0,guard)
    target=Target(device)
    initial=target.model.adapter.detach().clone()
    args=args_for(tmp_path)
    call=[0]
    def reward(values,truth):
        call[0]+=1
        return [float(call[0]%2)]
    def generate(*a,**k): return autoregressive_generate(*a,cache_factory=Cache,**k)
    actual_save=training.save_checkpoint
    monkeypatch.setattr(training,'save_checkpoint',lambda *a,target,**k:
                        actual_save(*a,target=target,adapter_state=target.state_dict(),**k))
    try:
        summary=training.train(args,target=target,tokenizer=Tokenizer(),device=device,
            train_rows=[{'question':str(i),'answer':'0'} for i in range(8)],
            reward_functions=(lambda x:[0.],reward),generate=generate)
        assert summary['optimizer_steps']==4  # not deferred until accumulation=2
        assert not torch.equal(target.model.adapter,initial)
        assert not hasattr(target,'draft_model')
        assert summary['generated_samples']==32 and summary['total_rollout_tokens']>0
        report=json.loads((tmp_path/'summary.json').read_text())
        assert not any('aal' in k or 'acceptance' in k for k in report)
        rows=[json.loads(line) for line in (tmp_path/'logs/metrics.jsonl').read_text().splitlines()]
        step_rows=[r for r in rows if r['phase']=='target_train']
        assert len(step_rows)==len({r['step'] for r in step_rows})==3
        import csv
        timings=list(csv.DictReader((tmp_path/'logs/timing.csv').open()))
        assert len(timings)==len(step_rows)
        before=deepcopy(target.state_dict())
        evaluation=evaluate(args,target,Tokenizer(),rows=[{'question':'heldout','answer':'0'}],
                            reward_functions=(lambda x:[0.],reward),generate=generate)
        assert evaluation['generated_samples']==4
        for key,value in target.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        assert (tmp_path/'evaluation/final_per_response.jsonl').is_file()
        payload=torch.load(tmp_path/'checkpoints/resume/latest.pt',weights_only=False,map_location='cpu')
        assert not any('draft' in k or 'reflex' in k for k in payload)
        assert payload['format']=='puregrpo_target_checkpoint_v1'
    finally:
        sys.meta_path.remove(guard)


def test_target_checkpoint_optimizer_rng_restore_and_foreign_rejection(tmp_path):
    target=Target()
    optimizer=torch.optim.AdamW(target.parameters(),lr=1e-5)
    target.model.adapter.grad=torch.ones_like(target.model.adapter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    saved=deepcopy(target.state_dict())
    checkpoint=save_checkpoint(tmp_path,target=target,optimizer=optimizer,data={'used_items':5},
        epoch=0,next_batch=2,step=1,wall=1.,adapter_state=target.state_dict())
    sample=torch.rand(5)
    target.model.adapter.data.add_(3)
    load_checkpoint(checkpoint,target=target,optimizer=optimizer,adapter_loader=lambda m,s:m.load_state_dict(s))
    assert torch.equal(sample,torch.rand(5))
    for key,value in target.state_dict().items(): torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
    torch.save({'format':'specnaacl_fastgrpo_checkpoint_v3'},tmp_path/'foreign.pt')
    with pytest.raises(ValueError,match='Pure GRPO checkpoint'):
        load_checkpoint(tmp_path/'foreign.pt',target=target,optimizer=optimizer)


@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
def test_mid_epoch_resume_matches_uninterrupted_policy_and_generated_counts(tmp_path,monkeypatch,device):
    actual_save,actual_load=training.save_checkpoint,training.load_checkpoint
    monkeypatch.setattr(training,'save_checkpoint',lambda *a,target,**k:
                        actual_save(*a,target=target,adapter_state=target.state_dict(),**k))
    monkeypatch.setattr(training,'load_checkpoint',lambda *a,**k:
                        actual_load(*a,adapter_loader=lambda model,state:model.load_state_dict(state),**k))
    rows=[{'question':str(i),'answer':'0'} for i in range(8)]
    def generate(*a,**k): return autoregressive_generate(*a,cache_factory=Cache,**k)
    def run(target,args):
        count=[0]
        def reward(values,truth):
            count[0]+=1
            return [float(count[0]%2)]
        return training.train(args,target=target,tokenizer=Tokenizer(),device=device,
            train_rows=rows,reward_functions=(lambda x:[0.],reward),generate=generate)
    full=Target(device); full_summary=run(full,args_for(tmp_path/'full'))
    partial=Target(device); short=args_for(tmp_path/'resume'); short.max_grpo_steps=1
    run(partial,short)
    continued=Target(device); resume=args_for(tmp_path/'resume')
    resume.resume_checkpoint=str(tmp_path/'resume/checkpoints/resume/latest.pt')
    resume_summary=run(continued,resume)
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,continued.state_dict()[key],rtol=0,atol=0)
    assert full_summary['total_rollout_tokens']==resume_summary['total_rollout_tokens']
    assert full_summary['optimizer_steps']==resume_summary['optimizer_steps']==4
    assert full_summary['generated_samples']==resume_summary['generated_samples']==32


def test_optional_evaluation_does_not_change_main_trajectory(tmp_path,monkeypatch):
    from evaluation import load_eval_rows
    import evaluation
    monkeypatch.setattr(evaluation,'load_eval_rows',lambda args:[{'question':'heldout','answer':'0'}])
    actual_save=training.save_checkpoint
    monkeypatch.setattr(training,'save_checkpoint',lambda *a,target,**k:
                        actual_save(*a,target=target,adapter_state=target.state_dict(),**k))
    rows=[{'question':str(i),'answer':'0'} for i in range(8)]
    states=[]
    for periodic in (False,True):
        target=Target(); args=args_for(tmp_path/str(periodic))
        if periodic: args.eval_interval=1; args.eval_dataset_path='injected-heldout-test'
        count=[0]
        def reward(values,truth):
            count[0]+=1
            return [float(count[0]%2)]
        training.train(args,target=target,tokenizer=Tokenizer(),device='cpu',train_rows=rows,
            reward_functions=(lambda x:[0.],reward),
            generate=lambda *a,**k:autoregressive_generate(*a,cache_factory=Cache,**k))
        states.append(deepcopy(target.state_dict()))
    for key,value in states[0].items(): torch.testing.assert_close(value,states[1][key],rtol=0,atol=0)


def test_actual_optimizer_and_prompt_budgets_do_not_overshoot(tmp_path,monkeypatch):
    actual_save=training.save_checkpoint
    monkeypatch.setattr(training,'save_checkpoint',lambda *a,target,**k:
        actual_save(*a,target=target,adapter_state=target.state_dict(),**k))
    args=args_for(tmp_path);args.grpo_iteration_num=3
    args.max_target_optimizer_steps=1;args.max_rollout_prompts=1
    calls=[0]
    def reward(values,truth):
        calls[0]+=1;return [float(calls[0]%2)]
    def generate(*a,**k):return autoregressive_generate(*a,cache_factory=Cache,**k)
    report=training.train(args,target=Target(),tokenizer=Tokenizer(),device='cpu',
        train_rows=[{'question':str(i),'answer':'0'} for i in range(8)],
        reward_functions=(lambda x:[0.],reward),generate=generate)
    assert report['optimizer_steps']==1 and report['rollout_prompts_seen']==1
    assert report['optimizer_step_cadence']==[(1,1)]
