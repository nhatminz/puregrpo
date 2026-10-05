import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from helper import grpo_core
from helper.train_ops import accumulate_groups,pack_training_inputs
from training import initial_data
from fixtures import Target,Tokenizer

SOURCE=Path(__file__).resolve().parents[2]/'SpecNaacl'


def original_scope():
    names={'_get_base_causal_lm','_autocast_dtype','_token_logps_from_hidden',
           'compute_model_token_logps','compute_target_loss_and_backward'}
    tree=ast.parse((SOURCE/'grpo_speculative.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_specnaacl_loss_source','exec'),scope)
    return scope


def test_core_functions_and_reward_files_are_verbatim():
    src=(SOURCE/'grpo_speculative.py').read_text()
    local=Path(grpo_core.__file__).read_text()
    before={n.name:ast.get_source_segment(src,n) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    after={n.name:ast.get_source_segment(local,n) for n in ast.parse(local).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name,code in after.items():
        assert code==before[name],name
    for file in ('rewards.py','get_QAs.py','checkpointing.py'):
        assert (SOURCE/'helper'/file).read_bytes()==(Path(grpo_core.__file__).parent/file).read_bytes()


@pytest.mark.parametrize('iterations',[1,2,3])
@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if torch.cuda.is_available() else []))
def test_real_chunked_loss_gradients_adamw_states_match_source(iterations,device):
    target=Target(device)
    with torch.no_grad(): target.model.adapter.fill_(.05)
    models=[SimpleNamespace(target_model=target),SimpleNamespace(target_model=deepcopy(target))]
    funcs=[grpo_core.compute_target_loss_and_backward,original_scope()['compute_target_loss_and_backward']]
    inputs=torch.tensor([[3,5,4,8,16],[3,5,2,0,0]],device=device)
    attention=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],device=device)
    masks=torch.tensor([[0,1,1,1,1],[0,1,1,0,0]],device=device)
    reward=torch.tensor([[1.],[-1.]],device=device)
    states=[]
    for model,fn in zip(models,funcs):
        optimizer=torch.optim.AdamW(model.target_model.parameters(),lr=1e-5)
        old=reference=None
        losses=[]
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            result=fn(model,inputs,attention,masks,reward,.1,.04,iteration,
                      old_logps=old,ref_logps=reference,chunk_size=2,loss_scale=.5)
            old,reference=result[-2:]
            losses.append(result[:3])
            optimizer.step()
        states.append((losses,optimizer.state_dict()))
    assert states[0][0]==states[1][0]
    for a,b in zip(models[0].target_model.parameters(),models[1].target_model.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for pid,values in states[0][1]['state'].items():
        for key,value in values.items():
            torch.testing.assert_close(value,states[1][1]['state'][pid][key],rtol=0,atol=0)


def test_reward_filtering_normalization_and_text_pack_inherited():
    data=initial_data()
    out=dict(max_sequence_length=8,decoded_sequences=['0','0','1','1','0','0','0','0'])
    messages=[[{'role':'user','content':'q'}]]*2
    used,_=accumulate_groups(data,out,messages,['a','b'],4,
                            lambda values:[1.]*len(values),lambda values,gold:[float(x) for x in values])
    assert used==1 and data['ignore_due_incorrect']==1
    assert data['rewards']==[.2,.2,1.2,1.2]
    # Original NumPy population standard deviation, including FP64 rounding.
    import numpy as np
    expected=np.array([.2,.2,1.2,1.2])
    assert data['std_rewards']==((expected-expected.mean())/expected.std()).tolist()
    ids,attention,mask=pack_training_inputs(data,Tokenizer())
    assert len(ids)==4 and all(len(x)==len(y)==len(z) for x,y,z in zip(ids,attention,mask))


def test_verbatim_update_loop_preserves_microbatch_and_legacy_reward_association():
    # Prove that no "standard GRPO" library/scheduler silently replaces Source's
    # sorting, packing, old/ref logps, scaling or optimizer update cadence.
    src=(SOURCE/'grpo_speculative.py').read_text()
    local=(Path(grpo_core.__file__).parent/'train_ops.py').read_text()
    a=src.index('        for grpo_iteration in range(grpo_iteration_num):',src.index('for epoch in epoch_bar:'))
    b=src.index('            phase_timings.end(target_phase_ticket)',a)+len('            phase_timings.end(target_phase_ticket)')
    original='\n'.join(line[4:] if line.startswith('    ') else line for line in src[a:b].splitlines())
    assert original in local
