import ast
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helper.shared_adapter import adapter_artifact,verify_loaded_adapter,validate_adapter_recipe
from scripts.check_fair_config import lora_recipe

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'SpecNaacl'


def test_shared_checkpoint_exact_despite_different_rng_and_no_new_parameters(tmp_path):
    peft = pytest.importorskip('peft')
    tf = pytest.importorskip('transformers')
    torch.manual_seed(42)
    config = tf.Qwen2Config(vocab_size=17,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    base = tf.Qwen2ForCausalLM(config)
    recipe = lora_recipe(SOURCE/'grpo_speculative.py')
    recipe['task_type'] = peft.TaskType.CAUSAL_LM
    initial = peft.get_peft_model(deepcopy(base),peft.LoraConfig(**recipe))
    initial.save_pretrained(tmp_path/'shared')
    reference = peft.get_peft_model_state_dict(initial)
    for method,seed in zip(('puregrpo','fastgrpo','opd_reflex'),(12,34,56)):
        # This is the actual upstream target initialization recipe and load API,
        # not an import of the SpecNaacl entrypoint (which would launch training).
        torch.manual_seed(seed)
        if method=='puregrpo':
            recipe = lora_recipe(ROOT/'helper/target.py')
            recipe['task_type'] = peft.TaskType.CAUSAL_LM
        target = peft.get_peft_model(deepcopy(base),peft.LoraConfig(**recipe))
        assert any(not torch.equal(reference[k],v) for k,v in peft.get_peft_model_state_dict(target).items())
        before = {k:(tuple(v.shape),v.requires_grad) for k,v in target.named_parameters()}
        target.load_adapter(tmp_path/'shared',adapter_name='default')
        verify_loaded_adapter(target,tmp_path/'shared')
        assert before == {k:(tuple(v.shape),v.requires_grad) for k,v in target.named_parameters()}
        for k,v in peft.get_peft_model_state_dict(target).items():
            assert torch.equal(v,reference[k]),method
        assert not hasattr(target,'draft_model')


def test_shared_initialization_rejects_changed_recipe_or_corrupt_manifest(tmp_path):
    config = {'r':64,'lora_alpha':32,'lora_dropout':0.,'bias':'none',
        'task_type':'CAUSAL_LM','peft_type':'LORA',
        'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}
    (tmp_path/'adapter_config.json').write_text(json.dumps(config))
    with pytest.raises(FileNotFoundError,match='Missing adapter weights'):
        adapter_artifact(tmp_path)
    (tmp_path/'adapter_model.bin').write_bytes(b'artifact only; never loaded')
    (tmp_path/'shared_initialization.json').write_text(json.dumps({'adapter_weights_sha256':'wrong'}))
    with pytest.raises(ValueError,match='changed after initialization'):
        adapter_artifact(tmp_path)
    config['r'] = 8
    (tmp_path/'adapter_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='does not match SpecNaacl'):
        validate_adapter_recipe(tmp_path)


def test_prepare_script_and_production_loader_roundtrip_without_draft(tmp_path,monkeypatch):
    pytest.importorskip('peft')
    tf = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    import sys
    from helper.target import load_target
    from scripts.prepare_shared_target_adapter import main
    base_path,shared = tmp_path/'tiny_target_test_only',tmp_path/'shared'
    cfg = tf.Qwen2Config(vocab_size=17,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
    tf.Qwen2ForCausalLM(cfg).save_pretrained(base_path)
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({'[UNK]':0,'[EOS]':16},unk_token='[UNK]'))
    tf.PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]').save_pretrained(base_path)
    monkeypatch.setattr(sys,'argv',['prepare_shared_target_adapter.py','--model_dir',str(base_path),
                                  '--output',str(shared),'--dtype','fp32'])
    main()
    manifest = json.loads((shared/'shared_initialization.json').read_text())
    assert manifest['adapter_weights_sha256']==adapter_artifact(shared)[1]
    assert manifest['attn_implementation']=='sdpa'
    torch.manual_seed(987)
    target,_ = load_target(SimpleNamespace(model_dir=str(base_path),dtype='fp32',
        attn_implementation='sdpa',load_lora_path=str(shared)),'cpu')
    assert not hasattr(target,'draft_model')
    verify_loaded_adapter(target,shared,str(base_path))
    with pytest.raises(ValueError,match='different target model'):
        validate_adapter_recipe(shared,tmp_path/'other_model')


def test_existing_sampler_reduced_precision_difference_is_not_hidden():
    import warnings
    from helper.sampling import build_sampling_probs
    source = SOURCE/'helper/fastgrpo_generate.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='sampling')
    captured = {}
    class MultinomialSpy:
        def __getattr__(self,name):
            return getattr(torch,name)
        def multinomial(self,probs,**kwargs):
            captured['probs']=probs.clone()
            return torch.multinomial(probs,**kwargs)
    scope = {'torch':MultinomialSpy(),'F':torch.nn.functional,'warnings':warnings}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    logits = torch.arange(17,dtype=torch.bfloat16).div(7).view(1,1,17)
    scope['sampling'](logits,temperature=1.,top_p=.95,eos_token_id=16)
    assert captured['probs'].dtype==torch.bfloat16
    assert build_sampling_probs(logits,1.,.95,None,16).dtype==torch.float32


def test_parquet_array_prompt_matches_current_source(tmp_path):
    pytest.importorskip('datasets')
    from helper.get_QAs import get_QAs_from_path
    import pandas as pd
    pd.DataFrame([{'prompt':[{'role':'user','content':'2+2?'}],
                   'reward_model':{'ground_truth':'4'}}]).to_parquet(tmp_path/'train.parquet')
    result = get_QAs_from_path(str(tmp_path/'train.parquet'),'train')
    assert result == [{'question':'2+2?','answer':r'\boxed{4}'}]


def test_runtime_reference_common_pins_and_compatible_bounds_match():
    from scripts.validate_environment import COMPATIBLE_VERSIONS
    def pins(path):
        return dict(line.split('==') for line in path.read_text().splitlines() if '==' in line and not line.startswith('#'))
    pure,spec = pins(ROOT/'requirements.txt'),pins(SOURCE/'requirements.txt')
    assert all(spec[k]==v for k,v in pure.items())
    tree = ast.parse((SOURCE/'scripts/validate_environment.py').read_text())
    bound = next(n.value for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='COMPATIBLE_VERSIONS' for t in n.targets))
    source_bounds = ast.literal_eval(bound)
    assert all(source_bounds[k]==v for k,v in COMPATIBLE_VERSIONS.items())


def test_all_model_checker_is_default_and_common_budget_is_forwarded():
    import os
    import subprocess
    import sys
    run = subprocess.run([sys.executable,str(ROOT/'scripts/check_fair_config.py'),
        '--training-arg=--max_grpo_steps=2'],capture_output=True,text=True,check=True)
    report = json.loads(run.stdout)
    assert report['status']=='PASS' and report['checks']==14
    assert {r['comparison_method'] for r in report['results']}=={'fastgrpo','opd_reflex'}
    for row in report['results']:
        assert row['common_update_budget']['max_grpo_steps']=='2'
        assert row['matched_common_flags']['target_lr']=='1e-6'
    # Validate actual exported initialization path, not a fabricated checkpoint.
    # DRY_RUN accepts a path without touching/loading it; tensor verification is
    # separately covered by the real PEFT test above.
    from scripts.check_fair_config import compare
    result = compare('qwen25_3b',SOURCE,dict(os.environ,TARGET_ADAPTER='/tmp/shared_adapter_dry_run_only'))
    assert result['matched_common_flags']['load_lora_path']=='/tmp/shared_adapter_dry_run_only'


def test_benchmark_dry_run_uses_three_methods_same_adapter_and_interpreter(tmp_path):
    import os
    import shlex
    import subprocess
    import sys
    env = dict(os.environ,DRY_RUN='true',PYTHON_BIN=sys.executable,
        BENCHMARK_ROOT=str(tmp_path/'no_writes'),BENCHMARK_STEPS='3',
        TARGET_ADAPTER=str(tmp_path/'initialization_dry_run_only'))
    result = subprocess.run(['bash',str(ROOT/'scripts/benchmark_pair.sh')],
        env=env,capture_output=True,text=True,check=True)
    lines = [shlex.split(line.split(':',1)[1]) for line in result.stdout.splitlines()
             if line.startswith('Command  :')]
    assert len(lines)==3
    assert {line[line.index('--method')+1] for line in lines}=={'fastgrpo','opd_reflex','puregrpo'}
    for line in lines:
        assert line[0]==sys.executable
        assert line[line.index('--load_lora_path')+1]==env['TARGET_ADAPTER']
        assert line[line.index('--max_grpo_steps')+1]=='3'
    assert not (tmp_path/'no_writes').exists()


def test_zero_valid_mask_difference_is_reported_not_silently_changed():
    from fixtures import Target
    from helper.grpo_core import compute_target_loss_and_backward
    from test_grpo_parity import original_scope
    ids = torch.tensor([[1,2,3]])
    mask = torch.zeros_like(ids)
    advantages = torch.tensor([[1.]])
    pure = compute_target_loss_and_backward(SimpleNamespace(target_model=Target()),
        ids,torch.ones_like(ids),mask,advantages,.1,.04,0)
    source = original_scope()['compute_target_loss_and_backward'](SimpleNamespace(target_model=Target()),
        ids,torch.ones_like(ids),mask,advantages,.1,.04,0)
    assert pure[0]==0.
    assert torch.isnan(torch.tensor(source[0]))
