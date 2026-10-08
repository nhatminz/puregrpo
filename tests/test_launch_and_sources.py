import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import pytest
from scripts.check_fair_config import compare,command

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'SpecNaacl'
KEYS=('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b','qwen3_1p7b','qwen3_4b','llama31_8b')


@pytest.mark.parametrize('key',KEYS)
@pytest.mark.parametrize('overrides',[False,True])
@pytest.mark.parametrize('method',['fastgrpo','opd_reflex'])
def test_source_and_pure_launchers_have_identical_common_settings(key,overrides,method):
    env=dict(os.environ,PYTHON_BIN=sys.executable)
    if overrides:
        env.update(DATASET='simplelr',TARGET_LR='2e-5',BATCH_SIZE='3',ACCUMULATION_STEPS='7',
                   REPEATED_GENERATE_NUMS='3',LOG_INTERVAL='5',TEMPERATURE='.8',TOP_P='.9',
                   TRAIN_SUBSET_SEED='101',CUDA_VISIBLE_DEVICES='2,3',NPROC_PER_NODE='2')
        env.pop('RESPONSES_PER_PROMPT',None)
    result=compare(key,SOURCE,env,method)
    assert result['pure_method']=='puregrpo' and result['comparison_method']==method


def test_unmatched_topk_and_periodic_evaluation_are_not_claimed_fair():
    for config in ({'TOP_K':'3'},{'EVAL_INTERVAL':'2'}):
        with pytest.raises(ValueError,match='Unfair common config'):
            compare('qwen25_3b',SOURCE,dict(os.environ,PYTHON_BIN=sys.executable,**config))


def test_no_speculative_runtime_imports_and_no_aal_placeholders():
    forbidden=('specforge','sglang','helper.specualtive_generate','helper.eagle3_specforge',
               'helper.modeling_draft','helper.tree_verification','helper.fast_lk_reflex')
    for path in list((ROOT/'helper').glob('*.py'))+[ROOT/'training.py',ROOT/'evaluation.py',ROOT/'grpo.py']:
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom): names=[node.module or '']
            else: continue
            assert not any(name.startswith(forbidden) for name in names),path
    pins=(ROOT/'requirements.txt').read_text()
    for package in ('sglang','cuda-tile','yunchang','triton','flash-attn'):
        assert package+'==' not in pins
    from helper.metrics import STEP_FIELDS
    assert not any('aal' in k or 'acceptance' in k or 'draft' in k for k in STEP_FIELDS)


def test_all_shell_syntax_and_cli_config_validation(tmp_path):
    for path in ROOT.rglob('*.sh'):
        subprocess.run(['bash','-n',str(path)],check=True)
    env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=sys.executable,OUTPUT_ROOT=str(tmp_path/'output'))
    tokens,flags=command(ROOT,'qwen25_3b',env)
    index=tokens.index(str(ROOT/'grpo.py'))
    result=subprocess.run([sys.executable,*tokens[index:],'--validate_config'],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    config=json.loads(result.stdout)
    assert config['repeated_generate_nums']==8 and config['target_lr']==1e-6
    assert config['attn_implementation']=='sdpa'
    assert config['train_option']=='simplelr_abel_level3to5'
    assert not (tmp_path/'output').exists()


def test_source_fingerprints_unchanged_and_output_parent_preserved():
    # Original SOURCE_PROVENANCE is historical, not a claim that Source never
    # evolves. This audit records the actual revision inspected for this task.
    manifest=json.loads((ROOT/'FAIRNESS_AUDIT.json').read_text())
    for path,sha in manifest['inspected_source_fingerprints'].items():
        assert hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()==sha,path
    for path,sha in manifest['unchanged_pure_execution_files'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    env=dict(os.environ,DRY_RUN='true',PYTHON_BIN=sys.executable)
    _,pure=command(ROOT,'qwen25_3b',env)
    _,spec=command(SOURCE,'qwen25_3b',env)
    # Only the unique method-specific RUN directory differs, not its parent.
    assert Path(pure['--log_file']).resolve().parents[2]==Path(spec['--log_file']).resolve().parents[2]
    assert 'method-puregrpo' in pure['--log_file']


def test_direct_cli_common_defaults_also_match_source(tmp_path):
    from grpo import parse_args
    original=ast.parse((SOURCE/'grpo_speculative.py').read_text())
    defaults={}
    for node in original.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Attribute) and call.func.attr=='add_argument':
                name=call.args[0].value.removeprefix('--')
                for k in call.keywords:
                    if k.arg=='default' and isinstance(k.value,ast.Constant): defaults[name]=k.value.value
    args=parse_args(['--model_dir','unused','--dataset_path','unused',
        '--log_file','unused','--timing_file','unused','--summary_file','unused',
        '--saved_model_dir','unused','--saved_statistics_dir','unused','--checkpoint_dir','unused'])
    for name in ('batch_size','accumulation_steps','num_epochs','sample_num','target_lr','beta',
                 'epsilon','temperature','top_p','train_data_fraction','repeated_generate_nums',
                 'max_length','max_prompt_length','max_training_token','max_training_padding_gap',
                 'logps_chunk_size','num_workers','seed','log_interval','save_checkpoint_steps',
                 'keep_last_checkpoints','dtype','attn_implementation','persistent_workers'):
        assert getattr(args,name)==defaults[name],name
