#!/usr/bin/env python3
"""Read-only launcher comparison; does not import SpecNaacl training code."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

COMMON_FLAGS = ('model_dir','model_type','dtype','attn_implementation','load_lora_path',
    'train_option','dataset_path','train_data_fraction','train_subset_seed','max_train_samples',
    'batch_size','num_epochs','sample_num','accumulation_steps','target_lr','temperature','top_p',
    'max_length','max_prompt_length','max_training_padding_gap','max_training_token','logps_chunk_size',
    'grpo_iteration_num','repeated_generate_nums','beta','epsilon','statistical_time','num_workers',
    'persistent_workers','log_interval','seed','save_checkpoint_steps','keep_last_checkpoints',
    'log_file','timing_file','summary_file','saved_model_dir','saved_statistics_dir','checkpoint_dir')
MODELS = ('qwen25_1p5b','qwen25_3b','qwen25_7b','qwen25_14b',
          'qwen3_1p7b','qwen3_4b','llama31_8b')
COMMON_DEFAULTS = {'train_split':'train', 'max_grpo_steps':'0', 'reset_rng_on_resume':'false',
                   'resume_checkpoint':''}


def lora_recipe(path):
    """Inspect configuration without importing either training entrypoint."""
    calls = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id == 'LoraConfig']
    if len(calls) != 1:
        raise ValueError('Expected one target LoRA recipe in '+str(path))
    recipe = {}
    for kw in calls[0].keywords:
        recipe[kw.arg] = ast.unparse(kw.value) if kw.arg == 'task_type' else ast.literal_eval(kw.value)
    if 'target_modules' in recipe:
        recipe['target_modules'] = sorted(recipe['target_modules'])
    return recipe


def default_environment():
    # Defaults must not accidentally pass because TARGET_LR/DATASET exported by
    # the caller mask an incorrect launcher default. Explicit override checks
    # use --use-environment or compare(..., env=...).
    names = ('PATH','HOME','PYTHONPATH','LD_LIBRARY_PATH','PYTHON_BIN','LANG','LC_ALL')
    return {key:os.environ[key] for key in names if key in os.environ}


def shared_behavior(root,source):
    """Read-only checks for common settings that are not launcher arguments."""
    for name in ('rewards.py','get_QAs.py'):
        if (root/'helper'/name).read_bytes() != (source/'helper'/name).read_bytes():
            raise ValueError('Unfair common config: dataset/reward implementation differs: '+name)
    pure_tree = ast.parse((root/'helper/grpo_core.py').read_text())
    source_tree = ast.parse((source/'grpo_speculative.py').read_text())
    def collator(tree):
        return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrainDataCollator')
    if ast.dump(collator(pure_tree)) != ast.dump(collator(source_tree)):
        raise ValueError('Unfair common config: target prompt template/collator differs')
    def adamw_settings(tree):
        call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute) and n.func.attr=='AdamW')
        # LR is already compared from the effective CLI. Parameter containers
        # have different variable names but refer to the same target recipe.
        return {kw.arg:ast.literal_eval(kw.value) for kw in call.keywords if kw.arg != 'lr'}
    if adamw_settings(ast.parse((root/'training.py').read_text())) != adamw_settings(source_tree):
        raise ValueError('Unfair common config: explicit target AdamW settings differ')
    return {'dataset_and_reward':'match','prompt_template_and_collator':'match',
            'target_optimizer':'torch.optim.AdamW with same shared-runtime defaults'}


def command(root, key, env, method=None, extra_args=()):
    suffix = '_fastgrpo' if method == 'fastgrpo' else ''
    text = subprocess.run(['bash',str(root/f'train_{key}{suffix}.sh'),*extra_args],cwd=root,env=env,
                          capture_output=True,text=True,check=True).stdout
    tokens = shlex.split(next(line.split(':',1)[1] for line in text.splitlines() if line.startswith('Command  :')))
    flags = {}
    for index,t in enumerate(tokens):
        if t.startswith('--'):
            if '=' in t:
                k,v = t.split('=',1)
            else:
                k,v = t,tokens[index+1]
            flags[k] = v
    return tokens,flags


def compare(key, source, env=None, method='opd_reflex', extra_args=()):
    root = Path(__file__).resolve().parents[1]
    env = dict(default_environment() if env is None else env, DRY_RUN='true',
               PYTHON_BIN=(env or os.environ).get('PYTHON_BIN',sys.executable),
               RUN_NAME='pure_grpo_config_check',RUN_DIR='/tmp/pure_grpo_config_check_no_writes',RESUME='')
    # A METHOD override must not turn the requested source baseline into OPD.
    env.pop('METHOD',None)
    pure_cmd,a = command(root,key,env,extra_args=extra_args)
    spec_cmd,b = command(source,key,env,method=method,extra_args=extra_args)
    differences = {k:(a.get('--'+k),b.get('--'+k)) for k in COMMON_FLAGS if a.get('--'+k)!=b.get('--'+k)}
    for key_name,default in COMMON_DEFAULTS.items():
        if a.get('--'+key_name,default) != b.get('--'+key_name,default):
            differences[key_name] = (a.get('--'+key_name,default),b.get('--'+key_name,default))
    recipes = (lora_recipe(root/'helper/target.py'),lora_recipe(source/'grpo_speculative.py'))
    behavior = shared_behavior(root,source)
    if recipes[0] != recipes[1]:
        differences['target_lora_recipe'] = recipes
    if a.get('--nproc_per_node') != b.get('--nproc_per_node'):
        differences['nproc_per_node'] = (a.get('--nproc_per_node'),b.get('--nproc_per_node'))
    if a.get('--top_k','0')!='0':
        differences['top_k'] = (a.get('--top_k'),'SpecNaacl training currently forwards None')
    if a.get('--eval_interval','0')!='0':
        differences['eval_interval'] = (a.get('--eval_interval'),'Source has no ordinary periodic target evaluation')
    if any('--'+key in a for key in ('adapter_path','draft_backend','draft_config','vocab_mapping','reflex_mode')):
        raise AssertionError('Pure GRPO launcher contains a forbidden generation component')
    if differences:
        raise ValueError('Unfair common config: '+json.dumps(differences))
    return dict(model_key=key,matched_common_flags={k:a['--'+k] for k in COMMON_FLAGS},
                nproc_per_node=a.get('--nproc_per_node'),top_k=None,eval_interval=0,
                shared_lora_recipe=recipes[0],
                common_behavior=behavior,
                target_initialization=('shared path configured; tensor verification occurs at load'
                                       if a['--load_lora_path'] else
                                       'NOT guaranteed by seed: set TARGET_ADAPTER for paired training'),
                common_update_budget={k:a.get('--'+k,v) for k,v in COMMON_DEFAULTS.items()},
                pure_method=a['--method'],comparison_method=b['--method'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-key',choices=MODELS,help='omit to check all supported models')
    p.add_argument('--method',choices=['fastgrpo','opd_reflex','both'],default='both')
    p.add_argument('--use-environment',action='store_true',help='check exported overrides instead of clean defaults')
    p.add_argument('--training-arg',action='append',default=[],help='extra CLI argument, e.g. --training-arg=--max_grpo_steps=2')
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'SpecNaacl')
    args = p.parse_args()
    env = dict(os.environ) if args.use_environment else default_environment()
    models = (args.model_key,) if args.model_key else MODELS
    methods = ('fastgrpo','opd_reflex') if args.method == 'both' else (args.method,)
    results = [compare(key,args.source.resolve(),env,method,args.training_arg)
               for key in models for method in methods]
    print(json.dumps({'status':'PASS','checks':len(results),'results':results},indent=2))


if __name__ == '__main__':
    main()
