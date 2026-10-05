#!/usr/bin/env python3
"""Read-only launcher comparison; does not import SpecNaacl training code."""
import argparse
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


def command(root, key, env):
    text = subprocess.run(['bash',str(root/f'train_{key}.sh')],cwd=root,env=env,
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


def compare(key, source, env=None):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ if env is None else env, DRY_RUN='true',
               PYTHON_BIN=(env or os.environ).get('PYTHON_BIN',sys.executable),
               RUN_NAME='pure_grpo_config_check',RUN_DIR='/tmp/pure_grpo_config_check_no_writes',RESUME='')
    pure_cmd,a = command(root,key,env)
    spec_cmd,b = command(source,key,env)
    differences = {k:(a.get('--'+k),b.get('--'+k)) for k in COMMON_FLAGS if a.get('--'+k)!=b.get('--'+k)}
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
                pure_method=a['--method'],comparison_method=b['--method'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-key',default='qwen25_3b')
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2]/'SpecNaacl')
    args = p.parse_args()
    print(json.dumps(compare(args.model_key,args.source.resolve()),indent=2))


if __name__ == '__main__':
    main()
