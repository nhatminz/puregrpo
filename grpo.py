"""Standalone Pure GRPO entrypoint; CLI validation never loads model/dependencies."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parent)
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0,ROOT)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['puregrpo'], default='puregrpo')
    p.add_argument('--model_dir', required=True)
    p.add_argument('--model_type', default='Qwen2___5-Math-7B')
    p.add_argument('--load_lora_path', default='')
    p.add_argument('--dtype', choices=['auto','bf16','fp16','fp32'], default='auto')
    p.add_argument('--attn_implementation', default='sdpa')
    p.add_argument('--train_option', default='simplelr_abel_level3to5')
    p.add_argument('--dataset_path', required=True)
    p.add_argument('--train_split', default='train')
    p.add_argument('--eval_dataset_path', default='')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--version_name', default='normal')
    p.add_argument('--log_file', required=True)
    p.add_argument('--timing_file', required=True)
    p.add_argument('--summary_file', required=True)
    p.add_argument('--saved_model_dir', required=True)
    p.add_argument('--saved_statistics_dir', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--resume_checkpoint', default='')
    p.add_argument('--append_log', default='')
    p.add_argument('--reset_rng_on_resume', default='false')
    p.add_argument('--persistent_workers', default='true')
    p.add_argument('--statistical_time', default='False')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--validate_config', action='store_true')
    for name, default in dict(batch_size=4, num_epochs=10, sample_num=100,
            accumulation_steps=2, grpo_iteration_num=1, repeated_generate_nums=8,
            max_length=2048, max_prompt_length=2048, max_training_token=3072,
            max_training_padding_gap=256, logps_chunk_size=256, train_subset_seed=42,
            max_train_samples=0, seed=42, num_workers=4, log_interval=1,
            save_checkpoint_steps=0, keep_last_checkpoints=3, max_grpo_steps=0,
            top_k=0, eval_interval=0, max_target_optimizer_steps=0, max_rollout_prompts=0).items():
        p.add_argument('--'+name, type=int, default=default)
    for name, default in dict(target_lr=1e-6, beta=.01, epsilon=.1, temperature=1.,
                              top_p=.95, train_data_fraction=.4).items():
        p.add_argument('--'+name, type=float, default=default)
    args = p.parse_args(argv)
    for key in ('batch_size','num_epochs','sample_num','accumulation_steps','grpo_iteration_num',
                'repeated_generate_nums','max_length','max_prompt_length','max_training_token',
                'logps_chunk_size','log_interval'):
        if getattr(args, key) < 1:
            p.error(key+' must be positive')
    if args.temperature <= 0 or args.target_lr <= 0 or args.train_data_fraction <= 0:
        p.error('temperature/LR/dataset fraction must be positive')
    if any(getattr(args,k) < 0 for k in ('top_k','num_workers','eval_interval','max_grpo_steps','max_train_samples','max_training_padding_gap','max_target_optimizer_steps','max_rollout_prompts')):
        p.error('counts/budgets cannot be negative')
    if args.eval_interval and not args.eval_dataset_path:
        p.error('periodic evaluation requires an explicit held-out eval_dataset_path')
    for key in ('statistical_time','persistent_workers','reset_rng_on_resume'):
        setattr(args, key, str(getattr(args,key)).lower() in {'1','true','yes','on'})
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.validate_config:
        print(json.dumps(vars(args), indent=2))
        return
    # Heavy runtime imports only after CLI/config validation.
    from training import train
    train(args)


if __name__ == '__main__':
    main()
