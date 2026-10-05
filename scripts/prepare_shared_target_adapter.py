#!/usr/bin/env python3
"""Create ONE target LoRA initialization for both methods, without any draft."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dtype',default='bf16',choices=['bf16','fp16','fp32'])
    p.add_argument('--attn_implementation',default='eager')
    p.add_argument('--load_lora_path',default='')
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('output exists; use a NEW initialization directory')
    from helper.grpo_core import _seed_everything
    from helper.target import load_target
    _seed_everything(args.seed)
    target, _ = load_target(args,'cpu')
    target.save_pretrained(args.output)
    print('Shared target-only adapter:',args.output)
    print('Set TARGET_ADAPTER='+args.output+' for BOTH SpecNaacl and Pure GRPO.')


if __name__ == '__main__':
    main()
