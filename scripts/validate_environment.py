#!/usr/bin/env python3
"""Validate only dependencies used by target-only GRPO; no serving/draft imports."""
import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys
from packaging.specifiers import SpecifierSet
from packaging.version import Version

IMPORT_NAMES = {'huggingface-hub':'huggingface_hub','pyyaml':'yaml',
                'typing-extensions':'typing_extensions','math-verify':'math_verify',
                'latex2sympy2-extended':'latex2sympy2_extended'}

# Same admission bounds as the current SpecNaacl validator, minus its kernels.
# Exact pins are installation references, not a reason to replace a working
# shared server environment with a different/faster Pure-only stack.
COMPATIBLE_VERSIONS = {
    'torch':'>=2.5.1,<3', 'transformers':'>=4.51.3,<6', 'peft':'>=0.17.1,<0.22',
    'datasets':'>=4,<6', 'accelerate':'>=1.10.1,<2', 'safetensors':'>=0.6.2,<1',
    'numpy':'>=2.2.6,<3', 'pandas':'>=2.3.2,<4', 'tqdm':'>=4.67.1,<5',
    'math-verify':'>=0.8,<1', 'latex2sympy2-extended':'>=1.10.2,<2',
    'sympy':'>=1.13.1,<2', 'matplotlib':'>=3.10.6,<4',
    'packaging':'>=25,<27', 'pytest':'>=8.4.2,<10',
}


def probe_target_runtime(device):
    """Tiny target-only SDPA/PEFT/cache execution check; no draft import."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.cache_utils import DynamicCache
    from peft import LoraConfig, get_peft_model
    with torch.random.fork_rng():
        config = Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)
        config._attn_implementation = 'sdpa'
        target = get_peft_model(Qwen2ForCausalLM(config).to(device),
                               LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=2,
                                          target_modules=['q_proj','v_proj']))
        ids = torch.tensor([[1,2,3]],device=device)
        result = target(input_ids=ids,use_cache=False)
        result.logits.float().sum().backward()
        if not any(p.grad is not None for p in target.parameters() if p.requires_grad):
            raise RuntimeError('target LoRA backward produced no gradients')
        target.eval()
        with torch.no_grad():
            cache = DynamicCache()
            out = target(input_ids=ids,past_key_values=cache,use_cache=True)
            cache = out.past_key_values
            cache.batch_repeat_interleave(2)
            cache.batch_select_indices(torch.tensor([1],device=device))
            target(input_ids=ids[:,-1:],past_key_values=cache,use_cache=True)
    return 'target SDPA, LoRA backward and autoregressive DynamicCache OK'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requirements',type=Path,default=Path(__file__).resolve().parents[1]/'requirements.txt')
    p.add_argument('--require-cuda',action='store_true')
    p.add_argument('--strict-versions',action='store_true')
    args = p.parse_args(argv)
    if sys.version_info < (3,10):
        raise RuntimeError('Python >=3.10 required, same interpreter gate as current SpecNaacl')
    failures = []
    for line in args.requirements.read_text().splitlines():
        line = line.split('#',1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s;]+)',line)
        if m is None:
            raise ValueError('dependency must be exactly pinned: '+line)
        name, expected = m.groups()
        try:
            actual = version(name).split('+',1)[0]
            bounds = COMPATIBLE_VERSIONS.get(name)
            if actual != expected and (args.strict_versions or not bounds or Version(actual) not in SpecifierSet(bounds)):
                raise RuntimeError(f'supported {expected if args.strict_versions else bounds or expected}, found {actual}')
            module = importlib.import_module(IMPORT_NAMES.get(name,name.replace('-','_')))
            if name == 'peft':
                for key in ('LoraConfig','TaskType','get_peft_model','get_peft_model_state_dict','set_peft_model_state_dict'):
                    if not hasattr(module,key):
                        raise RuntimeError('missing PEFT API '+key)
        except Exception as error:
            failures.append(f'{name}: {type(error).__name__}: {error}')
    if failures:
        raise RuntimeError(f'Interpreter: {sys.executable}\nenvironment validation failed:\n- '+'\n- '.join(failures))
    torch = importlib.import_module('torch')
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but torch.cuda.is_available() is false')
    print(probe_target_runtime('cuda' if args.require_cuda else 'cpu'))
    print(f'Pure GRPO environment validation passed: {sys.executable}, torch={torch.__version__}, cuda={torch.version.cuda}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError,ValueError) as error:
        raise SystemExit(str(error)) from None
