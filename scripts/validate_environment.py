#!/usr/bin/env python3
"""Validate only dependencies used by target-only GRPO; no serving/draft imports."""
import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys

IMPORT_NAMES = {'huggingface-hub':'huggingface_hub','pyyaml':'yaml',
                'typing-extensions':'typing_extensions','math-verify':'math_verify',
                'latex2sympy2-extended':'latex2sympy2_extended'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requirements',type=Path,default=Path(__file__).resolve().parents[1]/'requirements.txt')
    p.add_argument('--require-cuda',action='store_true')
    args = p.parse_args(argv)
    if sys.version_info < (3,12):
        raise RuntimeError('production Python >=3.12 required, same interpreter gate as SpecNaacl')
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
            # Server uses 0.21.1; 0.21.2 has the same APIs for this causal-LM
            # pipeline. Install reference remains exactly pinned to 0.21.1.
            allowed = {'0.21.1','0.21.2'} if name == 'peft' and expected=='0.21.1' else {expected}
            if actual not in allowed:
                raise RuntimeError(f'expected {expected}, found {actual}')
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
    if args.require_cuda and (not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10,0)):
        raise RuntimeError('production B200-class CUDA compute capability >=10.0 required')
    print(f'Pure GRPO environment validation passed: {sys.executable}, torch={torch.__version__}, cuda={torch.version.cuda}')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError,ValueError) as error:
        raise SystemExit(str(error)) from None
