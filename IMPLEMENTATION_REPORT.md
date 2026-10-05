# Pure GRPO implementation report — 2026-10-05

## Delivered scope

New sibling `puregrpo/`, no edits to SpecNaacl. Source commit and fingerprints
are in SOURCE_PROVENANCE.json; tests verify original source fingerprints and
byte-identical inherited loss/collator/helper functions.

Target-only generation uses the real public HF decoder + LM head and ordinary
target DynamicCache. The target sees one prompt prefill and width1 decode
forwards thereafter. Every response token is sampled from the inherited full
target distribution; no draft/checkpoint, tree/verifier, Reflex/LK/Sparse,
SpecForge/SGLang serving or proposal/acceptance code is imported/called.

Inherited without semantic rewrite: reward functions/weight0.2, dataset loader/
subset/order, math chat collator, GRPO selected-token logps/clipped ratio/KL,
reference via disabled LoRA, response-mask reduction, packing, AdamW defaults,
LoRA r64/alpha32/dropout0/target modules, constant LR/no scheduler, variance0
filtering, distributed manual gradient averaging and optimizer cadence.
Shared config/CLI defaults and all7 model launchers are checked against Source.

Removed entirely: EAGLE weights/config/mapping checks, feature extraction,
online draft trainer/optimizer, draft KV, verification masks/tree/acceptance,
concurrency-aware speculative scheduler, Reflex updates and policy-lag branches.
Pure has no AAL/acceptance/draft telemetry or checkpoint fields.

## Fairness and unavoidable differences

See README.md for explicit Source limitations: accumulation controls step labels
but not deferred optimizer updates; legacy length-sort/reward association is
preserved even though it can be a correctness issue. Do not fix just Pure.
Source duplicates a sampled prefill root; Pure independently samples all roots
as requested, without emulating speculative-only coupling/chunk overshoot.
Removing draft construction changes RNG consumption before LoRA creation: strict
comparison requires a shared TARGET_ADAPTER initialization, not just equal seeds.
Sampler distribution/config is shared, but generated responses need not match.

Periodic eval defaults0 because Source has no ordinary periodic target eval.
Optional held-out evaluation restores main RNG/module modes and never enters
training replay. Fair checker flags nonzero top-k/eval frequency as unmatched.
Training loss/reward mean and generation/job-wall throughput aliases keep Source
denominators; generation-only throughput has its explicit cumulative/step name.
Target-only checkpoint format is distinct and rejects foreign resume payloads.

Output/checkpoint parent defaults to sibling SpecNaacl/outputs, preserving the
requested resource path. New unique method-puregrpo run names and method-specific
active/latest links prevent collisions; no old checkpoint/code/link is changed.
OUTPUT_ROOT/RUN_DIR overrides remain available. Source code is not needed at
runtime; only comparison/audit tools require it.

## Files created

- Entrypoints: grpo.py, training.py, evaluation.py.
- Target/core: helper/autoregressive.py, target.py, target_checkpoint.py,
  grpo_core.py, train_ops.py, sampling.py, rewards.py, get_QAs.py,
  checkpointing.py, metrics.py, __init__.py.
- Config: configs/_shared/b200_common.env and seven model b200.env files.
- Train/eval wrappers for qwen25_1p5b, qwen25_3b, qwen25_7b, qwen25_14b,
  qwen3_1p7b, qwen3_4b, llama31_8b.
- scripts/launch/train_model.sh, run_puregrpo.sh, evaluate.sh, smoke_test.sh,
  benchmark_pair.sh, check_fair_config.py, prepare_shared_target_adapter.py,
  validate_environment.py, write_run_metadata.py, plot_training_time.py.
- Tests: fixtures/conftest plus autoregressive, GRPO parity, pipeline/checkpoint/
  resume/evaluation, environment and launch/source tests; pytest.ini.
- requirements.txt, LICENSE, SOURCE_PROVENANCE.json, README.md,
  huongdanchay.md, this report. No models/datasets/generated benchmark numbers.

## Validation / limits

- Final CPU-capable environment: **39 passed**. Final real-CUDA environment:
  **48 passed,1 skipped** (only real HF Qwen2 test skipped because Transformers
  is absent in that CUDA venv; it passed in the CPU-capable venv).
- `python -m compileall -q .`: pass. `bash -n` all19 shell files: pass.
  CLI config validation, all7 paired launcher defaults/overrides, direct-CLI
  parity and bounded benchmark dry-run: pass. CPU-capable venv `pip check`:
  **No broken requirements found**. Source git status clean after validation.

Validation uses two existing local environments, not the pinned B200 stack:
CPU-capable Torch2.11+cu130/Transformers5.9 (CUDA unavailable with local driver),
and real RTX3090 CUDA Torch2.5.1+cu124 (without Transformers/PEFT).
CPU tests include a real tiny HF Qwen2 model created from config, with actual
DynamicCache and cached-vs-full-prefix target greedy parity. This is not a
pretrained alternative inserted into the production pipeline.

Tests cover sampler bitwise parity to original source; target-only import guard;
one-token cached forwards; left padding, EOS/all-finished/compaction/real-length
limit; direct root sampling; source-identical loss/gradients/AdamW state for
multiple iterations; actual tiny-target collect→GRPO→checkpoint→held-out eval→
JSONL/CSV/summary; target unchanged in eval; RNG/module-mode restoration;
mid-epoch resume policy/token-count parity; no AAL; seven-model default/override
config parity and direct-CLI default parity; syntax/config validation.

No B200 production model/dataset paths or local PEFT installation are available.
Full pretrained HF+PEFT training, multi-rank NCCL, server pinned dependency imports,
real-model smoke and the production paired benchmark have NOT run. No full
training was launched and no speedup/reward result is claimed. The supplied
server smoke/benchmark commands perform those checks explicitly. Local pip check
is clean but does not certify the separate server environment. Test target doubles
are injected only in tests; production requires CUDA and real configured assets.
