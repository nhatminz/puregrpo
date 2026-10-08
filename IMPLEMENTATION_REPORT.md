# Pure GRPO fairness update — 2026-10-08

## Current revision and scope

Read-only Source: SpecNaacl commit `8366844b52c70835768dba57e06bda149a65f5d9`.
No Source files changed. Current audit is in FAIRNESS_AUDIT.json; the original
SOURCE_PROVENANCE.json below remains a historical record, not a latest-source
identity claim. No new Pure performance optimization or GRPO semantic rewrite.

Changed config: common dataset gsm8k / wrappers dapo → simplelr (same Source
parquet), wrapper target LR1e-5 →1e-6, eager/empty attention selection →sdpa.
Both train/eval wrappers for all seven models are synchronized. Unchanged:
batch8/accum4/responses8, max_length/max_prompt_length2048, temp1/top_p0.95,
bf16, beta0.04/epsilon0.1, seed42, epoch1, sample_num100, packing1024/gap4096,
reward/template/masking, AdamW defaults, constant LR, reference and update cadence.
Direct Python CLI keeps Source's direct-CLI defaults (distinct from wrappers).
No model/tokenizer/dataset/output/checkpoint paths were moved.

Runtime installation references now match the current Source common pins:
Torch2.8.0 / Transformers4.51.3 / PEFT0.17.1. Bounded compatibility admission
matches Source; use the SAME validated interpreter and CUDA runtime for all
methods. No installed packages were changed. New probe uses only a tiny target.
The only autoregressive code change is DynamicCache constructor API compatibility
(`DynamicCache()` on4.51, config-aware constructor on newer versions). No static,
persistent, remapped KV, tree, Triton, asynchronous stream or proposal code added.

Shared target initialization: prepare script saves one LoRA adapter plus artifact
hash/runtime manifest. Pure verifies every loaded adapter tensor once at startup;
different base-model manifests and corrupt/incompatible adapters fail clearly.
No trainable parameters/architecture were added. Three-way benchmark requires
TARGET_ADAPTER, shares interpreter/subset/settings/nominal step budget, checks
pip metadata before training, and calls existing Source launchers without edits.

## Mathematical comparison and disclosed differences

Pure's chunked GRPO loss, reward/advantage filtering, packing and training loop
were not edited. Current Source uses full logits. Tests compare the real current
Source formula/wrapper to Pure loss, gradients and AdamW state at FP32 tolerance.
For nonempty masks: same clipped surrogate + reverse-exp KL, per-response valid
position mean then response sum, population-std advantages, weight0.2 format
reward + accuracy, zero-variance filtering, disabled-LoRA base reference and
manual distributed gradient averaging. No one-sided mathematical changes.

Actual differences preserved and reported (not hidden by a config PASS):

- Pure autoregressive independent response-root sampling vs speculative root
  coupling/verification/overshoot; no AAL exists for Pure.
- Pure's existing FP32 sampler vs historical FastGRPO logits-dtype probabilities;
  invalid-row fallback and DataLoader-generator RNG consumption also differ.
- All-zero mask: Pure clamps denominator, Source can produce NaN. Normal valid
  masks are equivalent; no algorithm fix was made on just one side.
- Both retain the legacy length-sort/unsorted-advantage association. Both update
  whenever retained buffers are ready; accumulation_steps controls step labels,
  not deferred optimizer updates. Same max_grpo_steps is a shared nominal label
  budget, NOT proof of equal actual updates after stochastic reward filtering.
  Audit retained groups/actual updates; Source does not expose a total actual
  optimizer-step counter in its current summary. This cannot be corrected in
  Source under the user's no-Source-edits/no-algorithm-change restriction.

The one dataset correction is matching Source's string vs array prompt parsing:
NumPy-array parquet prompt now yields its content, not an array string. Dataset
order/subset selection, tokenizer and template are unchanged.

## Current validation

- Fair checker: PASS, **14 default comparisons** (7 models × fastgrpo/opd_reflex),
  plus environment-override, LoRA recipe, common CLI budget, dataset/reward,
  prompt collator, explicit AdamW settings and path checks.
- Tests: **72 passed** with Transformers4.51.3/PEFT0.17.1; **72 passed** with
  Transformers5.12.1/PEFT0.21.1. Both used existing Torch2.5.1+cu124/CUDA12.4,
  Python3.10, RTX3090-capable local runtime; no dependencies installed/upgraded.
- Tests include actual tiny HF target + PEFT save/load through the preparation
  script and production loader, different RNG before all three target recipes,
  exact LoRA tensor equality, unchanged trainable parameter inventory, target-only
  cached decoding, CPU/CUDA gradients, checkpoint/resume/eval and metric export.
- Target-only SDPA/PEFT/DynamicCache environment probe: CPU/CUDA PASS.
- compileall, all shell syntax, CLI validation, three-way benchmark DRY_RUN and
  diff checks PASS. Source git status remains clean. Core loss/training/sampling/
  metrics/evaluation hashes match the pre-edit Pure files.
- Local `pip check` in the existing4.51 overlay FAILED with four pre-existing
  conflicts: datasets/dill, datasets/fsspec, datasets/multiprocess and
  aiohttp/multidict. This is not a validated complete installation; passing
  runtime/unit probes does not resolve its package metadata. No package changes
  were attempted. Real server must pass pip check in its shared environment.
- Production B200 model/dataset training and multi-rank benchmark NOT run; local
  tiny test fixtures are not substitute production weights or measured results.
  No throughput/speedup/reward claim.

## Files changed for this update

configs/_shared/b200_common.env; all seven train_*.sh and eval_*.sh wrappers;
grpo.py; helper/get_QAs.py, target.py, shared_adapter.py, autoregressive.py
(constructor compatibility only); requirements.txt; scripts/check_fair_config.py,
benchmark_pair.sh, prepare_shared_target_adapter.py, validate_environment.py;
FAIRNESS_AUDIT.json; tests/fixtures.py, test_autoregressive.py, test_environment.py,
test_grpo_parity.py, test_launch_and_sources.py, test_shared_initialization.py;
README.md, huongdanchay.md and this report.

---

## Historical creation report — 2026-10-05 (superseded validation/config)

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
