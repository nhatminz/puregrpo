# Pure GRPO target-only baseline

Standalone sibling of SpecNaacl. No code import, symlink or runtime dependency
on its draft/SpecForge/tree/verifier/Reflex components. Source inheritance is
recorded in SOURCE_PROVENANCE.json. SpecNaacl sources are not modified.

Only rollout changes: ordinary target-only HF DynamicCache, one-token target
forwards, inherited FP32 temperature/top-p/top-k probabilities and multinomial.
Prefix KV is shared before response expansion; every response samples its own
first token. EOS trajectories leave the batch; active-batch real-length stopping
matches Source, without speculative chunk overshoot/tree padding.

Reward functions, collator and token logps are preserved. The parquet prompt
parser now matches current Source for NumPy-array prompts. Pure's existing
chunked GRPO loss is kept, not replaced by Source's newer full-logits loss;
tests compare formulas, gradients and AdamW states within FP32 tolerance.
LoRA configuration, AdamW defaults, constant LR/no scheduler, variance filtering,
packing, distributed gradient averaging and optimizer cadence match Source.
All7 train_<model>.sh launchers preserve their common target settings, including
SimpleLR/batch8/accum4/LR1e-6/responses8/bf16/sdpa. Generic launchers and direct Python CLI
retain their respective Source defaults, which differ from per-model wrappers;
do not mix invocation styles. scripts/check_fair_config.py defaults to checking
all seven models against BOTH fastgrpo and opd_reflex, plus the target LoRA recipe.
FAIRNESS_AUDIT.json records the current inspected Source revision and unchanged
Pure execution files; SOURCE_PROVENANCE.json remains historical provenance.
DynamicCache constructor selection is API compatibility only (4.51 vs newer HF),
not a static/persistent/custom cache or decoding optimization.

Output/checkpoint parent defaults to ../SpecNaacl/outputs as requested. Unique
method-puregrpo run names and active_run_puregrpo/latest_run_puregrpo links never
overwrite Source's runs or resume links. All explicit OUTPUT_ROOT/RUN_DIR
overrides, model/tokenizer/dataset paths are preserved. Runtime does not require
SpecNaacl code; only comparison/audit scripts need the sibling. No weights/data
are copied into this folder.

No AAL/acceptance/draft metrics or checkpoint tensors are fabricated. Common
JSONL/CSV uses inherited PhaseTimings/StepMetricsWriter and counter differences.
step/cumulative_generation_tokens_per_s divides ALL generated tokens (including
EOS and filtered groups) by generation time. Legacy tokens_per_s and summary
generation_tokens_per_s divide by JOB WALL, matching Source; use the cumulative
generation field for generation-only throughput. Reward/target_loss retains the
Source trained-response denominator, not unfiltered eval samples. Memory is GiB;
peaks are job-to-date. Final summary wall includes target/checkpoint I/O.

## Inherited behavior and comparison limits

- accumulation_steps controls Source step labels (used_groups // (batch*accum)),
  but optimizer updates occur whenever all ranks have a nonempty retained buffer.
  Pure preserves this, not standard deferred accumulation. One label can include
  several updates; Pure additionally records actual optimizer_steps.
- Source length-sorts token/mask rows but indexes advantages in original response
  order. This legacy association is preserved. It may be a GRPO correctness
  issue; a fix must be applied to BOTH methods before a new fair comparison,
  not silently to Pure alone.
- Draft construction consumes RNG before Source LoRA initialization. LoRA B=0
  gives the same initial distribution but LoRA A can differ and affect training.
  Strict paired benchmark therefore requires the SAME TARGET_ADAPTER. A
  target-only shared-initialization script is supplied; no draft is loaded.
  It saves an artifact hash and runtime manifest. Pure checks every loaded LoRA
  tensor against that artifact at initialization, outside generation/training.
- Latest historical FastGRPO samples in its logits dtype, whereas Pure retains
  its pre-existing FP32 probability implementation. Temperature/top-p/no-top-k
  match, but reduced-precision distributions/tokens need not be bitwise identical.
  This is a reported implementation difference, not a Pure-only sampler change.
  Source's seeded DataLoader generator and Pure's existing global generator also
  consume RNG differently; shared adapter tensors do not imply matched responses.
  Invalid-probability fallback also has different RNG consumption (Source draws
  only valid rows; Pure retains its existing EOS fallback draws for all rows).
- For an all-zero response mask Pure's existing denominator clamp returns zero,
  while current Source's unguarded division can return NaN. Normal retained
  nonempty responses have the same mathematical loss; no one-sided fix was made.
- Source duplicates a sampled prefill root across responses; Pure samples EACH
  response root directly, as requested. This speculative-prefix coupling is not
  emulated. Same seed never guarantees identical generated responses.
- Source has no ordinary periodic target evaluation; disable policy-lag branches
  when benchmarking the production methods.
  Pure EVAL_INTERVAL=0 matches it. Optional eval restores RNG/module modes and
  never enters replay. Fair checker rejects nonzero eval_interval or top_k:
  Source training currently forwards top_k=None.
- Resume schema is target-only; cannot resume a full Source checkpoint. Use its
  exported target HF adapter as TARGET_ADAPTER instead. Resume Pure with the
  saved world size; no draft assets are read.

See huongdanchay.md for commands and IMPLEMENTATION_REPORT.md for validation.
