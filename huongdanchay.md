# Chạy Pure GRPO trên B200

Đặt puregrpo/ cạnh SpecNaacl/. Giữ venv đã chạy được SpecNaacl; Pure không cần
SpecForge, SGLang, cuda-tile, yunchang hay pretrained draft. Requirement reference
giữ Torch2.13.0/cu130, Transformers5.12.1 như Source và PEFT0.21.1 theo server
hiện có (validator chấp nhận0.21.2 với cùng API). Không đổi venv khi so sánh.

```bash
cd /workspace/storage-shared/nlp/minhpn19/puregrpo
source ../SpecNaacl/.venv/bin/activate
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
```

Nếu environment mới, dùng Python>=3.12 và Torch/CUDA wheels đúng stack Source,
cài requirements từ wheelhouse thật đã có trên server (WHEELHOUSE phải được set):

```bash
python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt
```

## Train

```bash
CUDA_VISIBLE_DEVICES=5 DATASET=simplelr bash train_qwen25_1p5b.sh
CUDA_VISIBLE_DEVICES=0 DATASET=dapo bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=0 DATASET=dapo bash train_qwen3_1p7b.sh
CUDA_VISIBLE_DEVICES=0 DATASET=dapo bash train_qwen3_4b.sh
```

Cũng có train_qwen25_7b.sh, train_qwen25_14b.sh, train_llama31_8b.sh; mỗi model
có eval_<model>.sh tương ứng. Model/dataset paths giữ từ Source. Biến đúng là
DATASET, không phải DATSET. Per-model defaults:

- TARGET_LR=1e-5, BATCH_SIZE=8, ACCUMULATION_STEPS=4, RESPONSES_PER_PROMPT=8.
- NUM_EPOCHS=1, GEN_MAX_LENGTH=2048, MAX_PROMPT_LENGTH=2048.
- TEMPERATURE=1, TOP_P=0.95, beta0.04, epsilon0.1, bf16/eager, seed42.
- MAX_TRAINING_TOKEN=1024, MAX_TRAINING_PADDING_GAP=4096, LOGPS_CHUNK_SIZE=256.
- LOG_INTERVAL=1, SAVE_CHECKPOINT_STEPS=100, KEEP_LAST_CHECKPOINTS=3.

Override bằng env/CLI như Source: MODEL, TARGET_ADAPTER, DATASET_PATH,
OUTPUT_ROOT, RUN_DIR, RESUME, CUDA_VISIBLE_DEVICES/NPROC_PER_NODE và hyperparams.
Generic launch/direct Python defaults khác per-model, đúng như Source; so sánh
đúng cùng invocation style hoặc chạy check_fair_config.py trước.

```bash
DRY_RUN=true CUDA_VISIBLE_DEVICES=5 bash train_qwen25_3b.sh
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 TARGET_LR=1e-5 \
BATCH_SIZE=8 ACCUMULATION_STEPS=4 bash train_qwen25_3b.sh
RESUME=auto CUDA_VISIBLE_DEVICES=0 bash train_qwen25_3b.sh
```

## So sánh fair với SpecNaacl

Read-only check, không load weights/train:

```bash
CUDA_VISIBLE_DEVICES=5 DATASET=simplelr \
"$PYTHON_BIN" scripts/check_fair_config.py --model-key qwen25_1p5b
```

Dùng cùng target LoRA initialization r64/alpha32/base model. Có thể dùng adapter
đã có; hoặc tạo MỘT LẦN bằng target-only script, không load draft:

```bash
export TARGET_ADAPTER="/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/target_initializations/qwen25_3b_seed42_$(date -u +%Y%m%dT%H%M%S_%N)"
"$PYTHON_BIN" scripts/prepare_shared_target_adapter.py \
  --model_dir /workspace/storage-shared/models/Qwen2.5-3B-Instruct \
  --seed 42 --output "$TARGET_ADAPTER"
```

Set TARGET_ADAPTER giống nhau cho cả hai train commands. Benchmark helper dùng
cùng subset128 prompts và 1 epoch mặc định, không thay defaults của train thường.
Tăng BENCHMARK_SAMPLES và lặp TRAIN_SUBSET_SEED để đo nhiều seeds. Source phải
chạy được trong cùng venv; helper không sửa code hoặc bỏ qua Source preflight.

```bash
DRY_RUN=true MODEL_KEY=qwen25_3b CUDA_VISIBLE_DEVICES=0 bash scripts/benchmark_pair.sh
# Chỉ chạy khi TARGET_ADAPTER chung đã chuẩn bị:
MODEL_KEY=qwen25_3b CUDA_VISIBLE_DEVICES=0 BENCHMARK_SAMPLES=128 bash scripts/benchmark_pair.sh
```

Checklist:

1. Cùng target/tokenizer/TARGET_ADAPTER, versions và GPU/distributed topology.
2. Config check pass; cùng dataset/subset/seed/epoch/prompt ordering.
3. Profiling OFF, top-k=None, EVAL_INTERVAL=0 (Source không có periodic eval).
4. Cùng target objective/LR/AdamW/packing; log actual optimizer steps/retained
   groups vì Source's step label không phải optimizer-step counter.
5. Đúng generation denominator và weighted reward/loss; không dùng AAL và không
   assume cùng seed = cùng responses. Ghi samples/tokens generated và memory.
6. Giữ cold-start/JIT và wall conventions; báo nhiều seeds, không chỉ elapsed.

## Evaluate / smoke / tests

EVAL_DATASET_PATH phải là held-out split thật. Ví dụ SimpleLR có test.parquet
đúng convention Source; chỉ chạy nếu file đó thật sự tồn tại. Lấy adapter target
từ summary của Pure run đã train, không nhập checkpoint/model path giả:

```bash
PURE_RUN="$(readlink -f ../SpecNaacl/outputs/train/qwen25_1p5b/latest_run_puregrpo)"
export TARGET_ADAPTER="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["saved_model_dir"])' "$PURE_RUN/summary.json")"
EVAL_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet \
CUDA_VISIBLE_DEVICES=5 bash eval_qwen25_1p5b.sh

# Smoke nhỏ với REAL model/data; chỉ smoke mới đổi budget:
MODEL_KEY=qwen25_3b CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test.sh

python -m compileall -q .
pytest -q
python -m pip check
```

Không chạy full training trong tests. Test target doubles và HF Qwen2 nhỏ từ
config chỉ kiểm tra implementation/API, không thay model/data production.
Nếu smoke không có group reward variance>0, Source filtering có thể không update;
tăng sample/budget, không fake reward. Optional periodic EVAL_INTERVAL>0 cần split
disjoint, restore RNG/modes và không vào replay; không coi setup đó là matched
với Source mặc định.

## Outputs

Giữ parent SpecNaacl theo yêu cầu, không đè run/checkpoint/link cũ:

```text
../SpecNaacl/outputs/train/<model_key>/<unique_method-puregrpo_run>/
  config_resolved.yaml
  logs/{console.log,metrics.jsonl,timing.csv}
  checkpoints/target/stepN/
  checkpoints/resume/{latest.pt,step*.pt}
  statistics/
  summary.json, summary.txt
  evaluation/{stepN|final}_{summary.json,per_response.jsonl}
```

OUTPUT_ROOT override được nếu muốn tách nơi lưu. Resume/latest links có suffix
_puregrpo, không dùng Source's active_run/latest_run. Benchmark output:
`../SpecNaacl/outputs/benchmarks/pure_vs_spec_<key>_<timestamp>/`, hai subfolders
specnaacl/puregrpo và training_time.png.
