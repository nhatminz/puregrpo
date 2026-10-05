"""Shared telemetry implementation with only meaningful target-only counters."""
import csv
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist

COUNTERS=("wall_time_s","generation_time_s","target_train_time_s","rollout_tokens")
MEMORY_FIELDS=("gpu_allocated_gb","gpu_reserved_gb","gpu_peak_allocated_gb","gpu_free_gb")
STEP_FIELDS=("step","method","grpo_step","phase_time_basis")+tuple(f"{p}_{n}" for p in ("cumulative","step") for n in COUNTERS)+("step_generation_tokens_per_s","cumulative_generation_tokens_per_s")+MEMORY_FIELDS+("rollout_tokens","tokens_per_s","mean_reward","target_loss","optimizer_steps")


def aggregate(data, device, wall):
    sums = ('total_rollout_tokens', 'reward_sum', 'reward_count', 'target_loss_sum',
            'target_loss_count', 'used_items', 'trace_rollout_count', 'generated_samples',
            'ignore_due_correct', 'ignore_due_incorrect')
    maxima = ('generate_time_cost', 'train_time_cost', 'optimizer_steps')
    a = torch.tensor([float(data.get(k, 0)) for k in sums], device=device, dtype=torch.float64)
    b = torch.tensor([float(data.get(k, 0)) for k in maxima] + [wall], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        dist.all_reduce(b, op=dist.ReduceOp.MAX)
    result = dict(zip(sums, a.tolist()))
    result.update(zip(maxima + ('cumulative_elapsed_time_s',), b.tolist()))
    result['mean_reward'] = result['reward_sum'] / max(result['reward_count'], 1.)
    result['target_loss'] = result['target_loss_sum'] / max(result['target_loss_count'], 1.)
    # Match SpecNaacl's legacy tokens_per_s (ALL rollout tokens / job wall).
    result['tokens_per_s'] = result['total_rollout_tokens'] / max(result['cumulative_elapsed_time_s'], 1e-9)
    return result


def completed_step_snapshot(global_metrics, data, timings, device, wall_time_s):
    durations = timings.resolve()  # after aggregate's existing scalar transfer
    data['_phase_target_time_s'] = durations['target']
    memory = gpu_memory_stats(device)
    values = torch.tensor([wall_time_s, durations['target'], *[memory[k] for k in MEMORY_FIELDS[:3]],
                           -memory['gpu_free_gb']], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    wall, target, allocated, reserved, peak, negative_free = values.tolist()
    return dict(cumulative_wall_time_s=wall,
                cumulative_generation_time_s=float(global_metrics['generate_time_cost']),
                cumulative_target_train_time_s=target,
                cumulative_rollout_tokens=int(global_metrics['total_rollout_tokens']),
                phase_time_basis=timings.basis,
                **dict(zip(MEMORY_FIELDS, (allocated, reserved, peak, -negative_free))))


def step_record(step, current, previous, extras=None):
    result = dict(extras or {})
    result.update(current)
    result['step'] = int(step)
    for name in COUNTERS:
        field = f'cumulative_{name}'
        result[f'step_{name}'] = current[field] - previous.get(field, 0)
    for prefix in ('step', 'cumulative'):
        elapsed = result[f'{prefix}_generation_time_s']
        result[f'{prefix}_generation_tokens_per_s'] = result[f'{prefix}_rollout_tokens'] / elapsed if elapsed > 0 else 0.
    result.update(rollout_tokens=result['cumulative_rollout_tokens'],
                  tokens_per_s=result['cumulative_rollout_tokens'] / max(result['cumulative_wall_time_s'], 1e-9))
    return result

class PhaseTimings:
    """CUDA stream elapsed intervals; CPU monotonic durations without CUDA.

    CUDA events are read only after the existing metric scalar transfer has
    completed the main stream. There is no synchronize/wait in this class.
    """
    def __init__(self, device, target_s=0.0):
        self.device = torch.device(device)
        self.totals = {'target': float(target_s)}
        self.pending = []
        self.basis = 'cuda_stream_elapsed' if self.device.type == 'cuda' else 'cpu_monotonic'

    def begin(self, phase):
        started = time.perf_counter()
        event = None
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        return phase, started, event

    def end(self, ticket):
        phase, started, event = ticket
        ended = None
        if event is not None:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record(torch.cuda.current_stream(self.device))
        self.pending.append((phase, time.perf_counter() - started, event, ended))

    def resolve(self):
        for phase, host_s, started, ended in self.pending:
            if ended is not None:
                if not ended.query():
                    raise RuntimeError('phase timings must be resolved after the metric scalar transfer')
                elapsed = started.elapsed_time(ended) / 1000.0
            else:
                elapsed = host_s
            self.totals[phase] += elapsed
        self.pending.clear()
        return self.totals.copy()

def gpu_memory_stats(device):
    """Allocator snapshots in GiB; no synchronization, cache flush or reset."""
    device = torch.device(device)
    if device.type != 'cuda':
        return dict.fromkeys(MEMORY_FIELDS, 0.0)
    free, _ = torch.cuda.mem_get_info(device)
    values = (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
              torch.cuda.max_memory_allocated(device), free)
    return {name: value / (1024 ** 3) for name, value in zip(MEMORY_FIELDS, values)}

class StepMetricsWriter:
    """One row per inherited GRPO label, including repeated optimizer updates.

    FastGRPO can perform several optimizer updates with the same `step` label.
    Replace the pending snapshot until the label advances; then write it once.
    The final label is flushed at normal/short-run termination. State is saved
    with the existing per-rank batch_data checkpoint.
    """
    def __init__(self, jsonl_path, csv_path, *, enabled=True, append=False, baseline=None, state=None):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.enabled = enabled
        self.previous = dict(baseline or {})
        self.pending = None
        self.last_emitted = None
        if state is not None:
            self.previous = dict(state['previous'])
            self.pending = state.get('pending')
            self.last_emitted = state.get('last_emitted')
            if self.pending is None and self.last_emitted is not None:
                # A completed short run may resume while the inherited label
                # still repeats. Reopen that label and retain its full interval.
                self.previous = dict(self.last_emitted['baseline'])
                self.pending = {key: self.last_emitted[key] for key in ('step', 'snapshot', 'extras')}
        if enabled:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if not append or not self.csv_path.exists():
                with self.csv_path.open('w', newline='', encoding='utf-8') as stream:
                    csv.DictWriter(stream, fieldnames=STEP_FIELDS).writeheader()
            else:
                self._upgrade_csv_header()
                if self.pending is not None:
                    self._rewind_step_rows(self.pending['step'])

    def _rewind_step_rows(self, step):
        """Discard only telemetry past the restored checkpoint, with backups."""
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        kept = [row for row in rows if int(row['step']) < step]
        if len(kept) != len(rows):
            backup = self.csv_path.with_suffix('.pre_resume.csv')
            if not backup.exists():
                backup.write_bytes(self.csv_path.read_bytes())
            temporary = self.csv_path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(kept)
            temporary.replace(self.csv_path)
        if self.jsonl_path.exists():
            lines = self.jsonl_path.read_text(encoding='utf-8').splitlines(keepends=True)
            retained = []
            for line in lines:
                row = json.loads(line)
                if row.get('phase') == 'target_train' and int(row.get('step', -1)) >= step:
                    continue
                retained.append(line)
            if len(retained) != len(lines):
                backup = self.jsonl_path.with_suffix('.pre_resume.jsonl')
                if not backup.exists():
                    backup.write_bytes(self.jsonl_path.read_bytes())
                temporary = self.jsonl_path.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(retained), encoding='utf-8')
                temporary.replace(self.jsonl_path)

    def _upgrade_csv_header(self):
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) == STEP_FIELDS:
                return
            rows = list(reader)
        # Existing values survive schema migration; unavailable historical
        # per-step fields remain blank rather than being reconstructed/guessed.
        backup = self.csv_path.with_suffix('.pre_step_metrics.csv')
        if not backup.exists():
            backup.write_bytes(self.csv_path.read_bytes())
        temporary = self.csv_path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.csv_path)

    def advance(self, step):
        if self.pending is not None and int(step) != self.pending['step']:
            if int(step) < self.pending['step']:
                raise ValueError('GRPO step labels must be monotonic')
            self.flush()

    def submit(self, step, snapshot, extras):
        self.advance(step)
        self.pending = {'step': int(step), 'snapshot': dict(snapshot), 'extras': dict(extras)}

    def flush(self):
        if self.pending is None:
            return None
        row = step_record(self.pending['step'], self.pending['snapshot'], self.previous,
                          self.pending['extras'])
        if self.enabled:
            with self.jsonl_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            with self.csv_path.open('a', newline='', encoding='utf-8') as stream:
                csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore').writerow(row)
        self.last_emitted = {**self.pending, 'baseline': self.previous.copy()}
        self.previous = dict(self.pending['snapshot'])
        self.pending = None
        return row

    def state_dict(self):
        return {'previous': self.previous.copy(), 'pending': self.pending,
                'last_emitted': self.last_emitted}
