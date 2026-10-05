"""Held-out target-only evaluation; no optimizer, adapter update or replay input."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from helper.autoregressive import autoregressive_generate
from helper.grpo_core import TrainDataCollator, _seed_everything
from helper.metrics import gpu_memory_stats


def load_eval_rows(args):
    from helper.get_QAs import get_QAs_from_path
    if not args.eval_dataset_path:
        raise ValueError('explicit held-out eval_dataset_path required; training parquet is not silently reused')
    if (Path(args.eval_dataset_path).is_file() and
            Path(args.eval_dataset_path).resolve() == Path(args.dataset_path).resolve()):
        raise ValueError('the training FILE is not a held-out evaluation split')
    rows = get_QAs_from_path(args.eval_dataset_path, args.eval_split)
    if not rows:
        raise ValueError('evaluation split is empty')
    return rows


@torch.inference_mode()
def evaluate(args, target, tokenizer, *, rows=None, reward_functions=None, step=None,
             generate=autoregressive_generate):
    rows = load_eval_rows(args) if rows is None else rows
    if reward_functions is None:
        from helper.rewards import format_reward_func, accuracy_reward_func
        reward_functions = format_reward_func, accuracy_reward_func
    format_fn, answer_fn = reward_functions
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    ids = list(range(rank, len(rows), world))  # no duplicated padding samples
    target.eval()
    _seed_everything(args.seed + rank)
    collator = TrainDataCollator(tokenizer, args.max_prompt_length)
    records, wall, tokens, forwards = [], 0., 0, 0
    device = target.device
    for offset in range(0, len(ids), args.batch_size):
        batch_ids = ids[offset:offset+args.batch_size]
        batch = collator([rows[i] for i in batch_ids])
        if batch['input_ids'].shape[-1] >= args.max_length:
            raise ValueError('evaluation prompt consumes max_length')
        output = generate(target, batch['input_ids'].to(device), batch['attention_mask'].to(device), tokenizer,
                          max_length=args.max_length, repeated_generate_nums=args.repeated_generate_nums,
                          temperature=args.temperature, top_p=args.top_p, top_k=args.top_k or None,
                          statistical_time=args.statistical_time)
        wall += output['total_time_cost']
        forwards += output['target_prefill_forwards'] + output['target_decode_forwards']
        for index, generated in enumerate(output['generated_token_ids']):
            prompt = batch_ids[index // args.repeated_generate_nums]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            reward = .2*format_fn([text])[0] + answer_fn([text],[rows[prompt]['answer']])[0]
            records.append(dict(prompt_id=prompt, response_index=index%args.repeated_generate_nums,
                                seed=args.seed, generated_tokens=len(generated), reward=reward,
                                completion=text, token_ids=generated))
            tokens += len(generated)
    packed = dict(rows=records, wall=wall, tokens=tokens, forwards=forwards, memory=gpu_memory_stats(device))
    gathered = [None]*world
    if dist.is_initialized():
        dist.all_gather_object(gathered, packed)
    else:
        gathered[0] = packed
    if rank:
        return
    records = [r for item in gathered for r in item['rows']]
    wall = max(item['wall'] for item in gathered)
    tokens = sum(item['tokens'] for item in gathered)
    report = dict(phase='evaluation', method='puregrpo', step=step, seed=args.seed,
                  generated_samples=len(records), generated_tokens=tokens, generation_wall_time_s=wall,
                  generation_tokens_per_s=tokens/max(wall,1e-9),
                  mean_reward=sum(r['reward'] for r in records)/max(len(records),1),
                  target_forwards=sum(item['forwards'] for item in gathered))
    for key in gathered[0]['memory']:
        reduce = min if key == 'gpu_free_gb' else max
        report[key] = reduce(item['memory'][key] for item in gathered)
    root = Path(args.summary_file).parent / 'evaluation'
    root.mkdir(parents=True,exist_ok=True)
    tag = f'step{step}' if step is not None else 'final'
    (root/f'{tag}_per_response.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (root/f'{tag}_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report
