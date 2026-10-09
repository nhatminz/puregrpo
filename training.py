"""Inherited GRPO training schedule, with target-only rollout and checkpoints."""
import json
import os
from pathlib import Path
from statistics import mean, stdev
import time
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from helper.grpo_core import _seed_everything, TrainDataCollator
from helper.autoregressive import autoregressive_generate
from helper.train_ops import accumulate_groups, update_policy
from helper.metrics import PhaseTimings, StepMetricsWriter, aggregate, completed_step_snapshot
from helper.target_checkpoint import save_checkpoint, load_checkpoint
from helper.target import load_target


def initial_data():
    data = {key: 0 for key in ('used_items','generate_time_cost','train_time_cost',
        'total_rollout_tokens','generated_samples','trace_rollout_count','reward_sum',
        'reward_count','target_loss_sum','target_loss_count','optimizer_steps',
        'ignore_due_correct','ignore_due_incorrect')}
    data.update({key: [] for key in ('messages','rewards','std_rewards','response_metadata','length_stdev',
                                    'length_range','length_cv','generate_length_list')})
    return data


def train(args, *, target=None, tokenizer=None, train_rows=None, reward_functions=None,
          device=None, generate=autoregressive_generate):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError('production training requires CUDA; use pytest for CPU validation')
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)  # dependency-injected tests only
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank() if dist.is_initialized() else 0
    main_rank = rank == 0
    max_optimizer_steps=getattr(args,'max_target_optimizer_steps',0)
    max_prompts=getattr(args,'max_rollout_prompts',0)
    if max_prompts and max_prompts % world_size:
        raise ValueError('max_rollout_prompts must be divisible by world_size')
    local_prompt_budget=max_prompts//world_size
    _seed_everything(args.seed)
    if target is None:
        target, tokenizer = load_target(args, device)
    model = SimpleNamespace(target_model=target)  # no draft attributes/constructor
    if args.eval_only:
        if args.resume_checkpoint:
            state = torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False)
            if state.get('format') != 'puregrpo_target_checkpoint_v1':
                raise ValueError('evaluation resume requires a Pure GRPO checkpoint; otherwise use TARGET_ADAPTER')
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(target,state['target_lora'])
        from evaluation import evaluate
        report = evaluate(args, target, tokenizer, reward_functions=reward_functions, generate=generate)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return report
    if train_rows is None:
        from helper.get_QAs import get_QAs_from_path, select_train_subset
        all_rows = get_QAs_from_path(args.dataset_path, args.train_split)
        train_rows = select_train_subset(all_rows, fraction=args.train_data_fraction,
                                        max_samples=args.max_train_samples, seed=args.train_subset_seed)
        full_size = len(all_rows)
    else:
        full_size = len(train_rows)
    if not train_rows:
        raise ValueError('training split is empty')
    if args.eval_interval:
        from evaluation import load_eval_rows
        evaluation_rows = load_eval_rows(args)
        overlap = {str(x['question']).strip() for x in train_rows} & {str(x['question']).strip() for x in evaluation_rows}
        if overlap:
            raise ValueError('evaluation prompts overlap training; configure a disjoint eval_dataset_path')
    optimizer = torch.optim.AdamW(target.parameters(), lr=args.target_lr)
    optimizer.zero_grad(set_to_none=True)
    data = initial_data()
    step, start_epoch, start_batch, restored_wall = 0, 0, 0, 0.
    resume_rng = None
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, target=target, optimizer=optimizer)
        data.update(checkpoint['local_state']['batch_data'])
        step, start_epoch, start_batch = checkpoint['step'], checkpoint['epoch'], checkpoint['next_batch']
        restored_wall = checkpoint['cumulative_elapsed_time_s']
        resume_rng = checkpoint['local_state']['rng']
        if args.reset_rng_on_resume:
            _seed_everything(args.seed)
            resume_rng = None
    trace_start_step = step
    data['trace_rollout_count'] = 0  # inherited continuation trace definition
    append = bool(args.resume_checkpoint) if args.append_log == '' else str(args.append_log).lower() in {'true','1'}
    log_path = args.log_file if main_rank else os.devnull
    if main_rank:
        for path in (args.log_file, args.timing_file, args.summary_file):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        for path in (args.saved_model_dir, args.saved_statistics_dir, args.checkpoint_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a' if append else 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(vars(args), phase='run_config')) + '\n')
    started = time.perf_counter()
    def wall():
        return restored_wall + time.perf_counter() - started
    timings = PhaseTimings(device, target_s=data.get('_phase_target_time_s', data['train_time_cost']))
    def take_snapshot():
        values = aggregate(data, device, wall())
        snapshot = completed_step_snapshot(values, data, timings, device, wall())
        return values, snapshot
    baseline = {}
    if append and not data.get('_step_metrics_state'):
        _, baseline = take_snapshot()
    writer = StepMetricsWriter(log_path, args.timing_file, enabled=main_rank, append=append,
        baseline=baseline, state=data.get('_step_metrics_state') if append else None)
    sampler = DistributedSampler(train_rows, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(train_rows, collate_fn=TrainDataCollator(tokenizer, args.max_prompt_length),
                        num_workers=args.num_workers,
                        persistent_workers=args.persistent_workers and args.num_workers > 0,
                        batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=False)
    last_checkpoint_step, last_eval_step = -1, -1
    last_epoch, next_batch, stop = start_epoch, start_batch, False
    def checkpoint_at(epoch, batch):
        data['_step_metrics_state'] = writer.state_dict()
        return save_checkpoint(args.checkpoint_dir, target=target, optimizer=optimizer, data=data,
                               epoch=epoch, next_batch=batch, step=step, wall=wall(), keep_last=args.keep_last_checkpoints)
    disable_progress = not main_rank or os.environ.get('TQDM_DISABLE', '').lower() in {'1','true'}
    epoch_bar = tqdm(range(start_epoch, args.num_epochs), desc='GRPO epoch', disable=disable_progress)
    for epoch in epoch_bar:
        sampler.set_epoch(epoch)
        for key in ('ignore_due_correct','ignore_due_incorrect'):
            data[key] = 0
        for key in ('length_stdev','length_range','length_cv'):
            data[key] = []
        # Do not consume continuation RNG again when recreating a mid-epoch
        # iterator. Collation is deterministic; the DistributedSampler has its
        # own fixed epoch seed, identical to SpecNaacl's ordering.
        iterator = iter(loader)
        if resume_rng is not None and epoch == start_epoch and start_batch:
            from helper.checkpointing import restore_rng_state
            restore_rng_state(resume_rng)
            resume_rng = None
        batch_bar = tqdm(enumerate(iterator), total=len(loader), desc=f'GRPO epoch {epoch+1}',
                         disable=disable_progress, leave=False)
        for index, batch in batch_bar:
            if epoch == start_epoch and index < start_batch:
                continue
            last_epoch, next_batch = epoch, index + 1
            if ((max_optimizer_steps and data['optimizer_steps']>=max_optimizer_steps) or
                    (local_prompt_budget and data.get('rollout_prompts_seen',0)>=local_prompt_budget)):
                checkpoint_at(epoch,index)
                stop=True
                break
            if local_prompt_budget:
                remaining=local_prompt_budget-data.get('rollout_prompts_seen',0)
                if remaining<len(batch['answers']):batch={key:value[:remaining] for key,value in batch.items()}
            if batch['input_ids'].shape[-1] >= args.max_length or None in batch['answers']:
                continue
            import hashlib
            data['prompt_order_sha256']=hashlib.sha256((data.get('prompt_order_sha256','')+
                json.dumps(batch['messages'],sort_keys=True,ensure_ascii=True)).encode()).hexdigest()
            data['rollout_prompts_seen']=data.get('rollout_prompts_seen',0)+len(batch['answers'])
            output = generate(target, batch['input_ids'].to(device), batch['attention_mask'].to(device), tokenizer,
                do_sample=True, max_length=args.max_length, repeated_generate_nums=args.repeated_generate_nums,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k or None,
                statistical_time=args.statistical_time)
            output['decoded_sequences'] = [tokenizer.decode(x, skip_special_tokens=True) for x in output['generated_token_ids']]
            lengths = [len(x) for x in output['generated_token_ids']]
            data['total_rollout_tokens'] += sum(lengths)
            data['generated_samples'] += len(lengths)
            data['generate_time_cost'] += output['total_time_cost']
            data['generate_length_list'].extend(lengths)
            data['length_stdev'].append(stdev(lengths) if len(lengths)>1 else 0.)
            data['length_range'].append(max(lengths)-min(lengths))
            data['length_cv'].append((stdev(lengths) if len(lengths)>1 else 0.)/mean(lengths))
            functions = reward_functions or (None, None)
            used, _ = accumulate_groups(data, output, batch['messages'], batch['answers'],
                                       args.repeated_generate_nums, *functions, prompt_context=(rank, epoch, index))
            data['used_items'] += used
            data['trace_rollout_count'] += 1
            ready = torch.tensor(int(bool(data['messages'])), device=device, dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            if not bool(ready.item()):
                continue
            used_tensor = torch.tensor(data['used_items'], device=device, dtype=torch.long)
            if dist.is_initialized():
                dist.all_reduce(used_tensor, op=dist.ReduceOp.MIN)
            step = int(used_tensor.item()) // (args.batch_size * args.accumulation_steps)
            writer.advance(step)
            data['reward_sum'] += float(sum(data['rewards']))
            data['reward_count'] += len(data['rewards'])
            def on_iteration(iteration, losses):
                if iteration != args.grpo_iteration_num - 1 and not (max_optimizer_steps and data['optimizer_steps']>=max_optimizer_steps):
                    return
                values, snapshot = take_snapshot()
                extras = dict(phase='target_train', method='puregrpo', epoch=epoch+1,
                    grpo_step=max(0, step-trace_start_step), source_grpo_step=step,
                    used_items=values['used_items'], rollout_count=values['trace_rollout_count'],
                    optimizer_steps=values['optimizer_steps'], mean_reward=round(values['mean_reward'],4),
                    target_loss=values['target_loss'], train_dataset_full_size=full_size,
                    train_dataset_selected_size=len(train_rows), train_data_fraction=args.train_data_fraction,
                    train_subset_seed=args.train_subset_seed, max_train_samples=args.max_train_samples,
                    logps_chunk_size=args.logps_chunk_size,
                    used_time=round(snapshot['cumulative_wall_time_s']/60,3),
                    generate_time_cost=round(values['generate_time_cost']/60,3),
                    train_time_cost=round(values['train_time_cost']/60,3),
                    total_rollout_tokens=int(values['total_rollout_tokens']),
                    generated_samples=int(values['generated_samples']),
                    length_stdev=mean(data['length_stdev']), length_range=mean(data['length_range']),
                    length_cv=mean(data['length_cv']), grpo_iteration=iteration+1)
                writer.submit(step, snapshot, extras)
                data['_step_metrics_state'] = writer.state_dict()
                if step % args.log_interval == 0:
                    batch_bar.set_postfix(step=step, reward=extras['mean_reward'], loss=extras['target_loss'], phase='GRPO')
            update_policy(model, optimizer, data, tokenizer, args, timings, on_iteration)
            data.setdefault('optimizer_step_cadence',[]).append((data['rollout_prompts_seen'],data['optimizer_steps']))
            for key in ('messages','rewards','std_rewards','response_metadata'):
                data[key].clear()
            if main_rank and step and step % 500 == 0:
                target.save_pretrained(str(Path(args.saved_model_dir)/f'step{step}'))
            if args.eval_interval and step>0 and step%args.eval_interval==0 and step!=last_eval_step:
                from evaluation import evaluate
                from helper.checkpointing import capture_rng_state, restore_rng_state
                rng = capture_rng_state()
                modes = [(m,m.training) for m in target.modules()]
                try:
                    evaluate(args, target, tokenizer, rows=evaluation_rows,
                             reward_functions=reward_functions, step=step, generate=generate)
                finally:
                    for module, mode in modes:
                        module.training = mode
                    restore_rng_state(rng)
                last_eval_step = step
            local_steps = max(0,step-trace_start_step)
            if args.save_checkpoint_steps>0 and local_steps>0 and local_steps%args.save_checkpoint_steps==0 and step!=last_checkpoint_step:
                checkpoint_at(epoch, index+1)
                last_checkpoint_step = step
            if ((args.max_grpo_steps and local_steps>=args.max_grpo_steps) or
                    (max_optimizer_steps and data['optimizer_steps']>=max_optimizer_steps) or
                    (local_prompt_budget and data['rollout_prompts_seen']>=local_prompt_budget)):
                checkpoint_at(epoch,index+1)
                stop = True
                break
        if stop:
            break
    values, snapshot = take_snapshot()
    if writer.pending is not None:
        writer.submit(writer.pending['step'], snapshot, writer.pending['extras'])
    writer.flush()
    data['_step_metrics_state'] = writer.state_dict()
    saved_target = str(Path(args.saved_model_dir)/f'step{step}')
    if main_rank:
        target.save_pretrained(saved_target)
    if not stop:
        checkpoint_at(args.num_epochs,0)
    # Source's final job wall includes final target/checkpoint I/O. Step CSV is
    # drained before that I/O, exactly like its training_end snapshot.
    values, snapshot = take_snapshot()
    # Common summary names keep Source's wall-throughput aliases; no AAL fields.
    summary = dict(method='puregrpo', run_name=args.version_name, final_step=step,
        completed_grpo_steps=max(0,step-trace_start_step), optimizer_steps=int(values['optimizer_steps']),
        rollout_prompts_seen=data.get('rollout_prompts_seen',0),prompt_order_sha256=data.get('prompt_order_sha256'),
        optimizer_step_cadence=data.get('optimizer_step_cadence',[]),
        stopped_by_max_grpo_steps=stop, total_rollout_tokens=int(values['total_rollout_tokens']),
        generated_samples=int(values['generated_samples']), rollout_count=int(values['trace_rollout_count']),
        mean_reward=values['mean_reward'], target_loss=values['target_loss'],
        total_generate_time_s=values['generate_time_cost'], total_train_time_s=values['train_time_cost'],
        total_wall_time_s=snapshot['cumulative_wall_time_s'],
        tokens_per_s=values['tokens_per_s'], generation_tokens_per_s=values['tokens_per_s'],
        cumulative_generation_tokens_per_s=values['total_rollout_tokens']/max(values['generate_time_cost'],1e-9),
        saved_model_dir=saved_target, checkpoint_dir=args.checkpoint_dir,
        metrics_jsonl=args.log_file, timing_csv=args.timing_file, **snapshot)
    if main_rank:
        summary['initialization'] = getattr(target,'_initialization_report', {'status':'NOT VERIFIED'})
        summary['grpo_alignment_version'] = 'response_rows_v2'
        text = json.dumps(summary, indent=2)
        Path(args.summary_file).write_text(text+'\n',encoding='utf-8')
        Path(args.summary_file).with_suffix('.txt').write_text(text+'\n',encoding='utf-8')
        print(text)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return summary
