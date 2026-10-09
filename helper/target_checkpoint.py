"""Target/optimizer/per-rank RNG and progress only. No foreign resume format."""
from pathlib import Path
import torch
import torch.distributed as dist
from helper.checkpointing import capture_rng_state, restore_rng_state
from helper.grpo_core import _atomic_torch_save, _prune_checkpoints, _gradient_state, _restore_gradient_state


def save_checkpoint(path, *, target, optimizer, data, epoch, next_batch, step, wall,
                    keep_last=3, adapter_state=None):
    if adapter_state is None:
        from peft import get_peft_model_state_dict
        adapter_state = get_peft_model_state_dict(target)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local = dict(rank=rank, batch_data=data, rng=capture_rng_state(),
                 target_gradients=_gradient_state(target), cumulative_elapsed_time_s=wall)
    states = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank:
        return
    payload = dict(format='puregrpo_target_checkpoint_v1', grpo_alignment_version='response_rows_v2', world_size=world,
                   initial_target_tensor_sha256=getattr(target,'_shared_initialization',{}).get('loaded_tensor_sha256'),
                   target_lora=adapter_state, optimizer_target=optimizer.state_dict(),
                   scheduler_target=None, rank_states=states, epoch=epoch,
                   next_batch=next_batch, step=step,
                   cumulative_elapsed_time_s=max(x['cumulative_elapsed_time_s'] for x in states))
    root = Path(path)
    filename = root / f'step{step}_epoch{epoch + 1}_batch{next_batch}.pt'
    _atomic_torch_save(payload, filename)
    _atomic_torch_save(payload, root / 'latest.pt')
    _prune_checkpoints(root, keep_last)
    return filename


def load_checkpoint(path, *, target, optimizer, adapter_loader=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'puregrpo_target_checkpoint_v1':
        raise ValueError('resume requires a Pure GRPO checkpoint; use TARGET_ADAPTER for a target-only HF adapter')
    if state.get('grpo_alignment_version') != 'response_rows_v2':
        raise ValueError('GRPO alignment changed; old runs require retraining from initial weights')
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if state['world_size'] != world:
        raise ValueError('checkpoint world-size mismatch; use the saved number of ranks')
    if adapter_loader is None:
        from peft import set_peft_model_state_dict
        adapter_loader = set_peft_model_state_dict
    if state.get('initial_target_tensor_sha256') != getattr(target,'_shared_initialization',{}).get('loaded_tensor_sha256'):
        raise ValueError('resume initial target tensor hash mismatch')
    adapter_loader(target, state['target_lora'])
    optimizer.load_state_dict(state['optimizer_target'])
    local = state['rank_states'][rank]
    if local['rank'] != rank:
        raise ValueError('checkpoint rank mismatch')
    _restore_gradient_state(target, local['target_gradients'])
    restore_rng_state(local['rng'])
    state['local_state'] = local
    return state
