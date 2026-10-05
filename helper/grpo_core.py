"""Target-only primitives extracted verbatim from SpecNaacl; no entrypoint import."""
import os
import random
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
world_size = int(os.environ.get("WORLD_SIZE", "1"))

def _dtype_from_name(name):
    name = str(name or "auto").lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")

def _resolve_attn_implementation(requested):
    requested = str(requested or "")
    if not requested:
        return None
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed. Falling back to eager."
        )
        return "eager"
    return requested

def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _get_base_causal_lm(causal_lm):
    """Return the underlying causal LM while preserving injected LoRA modules."""
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm

def _autocast_dtype(causal_lm):
    dtype = getattr(causal_lm, "dtype", None)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16

def _token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    """Compute selected-token log-probabilities without a full [B, T, vocab] tensor."""
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    logps_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected_logits = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_denominator = torch.logsumexp(logits, dim=-1)
        logps_chunks.append(selected_logits - log_denominator)
        del logits, selected_logits, log_denominator

    if logps_chunks:
        return torch.cat(logps_chunks, dim=1)
    return hidden_states.new_zeros((hidden_states.shape[0], 0))

def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    """Forward the backbone once, then apply the LM head in bounded chunks."""
    base_model = _get_base_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type

    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    return _token_logps_from_hidden(
        hidden_states, base_model.lm_head, input_ids, chunk_size
    )

def compute_target_loss_and_backward(
    model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    grpo_iteration,
    old_logps=None,
    ref_logps=None,
    chunk_size=256,
    loss_scale=1.0,
):
    """Compute the GRPO loss and backpropagate without full-vocabulary logits."""
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    if grpo_iteration == 0:
        model.target_model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps_gpu = compute_model_token_logps(
                model.target_model,
                input_ids,
                attention_mask,
                chunk_size,
            ).detach()
        model.target_model.enable_adapter_layers()
        ref_logps_for_loss = ref_logps_gpu
        old_logps_for_loss = None
    else:
        if old_logps is None or ref_logps is None:
            raise ValueError(
                "old_logps and ref_logps are required when grpo_iteration > 0"
            )
        old_logps_for_loss = old_logps.to(device, non_blocking=True)
        ref_logps_for_loss = ref_logps.to(device, non_blocking=True)

    model.target_model.enable_adapter_layers()
    base_model = _get_base_causal_lm(model.target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    old_chunks_to_store = []
    loss_value = 0.0
    abs_loss1_value = 0.0
    loss2_value = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        cur_mask = token_mask[:, start:end]

        if grpo_iteration == 0:
            cur_old_logps = logps.detach()
            old_chunks_to_store.append(cur_old_logps.cpu())
        else:
            cur_old_logps = old_logps_for_loss[:, start:end]

        cur_ref_logps = ref_logps_for_loss[:, start:end]
        coef1 = torch.exp(logps - cur_old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)

        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        scaled_loss = chunk_loss * loss_scale
        scaled_loss.backward(retain_graph=(end < seq_len))

        with torch.no_grad():
            loss_value += float(chunk_loss.detach().cpu())
            abs_loss1_value += float(
                torch.abs((loss1 * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )
            loss2_value += float(
                ((kl * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )

        del (
            logits,
            logps,
            coef1,
            coef2,
            loss1,
            coef3,
            kl,
            token_loss,
            chunk_loss,
            scaled_loss,
        )

    if grpo_iteration == 0:
        if old_chunks_to_store:
            old_logps_out = torch.cat(old_chunks_to_store, dim=1)
        else:
            old_logps_out = torch.empty((input_ids.shape[0], 0))
        ref_logps_out = ref_logps_for_loss.detach().cpu()
    else:
        old_logps_out = old_logps
        ref_logps_out = ref_logps

    del hidden_states, policy_hidden
    if grpo_iteration == 0:
        del ref_logps_gpu, ref_logps_for_loss
    else:
        del old_logps_for_loss, ref_logps_for_loss

    return (
        loss_value,
        abs_loss1_value,
        loss2_value,
        old_logps_out,
        ref_logps_out,
    )

class TrainDataCollator:
    def __init__(self, tokenizer, max_prompt_length):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
    
    def __call__(self, batch):
        system_prompt = "You are a math problem assistant." 
        user_prompt =  '''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
        messages = []
        answers = []

        for example in batch:
            messages.append([
                {"role" : "system" , "content": system_prompt} , 
                {"role" : "user" , "content": user_prompt.format_map({"instruction" : example['question']}) }
            ])
            answers.append(example['answer'])
        tokenized_inputs = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            return_tensors='pt',padding='longest',truncation=True,max_length=self.max_prompt_length,padding_side='left'
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'messages': messages,        
            'answers': answers,           
        }

def _sync_gradients(module):
    if not dist.is_initialized():
        return
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

def _atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def _prune_checkpoints(checkpoint_dir, keep_last):
    keep_last = int(keep_last or 0)
    if keep_last <= 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("step*.pt"), key=lambda p: p.stat().st_mtime)
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)

def _gradient_state(module):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }

def _restore_gradient_state(module, state):
    parameters = dict(module.named_parameters())
    for name, value in (state or {}).items():
        if name not in parameters:
            raise ValueError(f"checkpoint gradient parameter is missing: {name}")
        parameters[name].grad = value.to(parameters[name].device, parameters[name].dtype)
