"""Target-only token-at-a-time decoding; one ordinary target KV cache.

No draft/model adapter, verifier, tree, proposal, acceptance or Reflex imports.
Uses the inherited FP32 target distribution, not HF generate's alternate sampler.
"""
import time
from contextlib import nullcontext
import torch
from helper.grpo_core import _get_base_causal_lm, _autocast_dtype
from helper.sampling import build_sampling_probs, sample_from_probs


@torch.inference_mode()
def autoregressive_generate(target_model, input_ids, attention_mask, tokenizer, *,
                            do_sample=True, max_length=2048, repeated_generate_nums=8,
                            temperature=1., top_p=.95, top_k=None, statistical_time=False,
                            cache_factory=None):
    started = time.time()  # includes completion metadata/setup, like Source
    base = _get_base_causal_lm(target_model)
    device = input_ids.device
    repeats = max(1, int(repeated_generate_nums))
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError('expected aligned [batch,prompt] inputs and attention mask')
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError('same scalar tokenizer EOS as SpecNaacl is required')
    prompt_width = input_ids.shape[1]
    prompt_lengths = [int(x) for x in attention_mask.sum(-1).tolist()]
    budget = int(max_length) - min(prompt_lengths)
    if min(prompt_lengths) <= 0 or max(prompt_lengths) >= int(max_length):
        raise ValueError('prompt consumes max_length; no generation budget remains')
    eos = tokenizer.eos_token_id
    if cache_factory is None:
        from transformers import DynamicCache
        cache_factory = lambda: DynamicCache(config=base.config)
    cache = cache_factory()
    # Preserve SpecNaacl's zero positions for left-padding and real RoPE indices.
    positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    def forward(ids, mask, pos, cache_pos):
        ctx = (torch.amp.autocast('cuda', dtype=_autocast_dtype(base))
               if device.type == 'cuda' else nullcontext())
        with ctx:
            out = base.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                             cache_position=cache_pos, past_key_values=cache, use_cache=True,
                             output_attentions=False, output_hidden_states=False, return_dict=True)
            logits = base.lm_head(out.last_hidden_state[:, -1:, :])
        return out.past_key_values, logits
    prefill_started = time.time()
    cache, logits = forward(input_ids, attention_mask, positions,
                            torch.arange(prompt_width, device=device))
    if statistical_time and device.type == 'cuda':
        torch.cuda.synchronize()  # inherited explicit diagnostic mode only
    prefill_time = time.time() - prefill_started
    # Shared deterministic prefix KV; EVERY response samples its own first token.
    cache.batch_repeat_interleave(repeats)
    logits = logits.repeat_interleave(repeats, 0)
    n = input_ids.shape[0] * repeats
    active = torch.arange(n, device=device)
    original = list(range(n))
    lengths = [0] * n
    history = torch.empty((n, budget), device=device, dtype=torch.long)
    mask_buffer = torch.ones((n, prompt_width + budget), device=device, dtype=attention_mask.dtype)
    mask_buffer[:, :prompt_width] = attention_mask.repeat_interleave(repeats, 0)
    real_prompt_lengths = attention_mask.sum(-1).long().repeat_interleave(repeats, 0)
    decode_forwards = 0
    for index in range(budget):
        if do_sample:
            probabilities = build_sampling_probs(logits, temperature, top_p, top_k, eos)
            token = sample_from_probs(probabilities)[:, 0]
            del probabilities
        else:
            token = logits[:, 0].argmax(-1)
        history[active, index] = token
        # One necessary completion packet, not copies of logits/features/cache.
        finished = token.eq(eos).tolist()
        keep_rows = []
        for row, ended in enumerate(finished):
            lengths[original[row]] = index + 1
            if not ended:
                keep_rows.append(row)
        # Same active-batch real-length stopping rule as SpecNaacl, but no
        # verification-chunk overshoot or artificial tree padding.
        reached_limit = max(prompt_lengths[row // repeats] + index + 1 for row in original) >= int(max_length)
        if not keep_rows or reached_limit:
            break
        if len(keep_rows) != len(original):
            keep = torch.tensor(keep_rows, device=device, dtype=torch.long)
            cache.batch_select_indices(keep)
            token = token.index_select(0, keep)
            active = active.index_select(0, keep)
            mask_buffer = mask_buffer.index_select(0, keep)
            real_prompt_lengths = real_prompt_lengths.index_select(0, keep)
            original = [original[row] for row in keep_rows]
        cache, logits = forward(token[:, None], mask_buffer[:, :prompt_width + index + 1],
                                (real_prompt_lengths + index)[:, None],
                                torch.tensor([prompt_width + index], device=device))
        decode_forwards += 1
    # Transfer once; outputs own ordinary Python token lists. No batch row views.
    rows = history[:, :max(lengths)].cpu().tolist()
    generated = [row[:length] for row, length in zip(rows, lengths)]
    elapsed = time.time() - started  # completion transfer has finished GPU work
    return dict(generated_token_ids=generated,
                response_generated_tokens=lengths,
                total_time_cost=elapsed, prefill_time_cost=prefill_time,
                max_sequence_length=prompt_width + max(lengths),
                target_prefill_forwards=1, target_decode_forwards=decode_forwards,
                generation_tokens_per_s=sum(lengths) / max(elapsed, 1e-9))
