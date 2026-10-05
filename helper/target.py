"""Load only target + tokenizer + existing target LoRA configuration."""
from helper.grpo_core import _dtype_from_name, _resolve_attn_implementation


def load_target(args, device):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    dtype = _dtype_from_name(args.dtype)
    if dtype != 'auto':
        config.torch_dtype = dtype
    target = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype, config=config,
        attn_implementation=_resolve_attn_implementation(args.attn_implementation),
        local_files_only=True).to(device)
    target.eval()
    for param in target.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side='left', local_files_only=True)
    if config.model_type == 'llama':
        tokenizer.pad_token, tokenizer.pad_token_id = '<|end_of_text|>', 128001
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=32,
                             lora_dropout=0., target_modules=[
                                 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                                 'gate_proj', 'up_proj', 'down_proj'])
    target = get_peft_model(target, lora_config)
    if args.load_lora_path:
        # Same existing-default-adapter load as SpecNaacl (not PeftModel reconstruction).
        target.load_adapter(args.load_lora_path, adapter_name='default')
    target.print_trainable_parameters()
    return target, tokenizer
