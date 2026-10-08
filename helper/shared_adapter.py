"""Initialization-only adapter audit. Never called from generation/training loops."""
import hashlib
import json
from pathlib import Path


TARGET_MODULES = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def validate_adapter_recipe(path,model_dir=None):
    path = Path(path)
    config = json.loads((path/'adapter_config.json').read_text())
    expected = {'r':64, 'lora_alpha':32, 'lora_dropout':0.0,
                'bias':'none', 'task_type':'CAUSAL_LM', 'peft_type':'LORA'}
    errors = {k:(config.get(k),v) for k,v in expected.items() if config.get(k) != v}
    if set(config.get('target_modules',())) != TARGET_MODULES:
        errors['target_modules'] = 'must match the shared target recipe'
    # Per-module overrides or saved extra trainable modules are not this recipe.
    for key in ('rank_pattern','alpha_pattern','modules_to_save','use_dora','use_rslora'):
        if config.get(key):
            errors[key] = 'not supported in the shared initialization recipe'
    if errors:
        raise ValueError('TARGET_ADAPTER does not match SpecNaacl target LoRA: '+json.dumps(errors))
    manifest = path/'shared_initialization.json'
    if model_dir is not None and manifest.is_file():
        recorded = json.loads(manifest.read_text()).get('model_dir')
        if recorded is not None and Path(recorded).resolve() != Path(model_dir).resolve():
            raise ValueError('Shared initialization was created for a different target model')
    return config


def adapter_artifact(path):
    validate_adapter_recipe(path)
    path = Path(path)
    candidates = [path/'adapter_model.safetensors',path/'adapter_model.bin']
    artifact = next((p for p in candidates if p.is_file()),None)
    if artifact is None:
        raise FileNotFoundError('Missing adapter weights in '+str(path))
    digest = hashlib.sha256()
    with artifact.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    manifest = path/'shared_initialization.json'
    if manifest.is_file():
        expected = json.loads(manifest.read_text())['adapter_weights_sha256']
        if digest.hexdigest() != expected:
            raise ValueError('Shared adapter weights changed after initialization')
    return artifact,digest.hexdigest()


def verify_loaded_adapter(target,path,model_dir=None):
    """Check every saved parameter; copies happen only once during initialization."""
    import torch
    from peft import get_peft_model_state_dict
    validate_adapter_recipe(path,model_dir)
    artifact,digest = adapter_artifact(path)
    if artifact.suffix == '.safetensors':
        from safetensors.torch import load_file
        saved = load_file(str(artifact),device='cpu')
    else:
        saved = torch.load(artifact,map_location='cpu',weights_only=True)
    actual = get_peft_model_state_dict(target,adapter_name='default')
    if saved.keys() != actual.keys():
        raise ValueError('Loaded target adapter keys do not match shared initialization')
    for key,value in actual.items():
        expected = saved[key].to(dtype=value.dtype)
        if not torch.equal(value.detach().cpu(),expected):
            raise ValueError('Loaded target adapter differs at '+key)
    result = {'adapter_weights_sha256':digest,'adapter_parameters_verified':len(saved)}
    print('Shared target initialization verified: '+json.dumps(result),flush=True)
    return result
