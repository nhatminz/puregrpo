"""Tiny differentiable TARGET doubles; not production models or benchmark data."""
from types import SimpleNamespace
from pathlib import Path
import torch


class Cache:
    def __init__(self):
        self.ids = None
    def batch_repeat_interleave(self,n):
        self.ids = self.ids.repeat_interleave(n,0)
    def batch_select_indices(self,indices):
        self.ids = self.ids.index_select(0,indices)


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(124)
        self.embedding = torch.nn.Parameter(torch.randn(17,8,generator=g),requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.zeros(8))
        self.enabled = True
        self.calls = []
    def forward(self,input_ids,attention_mask,use_cache=False,past_key_values=None,**kwargs):
        self.calls.append(dict(ids=input_ids.detach().clone(),positions=kwargs.get('position_ids'),use_cache=use_cache))
        ids = input_ids
        if use_cache:
            previous = past_key_values.ids
            ids = torch.cat((previous,ids),1) if previous is not None else ids
            past_key_values.ids = ids
        features = self.embedding[ids]
        visible = attention_mask[...,None].to(features.dtype)
        features = features + (features*visible).cumsum(1)/(visible.cumsum(1).clamp_min(1))
        if self.enabled:
            features = features + self.adapter
        return SimpleNamespace(last_hidden_state=features[:,-input_ids.shape[1]:],past_key_values=past_key_values)


class Target(torch.nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(8,17,bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(17,8,generator=torch.Generator().manual_seed(45))*.2)
        self.lm_head.weight.requires_grad = False
        self.to(device)
        self.eval()
    @property
    def device(self):
        return self.model.adapter.device
    @property
    def dtype(self):
        return self.model.adapter.dtype
    def get_base_model(self):
        return self
    def forward(self,input_ids,attention_mask,**kwargs):
        out = self.model(input_ids,attention_mask,**kwargs)
        return SimpleNamespace(logits=self.lm_head(out.last_hidden_state))
    def disable_adapter_layers(self):
        self.model.enabled = False
    def enable_adapter_layers(self):
        self.model.enabled = True
    def save_pretrained(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),Path(path)/'tiny_target_test_only.pt')


class Tokenizer:
    eos_token_id = 16
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        def one(row):
            if row[-1]['role'] == 'assistant':
                return 'prompt ' + row[-1]['content']
            return 'prompt'
        return [one(row) for row in messages] if isinstance(messages[0],list) else one(messages)
    def encode(self,text):
        return [3,5] if text=='prompt' else [3,5]+[int(x) for x in text.split()[1:]]
    def __call__(self,text=None,**kwargs):
        values = [self.encode(x) for x in text]
        masks = [[1]*len(x) for x in values]
        if kwargs.get('return_tensors')=='pt':
            width = max(map(len,values))
            return {'input_ids':torch.tensor([[0]*(width-len(x))+x for x in values]),
                    'attention_mask':torch.tensor([[0]*(width-len(x))+m for x,m in zip(values,masks)])}
        return SimpleNamespace(input_ids=values,attention_mask=masks)
    def decode(self,tokens,skip_special_tokens=True):
        return ' '.join(str(x) for x in tokens if not(skip_special_tokens and x==self.eos_token_id))
