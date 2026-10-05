import ast
from pathlib import Path
import torch
import pytest
from helper.autoregressive import autoregressive_generate
from helper.sampling import build_sampling_probs,sample_from_probs
from fixtures import Target,Cache,Tokenizer

DEVICES=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('sample',[False,True])
def test_target_cached_token_by_token_matches_full_prefix_reference(device,sample):
    target,tokenizer=Target(device),Tokenizer()
    inputs=torch.tensor([[0,3,5],[3,5,7]],device=device)
    mask=torch.tensor([[0,1,1],[1,1,1]],device=device)
    torch.manual_seed(77)
    output=autoregressive_generate(target,inputs,mask,tokenizer,max_length=10,repeated_generate_nums=8,
                                  do_sample=sample,temperature=.8,top_p=.9,top_k=7,cache_factory=Cache)
    calls=target.model.calls.copy()
    assert calls[0]['ids'].shape==(2,3)
    assert all(row['ids'].shape[1]==1 for row in calls[1:])
    assert len(calls)==1+output['target_decode_forwards']
    assert len(output['generated_token_ids'])==16
    torch.manual_seed(77)
    sequences=inputs.repeat_interleave(8,0)
    masks=mask.repeat_interleave(8,0)
    active=list(range(16)); answers=[[] for _ in active]
    prompt_lengths=mask.sum(-1).repeat_interleave(8,0).tolist()
    with torch.inference_mode():
        for step in range(10):
            # Same autocast/dtype as production sampler, but no cached forward.
            from contextlib import nullcontext
            with torch.amp.autocast('cuda',dtype=torch.float16) if device!='cpu' else nullcontext():
                out=target.model(sequences,masks,use_cache=False)
                logits=target.lm_head(out.last_hidden_state[:,-1:])
            token=(sample_from_probs(build_sampling_probs(logits,.8,.9,7,16))[:,0]
                   if sample else logits[:,0].argmax(-1))
            keep=[]
            for row,t in enumerate(token.tolist()):
                answers[active[row]].append(t)
                if t!=16: keep.append(row)
            if not keep or max(prompt_lengths[i]+step+1 for i in active)>=10: break
            sequences=torch.cat((sequences,token[:,None]),1)[keep]
            masks=torch.cat((masks,torch.ones_like(token[:,None])),1)[keep]
            active=[active[row] for row in keep]
    assert answers==output['generated_token_ids']
    for call in calls[1:]:
        assert bool((call['positions']>=0).all())


@pytest.mark.parametrize('device',DEVICES)
def test_zero_eos_rows_compact_and_first_response_tokens_are_independent(device):
    target=Target(device)
    target.lm_head.weight.data.zero_()
    # Equal logits: root samples should differ across 8 responses, not duplicate
    # a single sampled root across responses as in the speculative prefix.
    torch.manual_seed(3)
    output=autoregressive_generate(target,torch.tensor([[3,5]],device=device),
        torch.ones(1,2,device=device,dtype=torch.long),Tokenizer(),
        max_length=6,repeated_generate_nums=8,top_p=1.,cache_factory=Cache)
    assert len({row[0] for row in output['generated_token_ids']})>1
    assert all(1<=len(row)<=4 for row in output['generated_token_ids'])
    for row in output['generated_token_ids']:
        assert 16 not in row[:-1]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('stop_on_eos',[False,True])
def test_eos_compaction_and_shorter_remaining_prompt_length_limit(device,stop_on_eos):
    target=Target(device)
    class ControlledHead(torch.nn.Module):
        calls=0
        def forward(self,hidden):
            logits=torch.full((*hidden.shape[:-1],17),-float('inf'),device=hidden.device)
            self.calls+=1
            logits[...,1]=0.
            if self.calls==1:
                logits[0,:,1]=-float('inf'); logits[0,:,16]=0.
            elif self.calls==3 and stop_on_eos:
                logits[...,1]=-float('inf'); logits[...,16]=0.
            return logits
    target.lm_head=ControlledHead()
    output=autoregressive_generate(target,torch.tensor([[3,5,7],[0,0,3]],device=device),
        torch.tensor([[1,1,1],[0,0,1]],device=device),Tokenizer(),max_length=6,
        repeated_generate_nums=2,do_sample=False,cache_factory=Cache)
    assert output['generated_token_ids'][:2]==[[16],[16]]
    assert output['generated_token_ids'][2:]==([[1,1,16]]*2 if stop_on_eos else [[1]*5]*2)
    assert all(call['ids'].shape==(2,1) for call in target.model.calls[1:])
    assert output['target_decode_forwards']==(2 if stop_on_eos else 4)


def test_sampler_primitives_are_verbatim_and_target_probability_identity():
    original=Path(__file__).resolve().parents[2]/'SpecNaacl/helper/sampling.py'
    namespace={'torch':torch}
    tree=ast.parse(original.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'build_sampling_probs','sample_from_probs'}]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(original),'exec'),namespace)
    for logits in (torch.randn(3,2,17),torch.full((3,2,17),float('nan'))):
        a=build_sampling_probs(logits,.8,.95,5,16)
        b=namespace['build_sampling_probs'](logits,.8,.95,5,16)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.manual_seed(4); x=sample_from_probs(a)
        torch.manual_seed(4); y=namespace['sample_from_probs'](b)
        assert torch.equal(x,y)


def test_real_hf_qwen2_cached_target_without_checkpoint_or_draft():
    transformers=pytest.importorskip('transformers')
    torch.manual_seed(7)
    config=transformers.Qwen2Config(
        vocab_size=17,hidden_size=16,intermediate_size=24,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,eos_token_id=16)
    config._attn_implementation='eager'
    model=transformers.Qwen2ForCausalLM(config)
    model.eval()
    # Suppress EOS only in this test to cover several actual HF cached steps.
    model.lm_head.register_forward_hook(lambda m,i,o:o.index_fill(-1,torch.tensor([16]),-100.))
    torch.manual_seed(7)
    out=autoregressive_generate(model,torch.tensor([[0,3,5],[3,5,7]]),torch.tensor([[0,1,1],[1,1,1]]),
                                Tokenizer(),max_length=8,repeated_generate_nums=2,do_sample=False)
    assert len(out['generated_token_ids'])==4
    assert out['target_decode_forwards']>0
    assert not hasattr(model,'draft_model')
    tokens=torch.tensor([[0,3,5],[3,5,7]]).repeat_interleave(2,0)
    masks=torch.tensor([[0,1,1],[1,1,1]]).repeat_interleave(2,0)
    expected=[[] for _ in range(4)]
    with torch.inference_mode():
        for step in range(5):
            hidden=model.model(input_ids=tokens,attention_mask=masks,
                               position_ids=(masks.cumsum(-1)-1).clamp_min(0),
                               use_cache=False,return_dict=True).last_hidden_state
            nxt=model.lm_head(hidden[:,-1:]).argmax(-1)
            for row,value in enumerate(nxt[:,0].tolist()): expected[row].append(value)
            tokens=torch.cat((tokens,nxt),1)
            masks=torch.cat((masks,torch.ones_like(nxt)),1)
    assert out['generated_token_ids']==expected
