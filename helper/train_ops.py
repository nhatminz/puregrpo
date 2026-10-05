"""Inherited group filtering, text masks/packing and target optimizer schedule."""
from copy import deepcopy
import time
import numpy as np
import torch
from helper.grpo_core import compute_target_loss_and_backward, _sync_gradients


def accumulate_groups(batch_data, outputs, messages, answers, repeated_generate_nums, format_reward_func=None, accuracy_reward_func=None):
    if format_reward_func is None or accuracy_reward_func is None:
        from helper.rewards import format_reward_func, accuracy_reward_func
    used_items = 0
    generate_length=0
    for idx_batch in range(len(answers)):
        generate_length += outputs['max_sequence_length']
        rewards=[]
        new_messages=[]
        for idx_k in range(repeated_generate_nums):
            idx_sequence=idx_batch*repeated_generate_nums+idx_k
            decoded_sequence=outputs['decoded_sequences'][idx_sequence]
            ground_truth=answers[idx_batch]
            
            new_message=deepcopy(messages[idx_batch])
            new_message.append({
                "role": "assistant",
                "content":decoded_sequence
            })
            
            format_reward=format_reward_func([decoded_sequence])
            answer_reward=accuracy_reward_func([decoded_sequence],[ground_truth])
            reward=0.2*format_reward[0]+answer_reward[0]
            
            rewards.append(reward)
            new_messages.append(new_message)
        
        
        rewards=np.array(rewards) 
        if rewards.std()==0:
            
            if rewards[0]>=1.0:
                batch_data['ignore_due_correct']+=1
            else:
                batch_data['ignore_due_incorrect']+=1
                
            continue
        
        std_rewards=(rewards-rewards.mean())/rewards.std()
        batch_data['messages']+=new_messages
        batch_data['rewards']+=rewards.tolist()
        batch_data['std_rewards']+=std_rewards.tolist()
        used_items+=1
        
    generate_length /= len(answers)
    
    return used_items, generate_length


def pack_training_inputs(batch_data, tokenizer):
    text=tokenizer.apply_chat_template(batch_data['messages'],tokenize=False,add_generation_prompt=False)
    text=tokenizer(text,padding=False)
    loss_mask=[]
    
    for idx_message, message in enumerate(batch_data['messages']):
        prompt_text=tokenizer.apply_chat_template(message[:-1],tokenize=False,add_generation_prompt=True)
        prompt_text=tokenizer.encode(prompt_text)
        cur_loss_mask=[0]*(len(prompt_text)-1)+[1]*(len(text.input_ids[idx_message])-len(prompt_text)+1)
        loss_mask.append(cur_loss_mask)
        
    input_ids=text.input_ids
    attention_mask=text.attention_mask
    
    sorted_pairs = sorted(
        zip(input_ids, attention_mask, loss_mask),
        key=lambda x: len(x[0]),
        reverse=False   
    )

    input_ids_sorted, attention_mask_sorted, loss_mask_sorted = zip(*sorted_pairs)

    input_ids, attention_mask, loss_mask = list(input_ids_sorted), list(attention_mask_sorted), list(loss_mask_sorted)

    return input_ids, attention_mask, loss_mask


def update_policy(model, optimizer_target, batch_data, tokenizer, args, phase_timings, on_iteration=None):
    input_ids, attention_mask, loss_mask = pack_training_inputs(batch_data, tokenizer)
    grpo_iteration_num = args.grpo_iteration_num
    statistical_time = args.statistical_time
    max_training_token = args.max_training_token
    max_training_padding_gap = args.max_training_padding_gap
    logps_chunk_size = args.logps_chunk_size
    epsilon, beta = args.epsilon, args.beta
    batch_old_logps, batch_ref_logps = [], []
    for grpo_iteration in range(grpo_iteration_num):
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        train_time_start=time.time()
        target_phase_ticket = phase_timings.begin('target')
        
        cur_max_length=0
        device=model.target_model.device
        microbatch_index=0
        
        cur_input_ids=[]
        cur_attention_mask=[]
        cur_loss_mask=[]
        cur_rewards=[]
        
        for j in range(len(batch_data['messages'])):
            
            if ((max(cur_max_length, len(input_ids[j])) * (len(cur_input_ids)+1)<=max_training_token and
                (len(input_ids[j])-cur_max_length)*len(cur_input_ids)<=max_training_padding_gap) or
                len(cur_input_ids)==0):
                cur_max_length=max(cur_max_length, len(input_ids[j]))
                
                cur_input_ids.append(input_ids[j])
                cur_attention_mask.append(attention_mask[j])
                cur_loss_mask.append(loss_mask[j])
                cur_rewards.append(batch_data['std_rewards'][j])
                
            else:
                
                cur_batch=len(cur_input_ids)
                for idx_seq in range(cur_batch):
                    
                    cur_len=len(cur_input_ids[idx_seq])
                    padding_len=cur_max_length-cur_len
                    
                    if padding_len>0:
                        
                        cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                        cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                        cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                        
                cur_input_ids=torch.tensor(cur_input_ids, device=device)
                cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
                cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
                cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                    model,
                    cur_input_ids,
                    cur_attention_mask,
                    cur_loss_mask,
                    cur_rewards,
                    epsilon,
                    beta,
                    grpo_iteration,
                    old_logps=old_logps,
                    ref_logps=ref_logps,
                    chunk_size=logps_chunk_size,
                    loss_scale=1.0 / max(len(batch_data['messages']), 1),
                )
                batch_data['target_loss_sum'] += float(loss)
                batch_data['target_loss_count'] += int(cur_batch)
                    
                if grpo_iteration==0:
                    batch_old_logps.append(old_logps)
                    batch_ref_logps.append(ref_logps)
                microbatch_index += 1
                del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards
                
                cur_input_ids=[input_ids[j]]
                cur_attention_mask=[attention_mask[j]]
                cur_loss_mask=[loss_mask[j]]
                cur_rewards=[batch_data['std_rewards'][j]]
                
                cur_max_length=len(input_ids[j])
                
        cur_batch=len(cur_input_ids)
        for idx_seq in range(cur_batch):
            
            cur_len=len(cur_input_ids[idx_seq])
            padding_len=cur_max_length-cur_len
            
            if padding_len>0:
                
                cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                
        cur_input_ids=torch.tensor(cur_input_ids, device=device)
        cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
        cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
        cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

        old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
        ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
        loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
            model,
            cur_input_ids,
            cur_attention_mask,
            cur_loss_mask,
            cur_rewards,
            epsilon,
            beta,
            grpo_iteration,
            old_logps=old_logps,
            ref_logps=ref_logps,
            chunk_size=logps_chunk_size,
            loss_scale=1.0 / max(len(batch_data['messages']), 1),
        )
        batch_data['target_loss_sum'] += float(loss)
        batch_data['target_loss_count'] += int(cur_batch)
            
        if grpo_iteration==0:
            batch_old_logps.append(old_logps)
            batch_ref_logps.append(ref_logps)
        microbatch_index += 1
        del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards

            
        _sync_gradients(model.target_model)
        optimizer_target.step()
        optimizer_target.zero_grad(set_to_none=True)
        phase_timings.end(target_phase_ticket)
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_data['train_time_cost'] += time.time() - train_time_start
        batch_data['optimizer_steps'] += 1
        if on_iteration is not None:
            on_iteration(grpo_iteration, (loss, abs_loss1, loss2))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # same per-iteration behavior as SpecNaacl
    return batch_old_logps, batch_ref_logps
