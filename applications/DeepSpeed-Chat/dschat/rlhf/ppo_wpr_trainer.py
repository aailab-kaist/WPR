# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import csv

from dschat.utils.utils import print_rank_0, unwrap_model_for_generation


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels, get_all=False):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedWPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = self.args.kl_ctl
        self.clip_reward_value = 2.5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = self.args.gamma
        self.lam = self.args.lam
        self.generate_time = 0.0
        self.temperature = self.args.temperature
        self.termination_condition = self.args.termination_condition
        self.value_function = self.args.value_function
        self.ngram = getattr(self.args, 'ngram', None)  # for fixed n-gram selection
        self.repeat_times = getattr(self.args, 'repeat_times', None)  # randomized n-gram selection
        self.parser_cutoff = getattr(self.args, 'parser_cutoff', None)  # parsing selection

        self.f_eval = self._get_f_divergence(self.args.f_div, self.args.f_div_alpha)

    def _get_f_divergence(self, f_divergence, alpha=0.5):
        if f_divergence == 'rkl':
            def f(x, logx=None):
                if logx is None:
                    logx = torch.log(x)
                return -logx
            f_prime_one = -1.
        elif f_divergence == 'fkl':
            def f(x, logx=None):
                if logx is None:
                    logx = torch.log(x)
                return x * logx
            f_prime_one = 1.
        elif f_divergence == 'js':
            def f(x, logx=None):
                if logx is None:
                    logx = torch.log(x)
                log_x_plus_1 = torch.log((x + 1) / 2)
                return x * logx - (x + 1) * log_x_plus_1
            f_prime_one = 0.
        elif f_divergence == 'alpha':
            assert alpha > 0, 'alpha must be positive'
            assert alpha < 1, 'alpha must be less than 1'
            def f(x, logx=None):
                return (1.0 / (alpha * (alpha - 1))) * (torch.pow(x, alpha) - alpha * x + alpha - 1)
            f_prime_one = 0.
        else:
            raise NotImplementedError

        return f #, f_prime_one

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        generation_config = dict(do_sample=True, temperature=self.temperature)

        with torch.no_grad():
            if self.args.enable_zero3_generation_gather and self.z3_enabled:
                with unwrap_model_for_generation(self.actor_model) as unwrapped_model:
                    seq = unwrapped_model.generate(
                        prompts,
                        attention_mask=mask,
                        max_length=max_min_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        synced_gpus=self.z3_enabled,
                        **generation_config)
            else:
                seq = self.actor_model.module.generate(
                    prompts,
                    attention_mask=mask,
                    max_length=max_min_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    synced_gpus=self.z3_enabled,
                    **generation_config)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.decode(prompts[i], skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.decode(ans[i], skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask, return_dict=True)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
            )
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        ppl = []
        if self.termination_condition == 'ppl':
            shift_logits = logits_ref[..., :-1, :].contiguous()
            shift_labels = seq[..., 1:].contiguous()
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels_flat = shift_labels.view(-1)

            loss_per_token = F.cross_entropy(shift_logits_flat, shift_labels_flat, reduction='none',
                                             ignore_index=pad_token_id)
            loss_per_token = loss_per_token.view(shift_labels.size())
            ppl_per_position = torch.exp(loss_per_token)
            ppl = ppl_per_position[0].tolist()[prompts.size(-1) - 1:]

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,  1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'ppl': [ppl],
            'logits': logits[:, :-1, :],
            'logits_ref': logits_ref[:, :-1, :],
        }

    def sparse_sinkhorn_sequence_topk(self,
                                      pi_r,  # (B,T,N)
                                      pi_theta,  # (B,T,N)
                                      K,  # (N,N)  sparse exp(-C/ε)
                                      start: int,
                                      ends: torch.Tensor,
                                      seq,  # (B,T)  label indices
                                      top_k: int = 128,
                                      epsilon: float = 0.01,
                                      max_iter: int = 10,
                                      tol: float = 1e-6,
                                      normalize = False):
        B, T, N = pi_r.shape
        M = B * T
        device = pi_r.device

        pi_r_flat = pi_r.view(M, N)
        pi_theta_flat = pi_theta.view(M, N)

        active_list = [torch.arange(start, min(T, ends[b].item()), device=device) + b * T for b in range(B)]
        active_list = torch.cat(active_list, dim=0)
        active = torch.zeros(M, dtype=torch.bool, device=device)
        active[active_list] = True

        idx_r = torch.topk(pi_r_flat[active], top_k, dim=-1).indices  # (M,K)
        idx_t = torch.topk(pi_theta_flat[active], top_k, dim=-1).indices  # (M,K)
        idx_lab = seq[:, start:]

        support = torch.unique(torch.cat([idx_r.flatten(), idx_t.flatten(), idx_lab.flatten()], dim=-1))
        support, _ = torch.sort(support)  # (Ns,)
        support = torch.unique(torch.cat([support[support != 0], torch.tensor([1], device=support.device)]))
        Ns = support.numel()

        mapping = torch.full((N,), -1, dtype=torch.long, device=device)
        mapping[support] = torch.arange(Ns, device=device)  # support[i] -> i
        dust_idx = Ns

        pi_r_sub = pi_r_flat[:, support]  # (M,Ns)
        pi_t_sub = pi_theta_flat[:, support]

        tail_r = (1.0 - pi_r_sub.sum(-1, keepdim=True)).clamp_min(0.)  # (M,1)
        tail_t = (1.0 - pi_t_sub.sum(-1, keepdim=True)).clamp_min(0.)

        pi_r_sub = torch.cat([pi_r_sub, tail_r], dim=-1)  # (M,Ns+1)
        pi_t_sub = torch.cat([pi_t_sub, tail_t], dim=-1)

        Nsub = Ns + 1

        idx = K.indices()  # (2, nnz)
        val = K.values()

        keep = (mapping[idx[0]] >= 0) & (mapping[idx[1]] >= 0)
        row_new = mapping[idx[0, keep]]
        col_new = mapping[idx[1, keep]]
        val_new = val[keep]

        idx_dust = torch.tensor([[dust_idx], [dust_idx]], device=device)
        val_dust = torch.ones(1, device=device)

        K_sub = torch.sparse_coo_tensor(
            torch.cat([torch.stack([row_new, col_new]), idx_dust], dim=1),
            torch.cat([val_new, val_dust]),
            size=(Nsub, Nsub)
        ).coalesce()
        Kt_sub = K_sub.transpose(0, 1).coalesce()

        u = torch.ones(M, Nsub, device=device)
        v = torch.ones_like(u)

        active_list = [torch.arange(start, min(T, ends[b].item()), device=device) + b * T
                       for b in range(B)]
        active = torch.cat(active_list, dim=0)

        for _ in range(max_iter):
            if active.numel() == 0:
                break

            u_act = u[active]
            v_act = v[active]

            Kv = torch.sparse.mm(K_sub, v_act.t()).t()
            u_new = pi_t_sub[active] / (Kv + 1e-16)

            Ktu = torch.sparse.mm(Kt_sub, u_new.t()).t()
            v_new = pi_r_sub[active] / (Ktu + 1e-16)

            delta = (u_new - u_act).abs().max(dim=1).values
            u[active] = u_new
            v[active] = v_new

            active = active[delta > tol]

        seq_flat = seq.reshape(M)
        col_sub = mapping[seq_flat]  # (M,)
        col_sub = torch.where(col_sub < 0, col_sub.new_full((), dust_idx), col_sub)

        if normalize:
            u_max, _ = torch.max(u, dim=1, keepdim=True)
            u_scale = u / torch.where(u_max == 0, torch.ones_like(u_max), u_max)
            tiny = torch.finfo(torch.float32).tiny
            u_scale = torch.clamp(u_scale, min=tiny)
            f_sub = epsilon * torch.log(u_scale)
            f_sub = f_sub - f_sub.mean(dim=1, keepdim=True)
        else:
            f_sub = epsilon * torch.log(u + 1e-16)  # (M,Nsub)
        f_lab = f_sub[torch.arange(M, device=device), col_sub]  # (M,)
        f_labels = f_lab.view(B, T)

        del u, v, pi_r_sub, pi_t_sub, Kv, Ktu, f_sub
        torch.cuda.empty_cache()

        return f_labels

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask,
                        logits, logits_ref, seq,
                        K, Kt, epsilon=0.01, max_iter=500, tol=1e-6, return_kl=False, top_k=None, normalize=False):

        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1) + 1

        penalty_start = time.time()
        if torch.abs(logits - logits_ref).sum() < 1e-10:
            print('same logits')
            kl_divergence_estimate = torch.zeros_like(log_probs)
        else:
            with torch.no_grad():
                pi_r = torch.softmax(logits_ref, dim=-1)
                pi_theta = torch.softmax(logits, dim=-1)
                f_labels = self.sparse_sinkhorn_sequence_topk(pi_r, pi_theta, K, start, ends, epsilon=epsilon, max_iter=max_iter, tol=tol, seq=seq, top_k=top_k, normalize=normalize)
                kl_divergence_estimate = - self.kl_ctl * f_labels
        penalty_end = time.time()
        print_rank_0(f"Regularization => Latency: {penalty_end - penalty_start:.4f}")

        rewards = torch.clone(kl_divergence_estimate)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)

        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        if return_kl:
            for k in range(batch_size):
                kl_divergence_estimate[k, :start] = 0
                kl_divergence_estimate[k, ends[k]:] = 0
            kl_divergence_estimate = (kl_divergence_estimate.sum(dim=1) / (ends - start)) / self.kl_ctl
            return rewards, kl_divergence_estimate
        return rewards

    def train_rlhf(self, inputs, constituent_tree=None, K=None, Kt=None, epsilon=0.01, max_iter=500, tol=1e-6, top_k=None, normalize=False):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        ppl = inputs['ppl']
        logits = inputs['logits']
        logits_ref = inputs['logits_ref']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        inputs['resp_length'] = torch.tensor(seq[:, start:].size(1)).to(seq)

        old_values = values
        with torch.no_grad():
            old_rewards, kl_divergence = self.compute_rewards(prompts, log_probs,
                                                              ref_log_probs, reward_score,
                                                              action_mask,
                                                              logits, logits_ref, seq[:, 1:],
                                                              K, Kt, epsilon=epsilon, max_iter=max_iter, tol=tol,
                                                              return_kl=True, top_k=top_k, normalize=normalize)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        return actor_loss, critic_loss, kl_divergence / self.kl_ctl

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)
