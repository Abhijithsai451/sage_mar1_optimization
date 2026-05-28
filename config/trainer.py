import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="lora_output",
    optim="paged_adamw_8bit",
    report_to="none",
    num_train_epochs=4,
    remove_unused_columns=False,
    )


class SAGETrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta=0.04, eps_clip=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.eps_clip = eps_clip
        if self.ref_model is not None:
            self.ref_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        REINFORCE ++ Loss Calculation using the unified LoRA Model and the base fine-tuned model (ref_model)
        """
        # Forward Pass through the model
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits
        # Extract active sequence masks
        labels = inputs["labels"]
        loss_mask = (labels != -100).float()

        # Shift logits and labels for causal language modeling objective
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous()

        # Compute current log probabilities per token
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        per_token_log_probs = per_token_log_probs * shift_mask

        with torch.no_grad():
            peft_outputs = self.ref_model(
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask']
            )
            ref_logits = peft_outputs.logits[..., :-1, :].contiguous()
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            per_token_ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            per_token_ref_log_probs = per_token_ref_log_probs * shift_mask
        # Compute Token-Level KL Divergence - KL = ln(pi_theta / pi_ref)
        kl_divergence = (per_token_log_probs - per_token_ref_log_probs) * shift_mask

        # Global Advantage Normalization (REINFORCE++ Core Step)
        raw_rewards = inputs["rewards"]  # Shape: [batch_size]

        # Integrate total sequence KL into the reward penalty
        total_sequence_kl = kl_divergence.sum(dim=-1)
        adjusted_rewards = raw_rewards - (self.beta * total_sequence_kl)

        # Normalize over the entire global batch
        batch_mean = adjusted_rewards.mean()
        batch_std = adjusted_rewards.std() + 1e-8
        global_advantages = (adjusted_rewards - batch_mean) / batch_std

        # Broadcast advantages to match token dimensions [batch_size, seq_len]
        global_advantages = global_advantages.unsqueeze(-1) * shift_mask

        # Policy Gradient with Clipped Importance Weights
        old_log_probs = inputs["old_log_probs"][..., 1:].contiguous() * shift_mask

        # Calculate ratio: r_t(theta)
        ratios = torch.exp(per_token_log_probs - old_log_probs)

        # Core PPO-style surrogate objectives using global advantage
        surr1 = ratios * global_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * global_advantages

        # We minimize the negative policy objective
        policy_loss = -torch.min(surr1, surr2)

        # Mask out padded or prompt tokens, average over valid positions
        loss = (policy_loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

        return (loss, outputs) if return_outputs else loss


