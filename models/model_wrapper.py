import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, pi

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner
from utils.eval_metrics import attribution_entropy, attribution_variance


class FullModel(nn.Module):
    def __init__(self, class_names, clip_wrapper, prompt_len=5, prefix_len=5,
                 attr_lambda=0.05, stab_lambda=0.1, adjustor_method='scale',
                 class_specific=True, warmup_epoch=0):
        super().__init__()
        self.clip = clip_wrapper
        self.class_names = class_names
        self.prompt_learner = PromptLearner(
            class_names, clip_wrapper, prompt_len, prefix_len
        )
        self.n_cls = len(class_names)
        self.attribution_monitor = AttributionMonitor(prompt_len, prefix_len)
        self.prompt_adjustor = PromptAdjustor(method=adjustor_method)

        self.attr_lambda = attr_lambda
        self.stab_lambda = stab_lambda
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.warmup_epoch = warmup_epoch
        self.training_epoch = 0

    def forward(self, images, labels=None, return_attribution=False):
        B = images.size(0)
        max_len = self.clip.get_max_text_len()

        # 平滑插值系数：在 warmup_epoch 前逐渐从 0 增长至 1
        scale = 0.5 * (1 - cos(pi * self.training_epoch / self.warmup_epoch)) \
            if self.training_epoch < self.warmup_epoch else 1.0

        prefix, context_bank, token_id_bank = self.prompt_learner.get_prompt_parts()
        with torch.no_grad():
            image_feat = self.clip.encode_image(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        def get_token_embedding(cls):
            ids = token_id_bank[cls].to(images.device)
            if ids.dim() == 2 and ids.size(0) == 1:
                ids = ids.squeeze(0)
            emb = self.clip.model.token_embedding(ids)
            max_token_len = max_len - prefix.size(0) - context_bank[cls].size(0)
            if emb.size(0) > max_token_len:
                emb = emb[:max_token_len]
            return emb.unsqueeze(0)

        prompt_list = []
        all_attr = []
        for cls in self.class_names:
            pre = prefix.unsqueeze(0).expand(B, -1, -1)
            ctx = context_bank[cls].unsqueeze(0).expand(B, -1, -1)
            token = get_token_embedding(cls).expand(B, -1, -1)

            # 获取 attention 权重并做 attribution
            prompt_for_attn = torch.cat([pre, ctx, token], dim=1)
            attn_map = self.clip.get_attention_weights_from_prompt_embedding(prompt_for_attn)
            attn_map = attn_map.view(B, attn_map.size(-1), attn_map.size(-1))
            attr = self.attribution_monitor(attn_map)

            if self.training and torch.rand(1).item() < 0.05:
                print(
                    f"[ATTR] class={cls} mean={attr.mean().item():.4f}, std={attr.std().item():.4f}, scale={scale:.2f}")

            adjusted = self.prompt_adjustor(ctx, attr)
            adjusted_ctx = (1 - scale) * ctx + scale * adjusted  # 关键插值操作
            full_prompt = torch.cat([pre, adjusted_ctx, token], dim=1)
            prompt_list.append(full_prompt)
            all_attr.append(attr)

        adjusted_prompts = torch.stack(prompt_list, dim=1)  # [B, C, T, D]
        all_attr = torch.stack(all_attr, dim=1)  # [B, C, prompt_len]

        flat_prompts = adjusted_prompts.view(B * self.n_cls, -1, adjusted_prompts.size(-1))
        text_feat = self.manual_encode_text(flat_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.view(B, self.n_cls, -1)

        sim = self.logit_scale.exp() * (image_feat.unsqueeze(1) * text_feat).sum(dim=-1)
        logits = sim
        outputs = {"logits": logits}

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)

            entropy_loss = attribution_entropy(all_attr.mean(dim=1))
            variance_loss = attribution_variance(all_attr.mean(dim=1), labels)

            cos_sim = F.cosine_similarity(text_feat.unsqueeze(2), text_feat.unsqueeze(1), dim=-1)
            div_loss = ((cos_sim - torch.eye(self.n_cls, device=cos_sim.device)) ** 2).mean()

            loss_total = loss_cls + scale * (
                    self.stab_lambda * entropy_loss +
                    self.attr_lambda * variance_loss +
                    0.1 * div_loss
            )

            outputs.update({
                "loss": loss_total,
                "loss_cls": loss_cls,
                "loss_entropy": entropy_loss,
                "loss_variance": variance_loss,
                "loss_div": div_loss
            })

        if return_attribution:
            outputs["attribution"] = all_attr.mean(dim=1)

        return outputs

    def manual_encode_text(self, prompt_emb):
        # prompt_emb: [B*C, T, D]
        x = prompt_emb + self.clip.model.positional_embedding[:prompt_emb.size(1)].to(prompt_emb.device)  # [B*C, T, D]
        x = self.clip.model.transformer(x)  # 通常为 ResidualAttentionBlock stack
        x = self.clip.model.ln_final(x)
        # 通常第一个 token 是 [CLS] 或者用平均
        x = x[torch.arange(x.shape[0]), -1]  # 如果模型是 [EOS] 聚合
        return x  # shape [B*C, D]
