import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner
from utils.eval_metrics import attribution_entropy, attribution_variance
from math import cos, pi


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
        self.training_epoch = 0  # for warmup schedule externally

    def forward(self, images, labels=None, return_attribution=False):
        B = images.size(0)

        # Prepare prompts
        raw_prompt = self.prompt_learner()  # [C, T, D]
        prompt_len = self.prompt_learner.prompt_len + self.prompt_learner.prefix_len
        context_prompt = raw_prompt[:, :prompt_len, :]
        class_tokens = raw_prompt[:, prompt_len:, :]

        image_feat = self.clip.encode_image(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # === One-pass Attribution ===
        prompt_list = []
        for i in range(self.n_cls):
            ctx = context_prompt[i].unsqueeze(0).expand(B, -1, -1)
            cls = class_tokens[i].unsqueeze(0).expand(B, -1, -1)
            prompt = torch.cat([ctx, cls], dim=1)
            prompt_list.append(prompt)
        all_prompts = torch.cat(prompt_list, dim=0)  # [B*C, T, D]

        with torch.no_grad():
            attn_map = self.clip.get_attention_weights_from_prompt_embedding(all_prompts)

        # ✅ 修复 reshape 报错：动态获取 B、C 并 reshape 成 [C, B, T, T]
        BxC, T, T2 = attn_map.shape
        assert T == T2, f"Expected square attention map, but got shape {attn_map.shape}"

        expected = B * self.n_cls
        if attn_map is None or attn_map.shape[0] != expected:
            raise RuntimeError(f"⚠️ Expected {expected} attention maps (B={B}, C={self.n_cls}), "
                               f"but got {attn_map.shape if attn_map is not None else 'None'}")
        attn_map = attn_map.view(B, self.n_cls, T, T).permute(1, 0, 2, 3)  # → [C, B, T, T]

        all_attr = []
        adjusted_prompts = []
        for i in range(self.n_cls):
            attr = self.attribution_monitor(attn_map[i])  # [B, prompt_len]
            ctx = context_prompt[i].unsqueeze(0).expand(B, -1, -1)  # [B, prefix + prompt, D]
            cls = class_tokens[i].unsqueeze(0).expand(B, -1, -1)

            prefix = ctx[:, :self.prompt_learner.prefix_len, :]  # [B, prefix_len, D]
            ctx_only = ctx[:, self.prompt_learner.prefix_len:, :]  # [B, prompt_len, D]
            adjusted_ctx_only = self.prompt_adjustor(ctx_only, attr)  # [B, prompt_len, D]
            adjusted_ctx = torch.cat([prefix, adjusted_ctx_only], dim=1)

            adjusted_prompt = torch.cat([adjusted_ctx, cls], dim=1)

            all_attr.append(attr)
            adjusted_prompts.append(adjusted_prompt)

        all_attr = torch.stack(all_attr, dim=1)  # [B, C, prompt_len]
        adjusted_prompts = torch.stack(adjusted_prompts, dim=1)  # [B, C, T, D]

        # === One-pass Text Encoding ===
        flat_prompts = adjusted_prompts.view(B * self.n_cls, -1, adjusted_prompts.size(-1))
        with torch.no_grad():
            text_feat = self.clip.encode_text_from_embeddings(flat_prompts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.view(B, self.n_cls, -1)

        sim = self.logit_scale.exp() * (image_feat.unsqueeze(1) * text_feat).sum(dim=-1)
        logits = sim

        outputs = {"logits": logits}

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            scale = 0.5 * (1 - cos(pi * self.training_epoch / self.warmup_epoch)) \
                if self.training_epoch < self.warmup_epoch else 1.0

            entropy_loss = attribution_entropy(all_attr.mean(dim=1))
            variance_loss = attribution_variance(all_attr.mean(dim=1), labels)
            loss_total = (
                loss_cls
                + scale * self.stab_lambda * entropy_loss
                + scale * self.attr_lambda * variance_loss
            )

            outputs.update({
                "loss": loss_total,
                "loss_cls": loss_cls,
                "loss_entropy": entropy_loss,
                "loss_variance": variance_loss
            })

        if return_attribution:
            outputs["attribution"] = all_attr.mean(dim=1)  # [B, prompt_len]

        return outputs
