# models/model_wrapper.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner


class FullModel(nn.Module):
    def __init__(self, class_names, clip_wrapper, prompt_len=5, attr_lambda=1.0, stab_lambda=0.1,
                 adjustor_method='scale', class_specific=False):
        super().__init__()
        self.clip = clip_wrapper
        # print(class_names)
        self.class_names = class_names
        self.prompt_learner = PromptLearner(class_names, clip_wrapper, prompt_len, class_specific)
        self.n_cls = len(class_names)
        self.attribution_monitor = AttributionMonitor(prompt_len)
        self.prompt_adjustor = PromptAdjustor(method=adjustor_method)

        self.attr_lambda = attr_lambda
        self.stab_lambda = stab_lambda
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, labels=None):
        B = images.size(0)

        # Step 1: 获取原始 Prompt
        raw_prompt = self.prompt_learner()  # [n_cls, total_len, dim]
        prompt_len = self.prompt_learner.prompt_len
        context_prompt = raw_prompt[:, :prompt_len, :]  # [n_cls, prompt_len, dim]
        class_tokens = raw_prompt[:, prompt_len:, :]  # [n_cls, cls_len, dim]
        # print("🧩 PromptLearner.n_cls:", self.prompt_learner.n_cls)
        # print("📦 raw_prompt.shape[0]:", raw_prompt.shape[0])

        # Step 2: 获取图像特征
        image_feat = self.clip.encode_image(images)  # [B, dim]
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        logits = []

        # ✅ 修复点：动态确定 prompt 数量，防止漏算类
        # for i in range(raw_prompt.shape[0]):
        class_names = list(self.prompt_learner.context_bank.keys())
        for i, class_name in enumerate(class_names):
            ctx = context_prompt[i].unsqueeze(0).expand(B, -1, -1)
            cls = class_tokens[i].unsqueeze(0).expand(B, -1, -1)
            full_prompt = torch.cat([ctx, cls], dim=1)  # [B, total_len, dim]

            # Step 3: Attribution loop
            attributions = []
            for b in range(B):
                single_prompt = full_prompt[b].unsqueeze(0)  # [1, T, D]
                self.clip.reset()
                _ = self.clip.model.transformer(single_prompt)
                attn_map = self.clip.get_attention_map()  # [T, T] or [1, T, T]
                if attn_map.dim() == 2:
                    attn_map = attn_map.unsqueeze(0)
                attr = self.attribution_monitor(attn_map)  # [1, prompt_len]
                attributions.append(attr)

            attribution = torch.cat(attributions, dim=0)  # [B, prompt_len]

            # Step 4: Prompt Adjustor
            adjusted_ctx = self.prompt_adjustor(ctx, attribution)  # [B, prompt_len, dim]
            adjusted_prompt = torch.cat([adjusted_ctx, cls], dim=1)  # [B, total_len, dim]

            # Step 5: 获取文本嵌入
            text_feat = self.clip.model.transformer(adjusted_prompt)  # [B, T, D]
            text_feat = text_feat[torch.arange(B), -1, :]
            text_feat = text_feat @ self.clip.model.text_projection  # [B, dim]
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # Step 6: cosine similarity
            # sim = (image_feat * text_feat).sum(dim=-1, keepdim=True)  # [B, 1]
            sim = self.logit_scale.exp() * (image_feat * text_feat).sum(dim=-1, keepdim=True)

            logits.append(sim)

        logits = torch.cat(logits, dim=1)  # [B, n_cls]
        # print("🚨 Logits shape:", logits.shape)
        # if labels is not None:
        #     print("🎯 Labels max value:", labels.max().item())

        outputs = {"logits": logits}

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            loss_total = loss_cls
            outputs.update({"loss": loss_total, "loss_cls": loss_cls})
        # 改成 CLIP 的 cosine similarity, 两个方向
        """
            logits_per_image = logit_scale * image_features @ text_features.t()  # [B, B]
            # 文本->图像方向
            logits_per_text = logit_scale * text_features @ image_features.t()  # [B, B]
        """
        return outputs


