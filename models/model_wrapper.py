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
        # self.clip.reset()

        B = images.size(0)
        max_len = self.clip.get_max_text_len()

        scale = 0.5 * (1 - cos(pi * self.training_epoch / self.warmup_epoch)) \
            if self.training_epoch < self.warmup_epoch else 1.0

        prefix, context_bank, token_id_bank = self.prompt_learner.get_prompt_parts()
        image_feat = self.clip.encode_image(images)  # 保留梯度
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        def get_token_embedding(cls):
            ids = token_id_bank[cls].to(images.device)
            if ids.dim() == 2 and ids.size(0) == 1:
                ids = ids.squeeze(0)
            emb = self.clip.model.token_embedding(ids)
            max_token_len = max_len - prefix.size(0) - context_bank[cls].size(0)
            if emb.size(0) > max_token_len:
                emb = emb[:max_token_len]
            return emb.unsqueeze(0).detach()  # ✅ 防止 token embedding 引发图扩散

        text_feats = []
        all_attr = []

        for cls in self.class_names:
            pre = prefix.unsqueeze(0).expand(B, -1, -1)  # ✅ 明确 detach
            ctx = context_bank[cls].unsqueeze(0).expand(B, -1, -1)
            token = get_token_embedding(cls).expand(B, -1, -1)

            with torch.no_grad():  # ✅ clip attention 不构建图
                attn_prompt = torch.cat([pre, ctx, token], dim=1)
                attn_map = self.clip.get_attention_weights_from_prompt_embedding(attn_prompt)
                attn_map = attn_map.view(B, attn_map.size(-1), attn_map.size(-1))

            attr = self.attribution_monitor(attn_map)  # [B, prompt_len]
            adjusted = self.prompt_adjustor(ctx, attr)  # ✅ 保留梯度
            adjusted_ctx = (1 - scale) * ctx + scale * adjusted  # ✅ 有梯度，只作用于 adjusted

            # ⚠️ 拼接 prompt 立即构造 text_feat，防止全保留计算图
            full_prompt = torch.cat([pre, adjusted_ctx, token], dim=1)

            # ✅ 一次 forward，一次编码，立即归一化后拼接，避免大图保留
            with torch.cuda.amp.autocast():
                chunk_feat = self.manual_encode_text(full_prompt)
                chunk_feat = chunk_feat / chunk_feat.norm(dim=-1, keepdim=True)
            text_feats.append(chunk_feat)
            all_attr.append(attr.detach())

        # ✅ 拼接结果（全为叶子节点，无旧图）
        text_feat = torch.stack(text_feats, dim=1)  # [B, C, D]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sim = self.logit_scale.exp() * (image_feat.unsqueeze(1) * text_feat).sum(dim=-1)
        logits = sim
        outputs = {"logits": logits}

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            entropy_loss = attribution_entropy(torch.stack(all_attr, dim=1).mean(dim=1))  # ✅ detached 了
            variance_loss = attribution_variance(torch.stack(all_attr, dim=1).mean(dim=1), labels)
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
            outputs["attribution"] = torch.stack(all_attr, dim=1).mean(dim=1)

        return outputs

    def manual_encode_text(self, prompt_emb):
        # prompt_emb: [B*C, T, D]
        x = prompt_emb + self.clip.model.positional_embedding[:prompt_emb.size(1)].to(prompt_emb.device)
        x = self.clip.model.transformer(x)
        x = self.clip.model.ln_final(x)
        x = x[torch.arange(x.shape[0]), -1]  # 最后一个 token（EOS）
        return x
