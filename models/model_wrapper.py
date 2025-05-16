import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner
from utils.eval_metrics import attribution_entropy, attribution_variance


class FullModel(nn.Module):
    def __init__(self, class_names, clip_wrapper, prompt_len=5, attr_lambda=0.05, stab_lambda=0.1,
                 adjustor_method='scale', class_specific=False, warmup_epoch=0):
        super().__init__()
        self.clip = clip_wrapper
        self.class_names = class_names
        self.prompt_learner = PromptLearner(class_names, clip_wrapper, prompt_len, class_specific)
        self.n_cls = len(class_names)
        self.attribution_monitor = AttributionMonitor(prompt_len)
        self.prompt_adjustor = PromptAdjustor(method=adjustor_method)

        self.attr_lambda = attr_lambda
        self.stab_lambda = stab_lambda
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.warmup_epoch = warmup_epoch

        self.training_epoch = 0  # ✅ 用于外部控制当前 epoch（推荐在 train.py 中手动设置）

    def forward(self, images, labels=None, return_attribution=False):
        B = images.size(0)

        raw_prompt = self.prompt_learner()
        prompt_len = self.prompt_learner.prompt_len
        context_prompt = raw_prompt[:, :prompt_len, :]
        class_tokens = raw_prompt[:, prompt_len:, :]

        image_feat = self.clip.encode_image(images)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        logits = []
        all_attributions = []

        class_names = list(self.prompt_learner.context_bank.keys())
        for i, class_name in enumerate(class_names):
            ctx = context_prompt[i].unsqueeze(0).expand(B, -1, -1)
            cls = class_tokens[i].unsqueeze(0).expand(B, -1, -1)
            full_prompt = torch.cat([ctx, cls], dim=1)

            attributions = []
            for b in range(B):
                single_prompt = full_prompt[b].unsqueeze(0)
                self.clip.reset()
                _ = self.clip.model.transformer(single_prompt)
                attn_map = self.clip.get_attention_map().to(images.device)
                if attn_map.dim() == 2:
                    attn_map = attn_map.unsqueeze(0)
                attr = self.attribution_monitor(attn_map)
                attributions.append(attr)

            attribution = torch.cat(attributions, dim=0)  # [B, prompt_len]
            all_attributions.append(attribution)

            adjusted_ctx = self.prompt_adjustor(ctx, attribution)
            adjusted_prompt = torch.cat([adjusted_ctx, cls], dim=1)

            text_feat = self.clip.model.transformer(adjusted_prompt)
            text_feat = text_feat[torch.arange(B), -1, :]
            text_feat = text_feat @ self.clip.model.text_projection
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            sim = self.logit_scale.exp() * (image_feat * text_feat).sum(dim=-1, keepdim=True)
            logits.append(sim)

        logits = torch.cat(logits, dim=1)  # [B, n_cls]
        outputs = {"logits": logits}
        # if labels is not None or return_attribution:
        all_attr = torch.stack(all_attributions, dim=1)  # [B, C, prompt_len]

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)
            all_attr = torch.stack(all_attributions, dim=1)  # [B, C, prompt_len]

            if True or self.training_epoch < self.warmup_epoch:
                loss_total = loss_cls
                print(f"🔧 Warmup epoch {self.training_epoch + 1}/{self.warmup_epoch}: skipping attribution regularization")
            else:
                entropy_loss = attribution_entropy(all_attr.mean(dim=1))
                variance_loss = attribution_variance(all_attr.mean(dim=1), labels)
                loss_total = loss_cls + self.stab_lambda * entropy_loss + self.attr_lambda * variance_loss
                outputs.update({
                    "loss_entropy": entropy_loss,
                    "loss_variance": variance_loss
                })

            outputs.update({
                "loss": loss_total,
                "loss_cls": loss_cls
            })

        if return_attribution:
            outputs["attribution"] = all_attr.mean(dim=1)  # [B, prompt_len]

        return outputs
