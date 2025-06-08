import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, pi
from torch.utils.checkpoint import checkpoint

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner
from utils.eval_metrics import attribution_entropy, attribution_variance


class FullModel(nn.Module):
    """TAP‑CLIP + PromptStyler (内存友好版)

    Prompt = prefix + S_k(1) + ctx + class‑token
    对每类平均 K_style 个风格 token。
    内存要点：
        • text transformer gradient‑checkpoint
        • style 循环 running‑mean，无 stack
        • attribution buf 存 CPU mean，避免累涨
    """

    def __init__(
        self,
        class_names,
        clip_wrapper,
        prompt_len: int = 5,
        prefix_len: int = 5,
        K_style: int = 4,
        attr_lambda: float = 0.05,
        stab_lambda: float = 0.1,
        lambda_div_s: float = 0.05,
        lambda_con: float = 0.1,
        adjustor_method: str = "scale",
        warmup_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.clip = clip_wrapper
        self.class_names = class_names
        self.n_cls = len(class_names)
        self.K_style = K_style

        # gradient checkpoint, if available
        if hasattr(clip_wrapper.model.transformer, "gradient_checkpointing_enable"):
            clip_wrapper.model.transformer.gradient_checkpointing_enable()

        # prompt components
        self.prompt_learner = PromptLearner(class_names, clip_wrapper, prompt_len, prefix_len)
        self.attribution_monitor = AttributionMonitor(prompt_len, prefix_len)
        self.prompt_adjustor = PromptAdjustor(method=adjustor_method)

        # loss weights
        self.attr_lambda = attr_lambda
        self.stab_lambda = stab_lambda
        self.lambda_div_s = lambda_div_s
        self.lambda_con = lambda_con

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.warmup_epoch = warmup_epoch
        self.training_epoch = 0

        self._plain_cache = None  # to be built at first forward

    # ------------------------------------------------------------------
    @property
    def plain_ready(self):
        return self._plain_cache is not None

    def _build_plain_cache(self, prefix, ctx_bank, token_bank, device):
        feats = []
        for cls in self.class_names:
            prompt = torch.cat([
                prefix.unsqueeze(0),
                ctx_bank[cls].unsqueeze(0),
                self.clip.model.token_embedding(token_bank[cls].to(device)).unsqueeze(0),
            ], dim=1)
            with torch.no_grad():
                f = self.manual_encode_text(prompt)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.squeeze(0))
        self._plain_cache = torch.stack(feats, 0).cpu()

    # ------------------------------------------------------------------
    def forward(self, images, labels=None, return_attribution=False):
        B, device = images.size(0), images.device

        scale = 1.0 if self.warmup_epoch == 0 else 0.5 * (1 - cos(pi * min(self.training_epoch, self.warmup_epoch) / self.warmup_epoch))

        img_feat = self.clip.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        prefix, ctx_bank, token_bank = self.prompt_learner.get_prompt_parts()
        style_tokens = self.prompt_learner.get_style_tokens()  # K×1×D

        if not self.plain_ready:
            self._build_plain_cache(prefix, ctx_bank, token_bank, device)

        entropy_buf_cpu = []  # store mean attr per sample on CPU
        var_buf_cpu = []
        text_feat_cls = []

        for cls in self.class_names:
            pre = prefix.unsqueeze(0).expand(B, -1, -1)
            ctx = ctx_bank[cls].unsqueeze(0).expand(B, -1, -1)

            tok_ids = token_bank[cls]
            if tok_ids.dim() == 2 and tok_ids.size(0) == 1:
                tok_ids = tok_ids.squeeze(0)
            avail = 77 - pre.size(1) - 1 - ctx.size(1)
            tok_ids = tok_ids[:avail]
            token = self.clip.model.token_embedding(tok_ids.to(device)).unsqueeze(0).expand(B, -1, -1)

            running_sum = 0.0
            for k in range(self.K_style):
                s = style_tokens[k].expand(B, -1, -1)
                prompt_raw = torch.cat([pre, s, ctx, token], 1)

                with torch.no_grad():
                    attn = self.clip.get_attention_weights_from_prompt_embedding(prompt_raw)
                    attn = attn.view(B, attn.size(-1), attn.size(-1))
                attr = self.attribution_monitor(attn)  # B×prompt_len

                adj_ctx = (1 - scale) * ctx + scale * self.prompt_adjustor(ctx, attr)
                prompt_adj = torch.cat([pre, s, adj_ctx, token], 1)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feat = checkpoint(self.manual_encode_text, prompt_adj)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                running_sum = running_sum + feat / self.K_style

                entropy_buf_cpu.append(attr.mean(1).cpu())
                var_buf_cpu.append(attr.mean(1).cpu())

            text_feat_cls.append(running_sum)

        text_feat = torch.stack(text_feat_cls, 1)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * (img_feat.unsqueeze(1) * text_feat).sum(-1)
        outputs = {"logits": logits}

        if labels is not None:
            loss_cls = F.cross_entropy(logits, labels)

            entropy_tensor = torch.stack(entropy_buf_cpu, 0).to(device)
            ent_loss = attribution_entropy(entropy_tensor.mean(0))
            var_loss = attribution_variance(entropy_tensor.mean(0), labels)

            cos_sim = F.cosine_similarity(text_feat.unsqueeze(2), text_feat.unsqueeze(1), dim=-1)
            div_prompt = ((cos_sim - torch.eye(self.n_cls, device=device)) ** 2).mean()

            con_loss = ((text_feat[torch.arange(B), labels] - self._plain_cache.to(device)[labels]) ** 2).mean()

            S = self.prompt_learner.style_tokens
            div_style = (F.cosine_similarity(S.unsqueeze(1), S.unsqueeze(0), dim=-1) * (1 - torch.eye(self.K_style, device=device))).mean()

            loss_total = (loss_cls + scale * (self.stab_lambda * ent_loss + self.attr_lambda * var_loss + 0.1 * div_prompt) + self.lambda_con * con_loss + self.lambda_div_s * div_style)

            outputs.update(dict(loss=loss_total, loss_cls=loss_cls))

        if return_attribution:
            outputs["attribution"] = entropy_tensor.mean(0)

        return outputs

    # ------------------------------------------------------------------
    def manual_encode_text(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        B, T, _ = prompt_embed.shape
        pos = self.clip.model.positional_embedding[:T, :].unsqueeze(0).to(prompt_embed.device)
        x = prompt_embed + pos
        x = self.clip.model.transformer(x)
        x = self.clip.model.ln_final(x)
        return x[torch.arange(B), -1] @ self.clip.model.text_projection
