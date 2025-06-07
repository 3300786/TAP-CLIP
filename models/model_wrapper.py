import torch
import torch.nn as nn
import torch.nn.functional as F
from math import cos, pi

from models.attribution_monitor import AttributionMonitor
from models.prompt_adjustor import PromptAdjustor
from models.prompt_learner import PromptLearner
from utils.eval_metrics import attribution_entropy, attribution_variance


class FullModel(nn.Module):
    """TAP‑CLIP ↔ PromptStyler 版本

    • Prompt  = prefix  +  S_k (1 token)  +  ctx  +  class‑token
    • 对每个类别循环 K_style 个 style‑token 后取平均
    • 新增两项损失:
        ▸ L_con   内容保持
        ▸ L_div_s style 多样性
    """

    def __init__(
            self,
            class_names,
            clip_wrapper,
            prompt_len: int = 5,
            prefix_len: int = 5,
            K_style: int = 8,
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

        # ── Prompt 相关 ────────────────────────────────────────────────────────
        # optional gradient checkpointing (only if available)
        if hasattr(clip_wrapper.model.transformer, "gradient_checkpointing_enable"):
            print("YES")
            clip_wrapper.model.transformer.gradient_checkpointing_enable()
        #clip_wrapper.model.transformer.gradient_checkpointing_enable()

        self.prompt_learner = PromptLearner(class_names, clip_wrapper, prompt_len, prefix_len)
        self.attribution_monitor = AttributionMonitor(prompt_len, prefix_len)
        self.prompt_adjustor = PromptAdjustor(method=adjustor_method)
        self.K_style = K_style

        # ── 损失权重 ────────────────────────────────────────────────────────────
        self.attr_lambda = attr_lambda  # attribution variance
        self.stab_lambda = stab_lambda  # entropy
        self.lambda_div_s = lambda_div_s  # style diversity
        self.lambda_con = lambda_con  # content consistency

        # ── 其它 ────────────────────────────────────────────────────────────────
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07)))
        self.warmup_epoch = warmup_epoch
        self.training_epoch = 0

        # plain cache 在 build 时创建 (shape: [C, D])
        self._plain_cache = None

    # ---------------------------------------------------------------------
    @property
    def plain_ready(self):
        return self._plain_cache is not None

    # ---------------------------------------------------------------------
    def _build_plain_cache(self, prefix, ctx_bank, token_bank, device):
        """预计算每个类别的 *plain* 文本特征 (无 style‑token)"""
        feats = []
        for cls in self.class_names:
            pre = prefix.unsqueeze(0)  # 1×P×D
            ctx = ctx_bank[cls].unsqueeze(0)  # 1×C×D
            token_emb = self.clip.model.token_embedding(token_bank[cls].to(device)).unsqueeze(0)
            prompt = torch.cat([pre, ctx, token_emb], dim=1)  # 1×T×D
            with torch.no_grad():
                feat = self.manual_encode_text(prompt)  # 1×D
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.squeeze(0))
        self._plain_cache = torch.stack(feats, dim=0).detach()  # C×D

    # ---------------------------------------------------------------------
    def forward(self, images, labels=None, return_attribution=False):
        B, device = images.size(0), images.device

        # warm‑up cosine scale
        if self.warmup_epoch > 0:
            scale = 0.5 * (1 - cos(pi * min(self.training_epoch, self.warmup_epoch) / self.warmup_epoch))
        else:
            scale = 1.0

        # ── Encode image ───────────────────────────────────────────────────────
        img_feat = self.clip.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # ── Prompt parts ───────────────────────────────────────────────────────
        prefix, ctx_bank, token_bank = self.prompt_learner.get_prompt_parts()
        style_tokens = self.prompt_learner.get_style_tokens()  # K×1×D

        # build plain cache once
        if not self.plain_ready:
            self._build_plain_cache(prefix, ctx_bank, token_bank, device)

        entropy_buf, variance_buf = [], []
        text_feats_all_cls = []  # list[ B×D ]

        for cls in self.class_names:
            pre = prefix.unsqueeze(0).expand(B, -1, -1)  # B×P×D
            ctx = ctx_bank[cls].unsqueeze(0).expand(B, -1, -1)  # B×C×D
            # ---- class-token embedding with dynamic truncation (ensure ≤77) ----
            tok_ids = token_bank[cls].to(device)
            if tok_ids.dim() == 2 and tok_ids.size(0) == 1:
                tok_ids = tok_ids.squeeze(0)
            avail_len = 77 - pre.size(1) - 1 - ctx.size(1)  # 77 − prefix − style − ctx
            if tok_ids.size(0) > avail_len:
                tok_ids = tok_ids[:avail_len]
            token = self.clip.model.token_embedding(tok_ids).unsqueeze(0).expand(B, -1, -1)
            # -- memory‑efficient style loop -------------------------------------
            text_feat_sum = 0.0
            for k in range(self.K_style):
                s = style_tokens[k].expand(B, -1, -1)  # B×1×D
                prompt_raw = torch.cat([pre, s, ctx, token], dim=1)

                # attention & attribution (no grad)
                with torch.no_grad():
                    attn_map = self.clip.get_attention_weights_from_prompt_embedding(prompt_raw)
                    attn_map = attn_map.view(B, attn_map.size(-1), attn_map.size(-1))
                attr = self.attribution_monitor(attn_map)

                adj_ctx = (1 - scale) * ctx + scale * self.prompt_adjustor(ctx, attr)
                prompt_adj = torch.cat([pre, s, adj_ctx, token], dim=1)

                # manual_encode_text wrapped in torch.checkpoint for mem‑saving
                feat = torch.utils.checkpoint.checkpoint(self.manual_encode_text, prompt_adj)
                feat = feat / feat.norm(dim=-1, keepdim=True)

                text_feat_sum = text_feat_sum + feat / self.K_style  # running average

                entropy_buf.append(attr.detach())
                variance_buf.append(attr.detach())

            text_feats_all_cls.append(text_feat_sum)

        # ── Assemble class matrix & logits ────────────────────────────────────
        text_feat = torch.stack(text_feats_all_cls, dim=1)  # B×C×D
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * (img_feat.unsqueeze(1) * text_feat).sum(dim=-1)
        outputs = {"logits": logits}

        # ── Loss计算 ───────────────────────────────────────────────────────────
        if labels is not None:
            # 分类 loss
            loss_cls = F.cross_entropy(logits, labels)

            # 归因 regularizers
            entropy_loss = attribution_entropy(torch.stack(entropy_buf, dim=0).mean(0))
            variance_loss = attribution_variance(torch.stack(variance_buf, dim=0).mean(0), labels)

            # 类间 diversity (旧)
            cos_sim = F.cosine_similarity(text_feat.unsqueeze(2), text_feat.unsqueeze(1), dim=-1)
            div_prompt = ((cos_sim - torch.eye(self.n_cls, device=device)) ** 2).mean()

            # 内容保持
            plain_feats = self._plain_cache.to(device)[labels]  # B×D
            con_loss = ((text_feat[torch.arange(B), labels] - plain_feats) ** 2).mean()

            # style diversity
            S = self.prompt_learner.style_tokens  # K×D
            sim_mat = F.cosine_similarity(S.unsqueeze(1), S.unsqueeze(0), dim=-1)
            div_style = (sim_mat * (1 - torch.eye(self.K_style, device=device))).mean()

            loss_total = (loss_cls
                          + scale * (self.stab_lambda * entropy_loss
                                     + self.attr_lambda * variance_loss
                                     + 0.1 * div_prompt)
                          + self.lambda_con * con_loss
                          + self.lambda_div_s * div_style)

            outputs.update({
                "loss": loss_total,
                "loss_cls": loss_cls,
                "loss_entropy": entropy_loss,
                "loss_variance": variance_loss,
                "loss_div_prompt": div_prompt,
                "loss_con": con_loss,
                "loss_div_style": div_style,
            })

        if return_attribution:
            outputs["attribution"] = torch.stack(entropy_buf, dim=0).mean(0)

        return outputs

    # ---------------------------------------------------------------------
    def manual_encode_text(self, prompt_embed: torch.Tensor) -> torch.Tensor:
        """Forward pre‑embedded prompts through CLIP text tower (frozen)."""
        B, T, _ = prompt_embed.shape
        pos = self.clip.model.positional_embedding[:T, :].unsqueeze(0).to(prompt_embed.device)
        x = prompt_embed + pos
        x = self.clip.model.transformer(x)
        x = self.clip.model.ln_final(x)
        return x[torch.arange(B), -1] @ self.clip.model.text_projection
