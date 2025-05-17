# models/attribution_monitor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributionMonitor(nn.Module):
    def __init__(self, prompt_len, prefix_len=0, normalize=True):
        """
        prompt_len: class-specific 可调 context token 的长度
        prefix_len: prefix prompt 的长度（共享，不参与归因调整）
        normalize: 是否对归因权重归一化（softmax）
        """
        super().__init__()
        self.prompt_len = prompt_len
        self.prefix_len = prefix_len
        self.normalize = normalize

    def forward(self, attn_map):
        """
        attn_map: Tensor of shape (B, T, T)，来自 CLIPWrapper 中最后一层 attention（已对 head 取平均）

        返回:
        attribution: Tensor of shape (B, prompt_len)，每个 context token 的归因值
        """
        B, T, _ = attn_map.shape
        ctx_start = self.prefix_len                     # context 起始位置
        ctx_end = ctx_start + self.prompt_len           # context 结束位置（不含）

        cls_token_index = T - 1  # 默认 class token 在末尾位置

        # 取 context token → class token 的 attention
        raw_score = attn_map[:, ctx_start:ctx_end, cls_token_index]  # [B, prompt_len]

        if self.normalize:
            attribution = F.softmax(raw_score, dim=-1)
        else:
            attribution = raw_score

        return attribution  # [B, prompt_len]
