# models/attribution_monitor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributionMonitor(nn.Module):
    def __init__(self, prompt_len, normalize=True):
        """
        prompt_len: 可学习 prompt token 的长度（context token 数）
        normalize: 是否归一化 attribution 分数（映射到 [0, 1]）
        """
        super().__init__()
        self.prompt_len = prompt_len
        self.normalize = normalize

    def forward(self, attn_map):
        """
        attn_map: (B, T, T) attention map from CLIPWrapper (already mean over heads)

        返回:
        attribution: (B, prompt_len)，每个 context token 的归因值
        """
        B, T, _ = attn_map.shape
        ctx_range = self.prompt_len
        cls_token_index = T - 1  # 默认 class token 在末尾（或靠后位置）

        # 提取 context token 对 class token 的 attention 权重
        raw_score = attn_map[:, :ctx_range, cls_token_index]  # shape: [B, prompt_len]

        if self.normalize:
            attribution = F.softmax(raw_score, dim=-1)
        else:
            attribution = raw_score

        return attribution  # [B, prompt_len]
