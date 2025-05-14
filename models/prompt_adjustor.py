# models/prompt_adjustor.py

import torch
import torch.nn as nn

class PromptAdjustor(nn.Module):
    def __init__(self, method='scale'):
        """
        method: 'scale' | 'gate' | 'residual' 可扩展
        """
        super().__init__()
        self.method = method
        if self.method == 'gate':
            self.gate_net = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif self.method == 'residual':
            self.residual_net = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 512)  # 需与 embedding dim 对应
            )

    def forward(self, prompt_embed, attribution_score):
        """
        prompt_embed: shape [B, prompt_len, dim]
        attribution_score: shape [B, prompt_len]
        """
        B, T, D = prompt_embed.shape
        a = attribution_score.unsqueeze(-1)  # [B, T, 1]

        if self.method == 'scale':
            return prompt_embed * a  # Linear scaling

        elif self.method == 'gate':
            g = self.gate_net(a)  # [B, T, 1]
            return prompt_embed * g

        elif self.method == 'residual':
            delta = self.residual_net(a)  # [B, T, D]
            return prompt_embed + delta

        else:
            raise ValueError(f"Unknown method: {self.method}")
