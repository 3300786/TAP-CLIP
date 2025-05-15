import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptAdjustor(nn.Module):
    def __init__(self, method='scale', prompt_dim=512):
        """
        method: 'scale' | 'gate' | 'residual' | 'mlp' | 'attn'
        """
        super().__init__()
        self.method = method
        self.prompt_dim = prompt_dim

        if method == 'gate':
            self.gate_net = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        elif method == 'residual':
            self.residual_net = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, prompt_dim)
            )
        elif method == 'mlp':
            self.mlp_net = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, prompt_dim)
            )
        elif method == 'attn':
            self.attn_q = nn.Linear(1, prompt_dim)
            self.attn_k = nn.Linear(prompt_dim, prompt_dim)
            self.attn_v = nn.Linear(prompt_dim, prompt_dim)
            self.dropout = nn.Dropout(0.1)

    def forward(self, prompt_embed, attribution_score):
        """
        prompt_embed: [B, T, D]
        attribution_score: [B, T]
        """
        B, T, D = prompt_embed.shape
        a = attribution_score.unsqueeze(-1)  # [B, T, 1]

        if self.method == 'scale':
            return prompt_embed * a

        elif self.method == 'gate':
            g = self.gate_net(a)  # [B, T, 1]
            return prompt_embed * g

        elif self.method == 'residual':
            delta = self.residual_net(a)  # [B, T, D]
            return prompt_embed + delta

        elif self.method == 'mlp':
            delta = self.mlp_net(a)  # [B, T, D]
            return prompt_embed + delta

        elif self.method == 'attn':
            q = self.attn_q(a)                    # [B, T, D]
            k = self.attn_k(prompt_embed)         # [B, T, D]
            v = self.attn_v(prompt_embed)         # [B, T, D]

            attn_score = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5), dim=-1)  # [B, T, T]
            attn_out = torch.matmul(attn_score, v)  # [B, T, D]
            return prompt_embed + self.dropout(attn_out)

        else:
            raise ValueError(f"Unknown method: {self.method}")
