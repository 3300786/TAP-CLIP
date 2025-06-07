import torch
import torch.nn as nn
import open_clip
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=16, alpha=32, dropout=0.05):
        super().__init__()
        # 保留 bias 与原 Linear 一致（默认 True）
        self.linear = nn.Linear(in_f, out_f, bias=True)
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # 🔑 关键：显式把 weight / bias 暴露出去
        self.weight = self.linear.weight
        self.bias = self.linear.bias  # 若原来没 bias，可设置为 None

    def forward(self, x):
        # 与 nn.Linear 行为保持一致
        return F.linear(x, self.weight, self.bias) + \
            (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class CLIPWrapper(nn.Module):

    def __init__(self, model_name='ViT-B-16', pretrained_path='path/to/open_clip_pytorch_model_16.bin', device='cuda',
                 lora_layers=16):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='')
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()
        # self.model.transformer.gradient_checkpointing_enable()
        # self.model.visual.transformer.gradient_checkpointing_enable()
        for p in self.model.parameters():
            p.requires_grad = False

        self.add_lora_to_visual(self.model.visual, r=lora_layers, n_last=4)

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.token_embedding = self.model.token_embedding

    def add_lora_to_visual(self, visual, r=16, alpha=32,
                           dropout=0.05, n_last=4, prefix="lora"):

        def convert_linear(orig):
            if isinstance(orig, nn.Linear):
                lora = LoRALinear(orig.in_features, orig.out_features,
                                  r=r, alpha=alpha, dropout=dropout)
                # 复制权重 / bias
                lora.weight.data.copy_(orig.weight.data)
                if orig.bias is not None:
                    lora.bias.data.copy_(orig.bias.data)
                return lora
            return orig

        for i, blk in enumerate(visual.transformer.resblocks):
            if i < len(visual.transformer.resblocks) - n_last:
                continue

            # 1) Attention 子模块
            if isinstance(blk.attn, nn.MultiheadAttention):
                # 仅替换 out_proj (nn.Linear)
                blk.attn.out_proj = convert_linear(blk.attn.out_proj)
            else:  # open-clip <= v1 风格 (Attention with qkv/proj)
                for attr in ["qkv", "proj"]:
                    if hasattr(blk.attn, attr):
                        setattr(blk.attn, attr,
                                convert_linear(getattr(blk.attn, attr)))

            # 2) MLP 子模块
            for attr in ["fc1", "fc2"]:
                if hasattr(blk.mlp, attr):
                    setattr(blk.mlp, attr,
                            convert_linear(getattr(blk.mlp, attr)))

    def encode_image(self, image_tensor):
        return self.model.encode_image(image_tensor)

    def encode_text(self, token_tensor):
        self.reset()
        return self.model.encode_text(token_tensor)

    def get_tokenizer(self):
        return self.tokenizer

    def get_preprocess(self):
        return self.preprocess

    def get_max_text_len(self):
        return self.model.positional_embedding.size(0)

    def get_attention_weights_from_prompt_embedding(self, prompt_embeddings, chunk_size=4):
        BxC, T, D = prompt_embeddings.size()
        device = prompt_embeddings.device
        max_pos = self.model.positional_embedding.size(0)

        if T > max_pos:
            raise ValueError(f"🚫 Prompt token length ({T}) exceeds CLIP max position ({max_pos})")

        pos_emb = self.model.positional_embedding[:T, :].unsqueeze(0).to(device)
        outputs = []

        for i in range(0, BxC, chunk_size):
            chunk = prompt_embeddings[i: i + chunk_size]
            x = chunk + pos_emb

            for block in self.model.transformer.resblocks[:-1]:
                x = block(x)

            last_block = self.model.transformer.resblocks[-1]
            x_norm = last_block.ln_1(x)

            attn_out, attn_weights = last_block.attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=True,
                average_attn_weights=False
            )

            if attn_weights.dim() != 4 or attn_weights.shape[-2:] != (T, T):
                raise RuntimeError(f"❌ Unexpected attn_weights shape: {attn_weights.shape}, expected [B, H, T, T]")

            attn_mean = attn_weights.mean(dim=1)
            outputs.append(attn_mean)

        return torch.cat(outputs, dim=0)
