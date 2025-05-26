# models/clip_wrapper.py

import torch
import torch.nn as nn
import open_clip


class CLIPWrapper(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained_path='path/to/open_clip_pytorch_model.bin', device='cuda'):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='')
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.attention_maps = []
        self._register_text_attention_hook()

        self.tokenizer = open_clip.get_tokenizer(model_name)

        # ✅ 添加这一行：暴露 token embedding 层给外部访问
        self.token_embedding = self.model.token_embedding

    def _register_text_attention_hook(self):
        def hook_fn(module, input, output):
            # output[0]: [B, H, T, T] or sometimes [H, T, T]
            attn = output[0]
            if attn.dim() == 3:
                attn = attn.unsqueeze(0)  # [1, H, T, T]
            attn = attn.mean(dim=1)  # mean over heads → [B, T, T]
            self.attention_maps.append(attn.to(self.device))

        last_text_block = self.model.transformer.resblocks[-1].attn
        last_text_block.register_forward_hook(hook_fn)

    def reset(self):
        self.attention_maps.clear()

    def encode_image(self, image_tensor):
        return self.model.encode_image(image_tensor)

    def encode_text(self, token_tensor):
        self.reset()
        return self.model.encode_text(token_tensor)

    def get_attention_map(self):
        if len(self.attention_maps) == 0:
            return None
        return self.attention_maps[-1]  # [B, T, T]

    def get_tokenizer(self):
        return self.tokenizer

    def get_preprocess(self):
        return self.preprocess

    # ✅ 新增：从已构建 prompt embedding 中获得 attention
    def get_attention_weights_from_prompt_embedding(self, prompt_embeddings, chunk_size=4):
        """
        计算给定 prompt embedding 的 attention 权重（mean over heads, 可导）

        Args:
            prompt_embeddings: Tensor of shape [B*C, T, D]
            chunk_size: number of samples per forward pass to reduce memory usage

        Returns:
            Tensor of shape [B*C, T, T]: averaged attention maps
        """
        BxC, T, D = prompt_embeddings.size()
        device = prompt_embeddings.device
        max_pos = self.model.positional_embedding.size(0)

        if T > max_pos:
            raise ValueError(f"🚫 Prompt token length ({T}) exceeds CLIP max position ({max_pos})")

        pos_emb = self.model.positional_embedding[:T, :].unsqueeze(0).to(device)  # [1, T, D]
        outputs = []

        for i in range(0, BxC, chunk_size):
            chunk = prompt_embeddings[i: i + chunk_size]  # [chunk_size, T, D]
            x = chunk + pos_emb  # [chunk_size, T, D]

            for block in self.model.transformer.resblocks[:-1]:
                x = block(x)

            last_block = self.model.transformer.resblocks[-1]
            x_norm = last_block.ln_1(x)

            # Use attention module with need_weights=True to extract [B, H, T, T]
            attn_out, attn_weights = last_block.attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=True,
                average_attn_weights=False  # must be False to get full attn matrix
            )

            if attn_weights.dim() != 4 or attn_weights.shape[-2:] != (T, T):
                raise RuntimeError(f"❌ Unexpected attn_weights shape: {attn_weights.shape}, expected [B, H, T, T]")

            attn_mean = attn_weights.mean(dim=1)  # [chunk_size, T, T]
            outputs.append(attn_mean)

        return torch.cat(outputs, dim=0)  # [B*C, T, T]


    def get_max_text_len(self):
        return self.model.positional_embedding.size(0)
