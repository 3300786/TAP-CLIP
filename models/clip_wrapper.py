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
    def get_attention_weights_from_prompt_embedding(self, prompt_embeddings):
        """
        prompt_embeddings: [B, T, D]
        返回: attention map [B, T, T]（mean over heads, 可导）
        """
        T = prompt_embeddings.size(1)
        B = prompt_embeddings.size(0)
        D = prompt_embeddings.size(2)

        max_pos = self.model.positional_embedding.size(0)

        if T > max_pos:
            raise ValueError(f"🚫 Prompt token length ({T}) exceeds CLIP max position ({max_pos}). "
                             f"Reduce prompt_len or simplify class names.")

        # Positional embedding
        pos_emb = self.model.positional_embedding[:T, :].unsqueeze(0)  # [1, T, D]
        x = prompt_embeddings + pos_emb  # [B, T, D]

        for block in self.model.transformer.resblocks[:-1]:
            x = block(x)

        # 最后一层，提取注意力
        last_block = self.model.transformer.resblocks[-1]
        x_res, attn_weights = last_block.attn(x, need_weights=True)  # attn_weights: [B, H, T, T]
        attn_mean = attn_weights.mean(dim=1)  # [B, T, T]
        return attn_mean
