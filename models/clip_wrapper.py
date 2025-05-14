# models/clip_wrapper.py

import torch
import torch.nn as nn
import open_clip
from pathlib import Path


class CLIPWrapper(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained_path='path/to/open_clip_pytorch_model.bin', device='cuda'):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='')
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()

        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False

        # Attention hook 容器
        self.attention_maps = []
        self._register_text_attention_hook()

        # 提取 tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _register_text_attention_hook(self):
        """
        注册 hook 到文本 transformer 的最后一层 attention，获取 attention 权重。
        """

        def hook_fn(module, input, output):
            # output[0]: attention weights, shape: [B, H, T, T]
            attn = output[0].detach().mean(dim=1)  # mean over heads -> shape: [B, T, T]
            self.attention_maps.append(attn)

        last_text_block = self.model.transformer.resblocks[-1].attn
        last_text_block.register_forward_hook(hook_fn)

    def reset(self):
        """清空 attention map 缓存"""
        self.attention_maps.clear()

    def encode_image(self, image_tensor):
        return self.model.encode_image(image_tensor)

    def encode_text(self, token_tensor):
        self.reset()
        return self.model.encode_text(token_tensor)

    def get_attention_map(self):
        """
        返回上次 forward 的 text self-attention map（默认最后一层，已 mean over heads）。
        """
        if len(self.attention_maps) == 0:
            return None
        return self.attention_maps[-1]  # shape: [B, T, T]

    def get_tokenizer(self):
        return self.tokenizer

    def get_preprocess(self):
        return self.preprocess
