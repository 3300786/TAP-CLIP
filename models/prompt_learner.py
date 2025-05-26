import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, prompt_len=5, prefix_len=5,
                 use_init_prompt=True, device='cuda', max_len=77):
        super().__init__()
        self.prompt_len = prompt_len
        self.prefix_len = prefix_len
        self.ctx_dim = clip_model.model.token_embedding.embedding_dim
        self.tokenizer = clip_model.get_tokenizer()
        self.token_embedding = clip_model.model.token_embedding  # frozen
        self.device = device
        self.use_init_prompt = use_init_prompt
        self.max_len = max_len

        # 多样化模板增强语义
        self.templates = [
            "a photo of a {}",
            "a picture of the {}",
            "an image of a {}",
            "this is a {}",
            "this is a photo of a {}",
        ]

        self.prefix = nn.Parameter(torch.randn(prefix_len, self.ctx_dim))
        self.context_bank = nn.ParameterDict()
        self.token_id_bank = {}

        print(f"Using prefix_len={prefix_len}, prompt_len={prompt_len}, total limit={max_len}")
        for name in class_names:
            self.add_class_prompt(name)

    def add_class_prompt(self, class_name):
        if class_name in self.context_bank:
            return

        with torch.no_grad():
            template = random.choice(self.templates)
            text = template.format(class_name)

            tokenized = self.tokenizer(text).to(self.device)  # [T]
            valid_idx = tokenized != 0
            tokenized = tokenized[valid_idx]

            token_emb = self.token_embedding(tokenized.unsqueeze(0)).squeeze(0)  # [T, D]

            # 限制 token embedding 长度并补齐到 max_token_len
            max_token_len = self.max_len - self.prefix_len - self.prompt_len
            if token_emb.shape[0] > max_token_len:
                token_emb = token_emb[:max_token_len]
                tokenized = tokenized[:max_token_len]
            else:
                pad_len = max_token_len - token_emb.shape[0]
                token_emb = F.pad(token_emb, (0, 0, 0, pad_len))        # [T+pad_len, D]
                tokenized = F.pad(tokenized, (0, pad_len), value=0)     # [T+pad_len]

            # 初始化 context embedding
            if self.use_init_prompt and token_emb.shape[0] >= self.prompt_len:
                center = token_emb.shape[0] // 2
                start = max(center - self.prompt_len // 2, 0)
                end = start + self.prompt_len
                if end > token_emb.shape[0]:
                    end = token_emb.shape[0]
                    start = end - self.prompt_len
                ctx_init = token_emb[start:end].clone()
                ctx_init = (ctx_init - ctx_init.mean(dim=0)) / (ctx_init.std(dim=0) + 1e-6)
            else:
                ctx_init = torch.randn(self.prompt_len, self.ctx_dim).to(self.device)

        self.context_bank[class_name] = nn.Parameter(ctx_init)
        self.token_id_bank[class_name] = tokenized

    def get_prompt_parts(self):
        return self.prefix, self.context_bank, self.token_id_bank

    @property
    def n_cls(self):
        return len(self.context_bank)

    @property
    def context_len(self):
        return self.prefix_len + self.prompt_len
