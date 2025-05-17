import torch
import torch.nn as nn

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, prompt_len=5, prefix_len=5, use_init_prompt=True, device='cuda', max_len=77):
        super().__init__()
        self.prompt_len = prompt_len
        self.prefix_len = prefix_len
        self.ctx_dim = clip_model.model.token_embedding.embedding_dim
        self.tokenizer = clip_model.get_tokenizer()
        self.token_embedding = clip_model.model.token_embedding  # frozen
        self.device = device
        self.use_init_prompt = use_init_prompt
        self.max_len = max_len

        # Shared prefix prompt
        self.prefix = nn.Parameter(torch.randn(prefix_len, self.ctx_dim))

        # Class-specific prompt bank
        self.context_bank = nn.ParameterDict()
        self.token_bank = {}

        print(f"Using prefix_len={prefix_len}, class-specific context_len={prompt_len}, total limit={max_len}")

        for name in class_names:
            self.add_class_prompt(name)

    def add_class_prompt(self, class_name):
        if class_name in self.context_bank:
            return

        with torch.no_grad():
            text = f"a photo of a {class_name}"
            tokenized = self.tokenizer(text).to(self.device)
            token_emb = self.token_embedding(tokenized.unsqueeze(0)).squeeze(0)

            max_cls_token_len = self.max_len - (self.prefix_len + self.prompt_len)
            if token_emb.shape[0] > max_cls_token_len:
                # print(f"⚠️ Class name '{class_name}' tokens too long ({token_emb.shape[0]}), truncating to {max_cls_token_len}")
                token_emb = token_emb[:max_cls_token_len]

            self.token_bank[class_name] = token_emb.to(self.device)

            if self.use_init_prompt and token_emb.shape[0] >= 5 + self.prompt_len:
                ctx_init = token_emb[5:5 + self.prompt_len].clone()
                ctx_init = (ctx_init - ctx_init.mean(dim=0)) / (ctx_init.std(dim=0) + 1e-6)
            else:
                ctx_init = torch.randn(self.prompt_len, self.ctx_dim).to(self.device)

        self.context_bank[class_name] = nn.Parameter(ctx_init)

    def forward(self):
        prompts = []
        for cls in self.context_bank:
            prefix = self.prefix.unsqueeze(0)  # [1, prefix_len, D]
            ctx = self.context_bank[cls].unsqueeze(0)  # [1, prompt_len, D]
            token = self.token_bank[cls].unsqueeze(0) if self.token_bank[cls].dim() == 2 else self.token_bank[cls]

            total_len = prefix.size(1) + ctx.size(1) + token.size(1)
            if total_len > self.max_len:
                excess = total_len - self.max_len
                # print(f"⚠️ Prompt for '{cls}' too long ({total_len}), trimming class token by {excess} tokens")
                token = token[:, :-excess]

            prompt = torch.cat([prefix, ctx, token], dim=1)  # [1, T, D]
            prompts.append(prompt)

        return torch.cat(prompts, dim=0)  # [C, T, D]

    @property
    def n_cls(self):
        return len(self.context_bank)

    @property
    def context_len(self):
        return self.prefix_len + self.prompt_len
