import torch
import torch.nn as nn


class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, prompt_len=5, class_specific=True,
                 use_init_prompt=True, device='cuda'):
        super().__init__()
        self.prompt_len = prompt_len
        self.class_specific = class_specific
        self.ctx_dim = clip_model.model.token_embedding.embedding_dim
        self.tokenizer = clip_model.get_tokenizer()
        self.token_embedding = clip_model.model.token_embedding  # frozen
        self.device = device
        self.use_init_prompt = use_init_prompt

        # Prompt containers
        self.context_bank = nn.ParameterDict()
        self.token_bank = {}

        print(f"cls_specific: {class_specific}, use_init_prompt: {use_init_prompt}")

        for name in class_names:
            self.add_class_prompt(name)

    def add_class_prompt(self, class_name):
        if class_name in self.context_bank:
            return  # Already exists

        with torch.no_grad():
            text = f"a photo of a {class_name}"
            tokenized = self.tokenizer(text).to(self.device)  # [token_len]
            token_emb = self.token_embedding(tokenized.unsqueeze(0)).squeeze(0)  # [token_len, dim]
            self.token_bank[class_name] = token_emb  # store class token

            # Try to extract context tokens from language prompt
            if self.use_init_prompt and token_emb.shape[0] >= 5 + self.prompt_len:
                # Use tokens 5~(5+prompt_len) from language
                ctx_init = token_emb[5:5 + self.prompt_len].clone()
            else:
                ctx_init = torch.randn(self.prompt_len, self.ctx_dim).to(self.device)

        self.context_bank[class_name] = nn.Parameter(ctx_init)

    def forward(self):
        prompts = []
        for cls in self.context_bank:
            ctx = self.context_bank[cls]  # [prompt_len, dim]
            token = self.token_bank[cls]  # [token_len, dim]

            ctx = ctx.unsqueeze(0)  # [1, prompt_len, dim]

            if token.dim() == 2:
                token = token.unsqueeze(0)  # [1, token_len, dim]
            elif token.dim() == 3:
                pass  # already correct
            elif token.dim() == 4:
                token = token.squeeze(0)  # remove extraneous batch dim
            else:
                raise ValueError(f"Unexpected token shape: {token.shape}")

            prompt = torch.cat([ctx, token], dim=1)  # [1, total_len, dim]
            prompts.append(prompt)

        prompts = torch.cat(prompts, dim=0)  # [n_cls, total_len, dim]
        return prompts

    @property
    def n_cls(self):
        return len(self.context_bank)
