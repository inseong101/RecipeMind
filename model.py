import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.mha(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + attn_output)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights


class HerbMindModel(nn.Module):
    def __init__(self, num_herbs: int, embed_dim: int = 128, num_heads: int = 4, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_herbs = num_herbs
        self.embed = nn.Embedding(num_herbs + 1, embed_dim, padding_idx=num_herbs)
        self.blocks = nn.ModuleList([SetAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_blocks)])
        self.proj = nn.Linear(embed_dim, num_herbs)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        # mask: float tensor 1 for valid, 0 for pad
        padding_mask = (mask == 0)
        x = self.embed(input_ids)
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, padding_mask=padding_mask)
            attn_maps.append(attn)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        logits = self.proj(pooled)
        if return_attn:
            return logits, attn_maps
        return logits

    def recommend(self, input_ids: torch.Tensor, mask: torch.Tensor, topk: int = 3):
        logits = self.forward(input_ids, mask)
        probs = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=topk, dim=-1)
        return indices, values
