"""A two-tower scorer with optional trainable genre Transformer features."""

import math

import torch
from torch import nn
from transformers import BertConfig, BertModel


class TwoTower(nn.Module):
    def __init__(self, users, items, genres, dim=32, content=True):
        super().__init__()
        self.users = nn.Embedding(users, dim)
        self.items = nn.Embedding(items, dim)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("genres", torch.tensor(genres, dtype=torch.long))
        self.encoder = None
        if content:
            self.encoder = BertModel(
                BertConfig(
                    vocab_size=20,
                    hidden_size=dim,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    intermediate_size=dim * 2,
                    max_position_embeddings=20,
                    hidden_dropout_prob=0.0,
                    attention_probs_dropout_prob=0.0,
                    pad_token_id=0,
                    attn_implementation="eager",
                ),
                add_pooling_layer=False,
            )
        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def item_vectors(self, ids):
        result = self.items(ids)
        if self.encoder is not None:
            tokens = self.genres[ids]
            mask = tokens.ne(0)
            hidden = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
            result = result + (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return result

    def forward(self, users, items):
        unique, inverse = items.unique(return_inverse=True)
        vectors = self.item_vectors(unique)[inverse]
        return (
            (self.users(users) * vectors).sum(-1) / math.sqrt(vectors.shape[1])
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + self.bias
        )

    @torch.no_grad()
    def scores(self):
        self.eval()
        ids = torch.arange(self.items.num_embeddings)
        vectors = self.item_vectors(ids)
        return (
            self.users.weight @ vectors.T / math.sqrt(vectors.shape[1])
            + self.user_bias.weight
            + self.item_bias.weight.T
            + self.bias
        ).numpy()
