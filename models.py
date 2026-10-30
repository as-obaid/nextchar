"""The three next-character models.

Each one maps a (B, T) tensor of character ids to (B, T, V) logits, where the
logits at position t are a prediction for character t + 1. That shared interface
lets all three use the same training loop, loss, and sampler.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    """A V x V table: row i holds the logits for the character that follows character i.
    The prediction at position t depends on idx[t] alone."""

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.table.weight)  # start from a uniform distribution

    def forward(self, idx):
        return self.table(idx)


class LSTMLM(nn.Module):
    """Embedding -> stacked LSTM -> linear head. The hidden and cell states carry a
    summary of every earlier character in the window forward, one step at a time."""

    def __init__(self, vocab_size, n_embd, hidden_size, n_layer, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden_size, n_layer, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx):
        # No initial state is passed, so it starts at zero: every window is independent.
        out, _ = self.lstm(self.embed(idx))
        return self.head(self.drop(out))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention in which position t can only attend to positions <= t."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size): each head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # hide future positions
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    """Pre-norm transformer block: attention mixes information across positions,
    then the feedforward layer transforms each position on its own."""

    def __init__(self, n_embd, n_head, block_size, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TransformerLM(nn.Module):
    """Decoder-only transformer: token + learned position embeddings, a stack of
    blocks, a final LayerNorm, and a linear output layer."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, ff_dim, block_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, ff_dim, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        return self.head(self.ln_f(self.blocks(x)))


MODELS = {"bigram": BigramLM, "lstm": LSTMLM, "transformer": TransformerLM}


@torch.no_grad()
def generate(model, idx, n_new, block_size, temperature=1.0):
    """Sample n_new characters. Every model sees at most the last block_size
    characters, the same context budget it was trained with."""
    model.eval()
    for _ in range(n_new):
        logits = model(idx[:, -block_size:])[:, -1, :] / temperature
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
