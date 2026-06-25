# -*- coding: utf-8 -*-
# models.py — BiLSTM+Attention + Transformer + 分类器

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttentionEncoder(nn.Module):
    """BiLSTM + Multi-Head Self-Attention"""
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1), :]
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = self.norm(x)
        return x


class TransformerEncoder(nn.Module):
    """轻量 Transformer. nn.TransformerEncoder, GPU高效."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=512, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.pos_embedding[:, :input_ids.size(1), :]
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x


class SequenceClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, input_ids):
        enc_out = self.encoder(input_ids)
        pooled = enc_out.mean(dim=1)
        return self.pooler(pooled)


def create_model(model_type, vocab_size, **kwargs):
    d_model = kwargs.get('d_model', 128)
    n_layers = kwargs.get('n_layers', 2)
    dropout = kwargs.get('dropout', 0.1)
    max_len = kwargs.get('max_len', 512)

    if model_type == 'bilstm':
        encoder = BiLSTMAttentionEncoder(vocab_size, d_model=d_model,
                                          n_layers=n_layers, max_len=max_len, dropout=dropout)
    elif model_type == 'transformer':
        encoder = TransformerEncoder(vocab_size, d_model=d_model, n_heads=4,
                                      n_layers=n_layers, max_len=max_len, dropout=dropout)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    return SequenceClassifier(encoder, d_model, num_classes=2, dropout=dropout)