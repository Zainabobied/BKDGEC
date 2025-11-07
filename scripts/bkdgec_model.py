
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BKDGEC model:
- Shared Transformer encoder
- Two Transformer decoders:
    * L2R (teacher)
    * R2L (student)
- Dual knowledge distillation:
    * KL divergence over logits
    * MSE on hidden states (teacher->student projection)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return x


class BKDGEC(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder_l2r = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)  # teacher
        self.decoder_r2l = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)  # student

        self.proj = nn.Linear(d_model, vocab_size)
        self.proj_teacher_to_student = nn.Linear(d_model, d_model)  # hidden-state projection

    @staticmethod
    def _future_mask(sz: int, direction: str, device) -> torch.Tensor:

        if direction == "l2r":
            # standard causal: mask future tokens (upper triangle)
            m = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        else:
            # r2l: mask tokens to the left (lower triangle)
            m = torch.tril(torch.ones(sz, sz, device=device), diagonal=-1)
        m = m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, 0.0)
        return m

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(src_ids)
        x = self.pos(x)
        mem = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return mem

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        direction: str,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          hidden: (B, T, D)
          logits: (B, T, V)
        """
        x = self.embed(tgt_ids)
        x = self.pos(x)
        T = tgt_ids.size(1)
        tgt_mask = self._future_mask(T, direction=direction, device=tgt_ids.device)

        if direction == "l2r":
            hidden = self.decoder_l2r(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            hidden = self.decoder_r2l(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        logits = self.proj(hidden)
        return hidden, logits

    @staticmethod
    def sequence_nll(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> torch.Tensor:

        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)

    def kd_losses(
        self,
        hid_l2r: torch.Tensor,
        logits_l2r: torch.Tensor,
        hid_r2l: torch.Tensor,
        logits_r2l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # KL on distributions
        l2r_logprob = F.log_softmax(logits_l2r, dim=-1)
        r2l_prob = F.softmax(logits_r2l, dim=-1)
        kl = F.kl_div(l2r_logprob, r2l_prob, reduction="batchmean")

        # Hidden-state MSE after projection
        proj = self.proj_teacher_to_student(hid_l2r)
        mse = F.mse_loss(proj, hid_r2l)
        return kl, mse
