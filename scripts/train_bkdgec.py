#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for BKDGEC on EDSE-generated CSV.

Usage:
    python scripts/train_bkdgec.py \
        --input data/train.csv \
        --config configs/configs.yml

Input:
  - CSV with columns: src (corrupted), trg (clean)
Config (YAML, e.g. configs/configs.yml):
  paths:
    spm_model: "models/spm_ar1k.model"
    save_dir:  "checkpoints/bkdgec"
Output:
  - Single trained model checkpoint:
      <save_dir>/bkdgec.pt
"""

import argparse
import os
import random
from typing import List, Tuple

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from bkdgec_model import BKDGEC


PAD = 0
BOS = 1
EOS = 2
UNK = 3


# ----------------------- Dataset -----------------------
class CsvGEDataset(Dataset):
    """Simple GEC dataset over a CSV with columns src,trg."""

    def __init__(self, csv_path: str, sp: spm.SentencePieceProcessor, max_len: int = 400):
        self.df = pd.read_csv(csv_path)
        if not {"src", "trg"}.issubset(self.df.columns):
            raise ValueError("CSV must contain 'src' and 'trg' columns.")
        self.sp = sp
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, txt: str) -> List[int]:
        ids = [BOS] + self.sp.encode(txt, out_type=int)[: self.max_len - 2] + [EOS]
        return ids

    @staticmethod
    def _pad(ids: List[int], L: int) -> List[int]:
        return ids + [PAD] * (L - len(ids))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_txt = str(self.df.iloc[idx]["src"])
        trg_txt = str(self.df.iloc[idx]["trg"])

        src_ids = self._encode(src_txt)
        trg_ids = self._encode(trg_txt)

        # R2L target: reverse inner tokens (between BOS and EOS)
        trg_r2l = trg_ids[:]
        core = trg_r2l[1:-1]
        trg_r2l[1:-1] = list(reversed(core))

        L = max(len(src_ids), len(trg_ids))
        src_ids = self._pad(src_ids, L)
        trg_ids = self._pad(trg_ids, L)
        trg_r2l = self._pad(trg_r2l, L)

        return torch.tensor(src_ids), torch.tensor(trg_ids), torch.tensor(trg_r2l)


def collate(batch):
    s, t, tr = zip(*batch)
    return torch.stack(s), torch.stack(t), torch.stack(tr)


# ----------------------- Utilities -----------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    """
    Load YAML config and merge with sane defaults.
    Expected keys:
      - paths.spm_model
      - paths.save_dir
      - model.*
      - train.*
      - device
    """
    base = {
        "paths": {
            "spm_model": "models/spm_ar1k.model",
            "save_dir": "checkpoints/bkdgec",
        },
        "model": {
            "d_model": 256,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 1024,
            "dropout": 0.15,
        },
        "train": {
            "batch_size": 64,
            "lr": 3e-3,
            "epochs": 27,
            "clip": 1.0,
            "seed": 42,
            "max_len": 400,
            "lambda_kd": 1.0,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        # shallow-merge
        for k, v in user.items():
            if isinstance(v, dict) and k in base:
                base[k].update(v)
            else:
                base[k] = v

    return base


# ----------------------- Training -----------------------
def train_one_epoch(
    model: BKDGEC,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> float:
    model.train()
    total = 0.0
    clip = cfg["train"]["clip"]
    lambda_kd = cfg["train"]["lambda_kd"]

    pbar = tqdm(dl, desc="Train", leave=False)
    for src, trg_l2r, trg_r2l in pbar:
        src = src.to(device)
        trg_l2r = trg_l2r.to(device)
        trg_r2l = trg_r2l.to(device)

        # Encode
        mem = model.encode(src)

        # Teacher forcing
        in_l2r, tgt_l2r = trg_l2r[:, :-1], trg_l2r[:, 1:]
        in_r2l, tgt_r2l = trg_r2l[:, :-1], trg_r2l[:, 1:]

        # Decode
        hid_l2r, logits_l2r = model.decode(in_l2r, mem, "l2r")
        hid_r2l, logits_r2l = model.decode(in_r2l, mem, "r2l")

        # NLL
        nll_l2r = model.sequence_nll(logits_l2r, tgt_l2r, ignore_index=PAD)
        nll_r2l = model.sequence_nll(logits_r2l, tgt_r2l, ignore_index=PAD)

        # KD (teacher detached)
        kl, mse = model.kd_losses(hid_l2r.detach(), logits_l2r.detach(), hid_r2l, logits_r2l)
        loss = nll_l2r + nll_r2l + lambda_kd * (kl + mse)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total / len(dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Training CSV with columns src,trg")
    ap.add_argument("--config", default="configs/configs.yml", help="YAML config file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    paths = cfg["paths"]
    save_dir = paths.get("save_dir", "checkpoints/bkdgec")
    os.makedirs(save_dir, exist_ok=True)

    # SentencePiece tokenizer
    sp = spm.SentencePieceProcessor(model_file=paths["spm_model"])
    vocab_size = sp.get_piece_size()

    # Dataset / DataLoader
    train_ds = CsvGEDataset(args.input, sp, max_len=cfg["train"]["max_len"])
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )

    # Model
    model = BKDGEC(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        num_decoder_layers=cfg["model"]["num_decoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        pad_id=PAD,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # Training loop (train-only; no dev, single final checkpoint)
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        avg = train_one_epoch(model, train_dl, opt, device, cfg)
        print(f"Epoch {epoch:02d} - train loss {avg:.4f}")

    # Save single final model
    out_path = os.path.join(save_dir, "bkdgec.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Training finished. Saved model to: {out_path}")


if __name__ == "__main__":
    main()
