
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for BKDGEC on EDSE-generated CSV.

Input:
  - CSV with columns: src (corrupted), trg (clean)

Tokenization:
  - SentencePiece model (BPE) provided via --spm_model

Config:
  - YAML at configs/model.yml (optional). CLI args override configs.

Outputs:
  - checkpoints in --save_dir (best.pt + epoch###.pt)
"""

import argparse
import math
import os
import yaml
import random
import pandas as pd
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm

from bkdgec_model import BKDGEC


PAD = 0
BOS = 1
EOS = 2
UNK = 3


# ----------------------- Dataset -----------------------
class CsvGEDataset(Dataset):
    def __init__(self, csv_path: str, sp: spm.SentencePieceProcessor, max_len: int = 400):
        self.df = pd.read_csv(csv_path)
        if not {"src", "trg"}.issubset(set(self.df.columns)):
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

        # R2L target: reverse the inner tokens (BOS ... EOS)
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


def load_config(path: Optional[str]) -> dict:
    base = {
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
            "anneal_warm_steps": 1000,  # optional use if you add annealing
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
):
    model.train()
    total = 0.0
    pbar = tqdm(dl, desc="Train", leave=False)
    clip = cfg["train"]["clip"]
    lambda_kd = cfg["train"]["lambda_kd"]

    for src, trg_l2r, trg_r2l in pbar:
        src = src.to(device)
        trg_l2r = trg_l2r.to(device)
        trg_r2l = trg_r2l.to(device)

        # Encoder
        mem = model.encode(src)

        # Teacher forcing inputs/targets
        in_l2r, tgt_l2r = trg_l2r[:, :-1], trg_l2r[:, 1:]
        in_r2l, tgt_r2l = trg_r2l[:, :-1], trg_r2l[:, 1:]

        # Decode
        hid_l2r, logits_l2r = model.decode(in_l2r, mem, "l2r")
        hid_r2l, logits_r2l = model.decode(in_r2l, mem, "r2l")

        # NLL losses
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


@torch.no_grad()
def evaluate(model: BKDGEC, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for src, trg_l2r, trg_r2l in dl:
        src = src.to(device)
        trg_l2r = trg_l2r.to(device)
        trg_r2l = trg_r2l.to(device)

        mem = model.encode(src)

        in_r2l, tgt_r2l = trg_r2l[:, :-1], trg_r2l[:, 1:]
        _, logits_r2l = model.decode(in_r2l, mem, "r2l")

        total += model.sequence_nll(logits_r2l, tgt_r2l, ignore_index=PAD).item()
    return total / len(dl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv", help="CSV with columns src,trg")
    ap.add_argument("--dev_csv", default=None, help="optional dev CSV with src,trg")
    ap.add_argument("--spm_model", required=True, help="SentencePiece model file (e.g., spm_ar1k.model)")
    ap.add_argument("--save_dir", default="runs/bkdgec", help="Where to save checkpoints")
    ap.add_argument("--config", default="configs/model.yml", help="YAML config file (optional)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    vocab_size = sp.get_piece_size()

    # Datasets
    train_ds = CsvGEDataset(args.train_csv, sp, max_len=cfg["train"]["max_len"])
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    dev_dl = None
    if args.dev_csv and os.path.exists(args.dev_csv):
        dev_ds = CsvGEDataset(args.dev_csv, sp, max_len=cfg["train"]["max_len"])
        dev_dl = DataLoader(dev_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

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

    best = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        avg = train_one_epoch(model, train_dl, opt, device, cfg)

        if dev_dl is not None:
            dev_loss = evaluate(model, dev_dl, device)
        else:
            dev_loss = avg  # fall back to train loss as proxy

        # save per-epoch
        path_epoch = os.path.join(args.save_dir, f"epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), path_epoch)

        if dev_loss < best:
            best = dev_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))

        print(f"Epoch {epoch:02d} - train {avg:.4f} - dev {dev_loss:.4f}")

    print("Done. Best dev loss:", best)


if __name__ == "__main__":
    main()
