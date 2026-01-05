#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import random
import yaml
from typing import List
from tqdm import tqdm
from pyarabic import araby as A

# ------------------------------------------------------------
# Tokenization helpers (PyArabic + light heuristics)
# ------------------------------------------------------------
def tokenize(s: str) -> List[str]:
    s = A.strip_tashkeel(A.strip_tatweel(s))
    return s.split()

def detokenize(toks: List[str]) -> str:
    return " ".join(toks)

def is_ar_word(w: str) -> bool:
    return any(A.is_arabicrange(ch) for ch in w)

def looks_like_verb(w: str) -> bool:
    return w.startswith(("ي", "ت", "أ", "ن")) and len(w) >= 3 and is_ar_word(w)

def looks_like_noun(w: str) -> bool:
    return (
        (w.startswith("ال") or w.endswith(("ة", "ات", "ون", "ين", "ان")))
        or (len(w) >= 3 and is_ar_word(w) and not looks_like_verb(w))
    )

def remove_definite_article(w: str) -> str:
    return w[2:] if w.startswith("ال") and len(w) > 2 else w

def to_present_tense(w: str) -> str:
    # Crude tense unification: prepend "ي" if it looks nominal and is Arabic
    if not looks_like_verb(w) and len(w) >= 3 and is_ar_word(w):
        return "ي" + w
    return w

def drop_affix(w: str) -> str:
    if len(w) < 4:
        return w
    if random.random() < 0.5:
        for suf in ("كما","هما","كم","هن","هم","ها","نا","ان","ون","ين","ة","ات","ي","ك","ه","ا"):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return w[:-len(suf)]
        return w[:-1]
    else:
        for pre in ("وال","بال","كال","فال","لل","ال","و","ف","ب","ك","ل"):
            if w.startswith(pre) and len(w) > len(pre) + 1:
                return w[len(pre):]
        return w[1:]

# ------------------------------------------------------------
# EDSE Generator (two balanced pipelines)
# ------------------------------------------------------------
class EDSEGenerator:

    def __init__(self, cfg_path: str):
        self.cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
        random.seed(self.cfg.get("seed", 42))

        vocab_path = self.cfg.get("vocab_path", "data/vocab.txt")
        try:
            self.vocab = [l.strip() for l in open(vocab_path, encoding="utf-8") if l.strip()]
        except FileNotFoundError:
            self.vocab = ["شيء", "مكان", "زمن", "حدث", "موضوع", "كتاب", "مدينة", "شخص", "فكرة", "طريقة"]

        self.punct = self.cfg.get("punctuation", ["،","؟",".","!"])
        self.min_len_char = self.cfg.get("min_len_for_char_ops", 2)
        self.alpha = float(self.cfg.get("alpha", 0.10))

    # ---------------- Pipeline 1: misspelling / punctuation / structure ----------------
    def op_char_edit(self, toks: List[str]) -> List[str]:
        cand = [i for i, w in enumerate(toks) if is_ar_word(w) and len(w) >= self.min_len_char]
        if not cand:
            return toks
        i = random.choice(cand)
        w = toks[i]
        if len(w) < 2:
            return toks
        j = random.randrange(len(w))
        choice = random.random()
        if choice < 1/3:  # deletion
            w2 = w[:j] + w[j+1:]
        elif choice < 2/3:  # insertion
            ch = random.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
            w2 = w[:j] + ch + w[j:]
        else:  # substitution
            ch = random.choice("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
            w2 = w[:j] + ch + (w[j+1:] if j + 1 <= len(w) else "")
        toks[i] = w2
        return toks

    def op_punct_edit(self, toks: List[str]) -> List[str]:
        # Insert or remove punctuation
        if random.random() < 0.5 and len(toks) > 1:
            pos = random.randrange(len(toks) + 1)
            toks = toks[:pos] + [random.choice(self.punct)] + toks[pos:]
        else:
            idx = [i for i, w in enumerate(toks) if w in self.punct]
            if idx:
                toks.pop(random.choice(idx))
        return toks

    def op_structure(self, toks: List[str]) -> List[str]:
        # Light VSO/SVO perturbation using heuristics
        verbs = [i for i, w in enumerate(toks) if looks_like_verb(w)]
        nouns = [i for i, w in enumerate(toks) if looks_like_noun(w)]
        if not verbs or len(nouns) < 1:
            # Fallback: swap first two Arabic words
            pos = [i for i, w in enumerate(toks) if is_ar_word(w)]
            if len(pos) >= 2:
                i, j = pos[0], pos[1]
                toks[i], toks[j] = toks[j], toks[i]
            return toks

        v = verbs[0]
        s = nouns[0]
        if len(nouns) >= 2:
            o = nouns[1]
            order = random.choice(["SVO", "VSO", "OVS"])
            triple_idx = sorted([s, v, o])
            mapping = {"S": toks[s], "V": toks[v], "O": toks[o]}
            new_seq = [mapping[tag] for tag in order]
            toks[triple_idx[0]] = new_seq[0]
            toks[triple_idx[1]] = new_seq[1]
            toks[triple_idx[2]] = new_seq[2]
        else:
            # swap subject ↔ verb
            toks[v], toks[s] = toks[s], toks[v]
        return toks

    # --------------- Pipeline 2: syntax (5) + semantics (2) ----------------
    def op_delete_determiner(self, toks: List[str]) -> List[str]:
        ixs = [i for i, w in enumerate(toks) if w.startswith("ال") and len(w) > 2]
        if ixs:
            i = random.choice(ixs)
            toks[i] = remove_definite_article(toks[i])
        return toks

    def op_verb_subject_disagreement(self, toks: List[str]) -> List[str]:
        # Replace a noun-like token to induce agreement issues
        ixs = [i for i, w in enumerate(toks) if looks_like_noun(w)]
        if ixs and self.vocab:
            i = random.choice(ixs)
            toks[i] = random.choice(self.vocab)
        return toks

    def op_tense_unification(self, toks: List[str]) -> List[str]:
        # Convert one token to a present-like form
        ixs = [i for i, w in enumerate(toks) if is_ar_word(w)]
        if ixs:
            i = random.choice(ixs)
            toks[i] = to_present_tense(toks[i])
        return toks

    def op_affix_delete(self, toks: List[str]) -> List[str]:
        ixs = [i for i, w in enumerate(toks) if is_ar_word(w) and len(w) >= 4]
        if ixs:
            i = random.choice(ixs)
            toks[i] = drop_affix(toks[i])
        return toks

    def op_semantic_replace_one(self, toks: List[str]) -> List[str]:
        ixs = [i for i, w in enumerate(toks) if is_ar_word(w)]
        if ixs and self.vocab:
            i = random.choice(ixs)
            toks[i] = random.choice(self.vocab)
        return toks

    def op_semantic_replace_two(self, toks: List[str]) -> List[str]:
        ixs = [i for i, w in enumerate(toks) if is_ar_word(w)]
        random.shuffle(ixs)
        for i in ixs[:2]:
            if self.vocab:
                toks[i] = random.choice(self.vocab)
        return toks

    # --------------- One EDSE pass per sentence ----------------
    def generate_once(self, sent: str) -> str:
        toks = tokenize(sent)
        if not toks:
            return sent

        steps = max(1, math.ceil(len(toks) * self.alpha))

        pipeline1 = [self.op_char_edit, self.op_punct_edit, self.op_structure]
        pipeline2 = [
            self.op_delete_determiner,
            self.op_verb_subject_disagreement,
            self.op_tense_unification,
            self.op_affix_delete,
            self.op_semantic_replace_one,
            self.op_semantic_replace_two,
        ]

        # Equal probability of selecting a pipeline per sentence
        pipeline = random.choice([pipeline1, pipeline2])

        for _ in range(steps):
            op = random.choice(pipeline)
            toks = op(toks)

        return detokenize(toks)

# ------------------------------------------------------------
# CLI: read mono.txt, write a single CSV with columns: src,trg
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/source_text.txt", help="Clean Arabic sentences, one per line")
    ap.add_argument("--output", default="data/train.csv", help="Output CSV file path")
    ap.add_argument("--config", default="configs/configs.yml", help="EDSE configuration YAML at repo root")
    args = ap.parse_args()

    gen = EDSEGenerator(args.config)

    with open(args.input, encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["src", "trg"])  # header

        for line in tqdm(f_in, desc="EDSE Generating"):
            s = line.strip()
            if not s:
                continue
            corrupted = gen.generate_once(s)
            writer.writerow([corrupted, s])

if __name__ == "__main__":
    main()

