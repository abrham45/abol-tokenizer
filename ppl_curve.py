#!/usr/bin/env python3
"""
Perplexity vs training steps (0 → 250,000 tokens) for all tokenizers.
Outputs JSON: { algorithm: [ {step, ppl}, ... ], ... }

1 step = 1 training token (word).
At each checkpoint, train the tokenizer on the first N tokens from the
corpus, then evaluate unigram PPL on a fixed held-out test set.

GPT-2 and cl100k are pretrained (no Amharic training), so their PPL
is measured at every checkpoint without retraining — flat reference lines.
"""

import math
import json
import tempfile
import os
import random
from collections import Counter
from typing import List, Dict, Tuple

import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from amharic_tokenizer.tokenizer import Tokenizer
from amharic_tokenizer.decomposed_tokenizer import DecomposedTokenizer
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_PATH = "./ahun_corpus.txt"
CHECKPOINTS = [
    1_000, 5_000, 10_000, 25_000, 50_000, 75_000,
    100_000, 125_000, 150_000, 175_000, 200_000, 225_000, 250_000
]
TEST_RATIO  = 0.15
SEED        = 42

# ─────────────────────────────────────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sentences(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sentences_to_tokens(sents: List[str]) -> List[str]:
    """Split sentences to whitespace tokens (words)."""
    tokens = []
    for s in sents:
        tokens.extend(s.split())
    return tokens


def tokens_to_sentences(tokens: List[str], orig_sents: List[str]) -> List[str]:
    """
    Reconstruct minimal sentence list whose word count == len(tokens).
    We simply take sentences greedily until we have enough words.
    """
    result = []
    count = 0
    for sent in orig_sents:
        words = sent.split()
        if count + len(words) > len(tokens):
            break
        result.append(sent)
        count += len(words)
        if count >= len(tokens):
            break
    return result


def split_train_test(sents: List[str], test_ratio=0.15, seed=42):
    random.seed(seed)
    shuffled = sents[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def compute_ppl(train_toks: List[str], test_toks: List[str],
                smoothing: float = 1e-10) -> float:
    counts = Counter(train_toks)
    total = sum(counts.values())
    V = max(len(counts), 1)
    log_sum = 0.0
    for tok in test_toks:
        p = (counts.get(tok, 0) + smoothing) / (total + smoothing * V)
        log_sum += math.log(p)
    if not test_toks:
        return float("inf")
    return round(math.exp(-log_sum / len(test_toks)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-algorithm tokenize helpers
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_abol(model, sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        try:
            _, spans = model.encode_with_spans(s)
            out.extend(sp["token"] for sp in spans)
        except Exception:
            pass
    return out


def train_abol_gms(sents: List[str]) -> Tokenizer:
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write("\n".join(sents))
        tmp = f.name
    m = Tokenizer()
    m.train_from_corpus(tmp)
    os.unlink(tmp)
    return m


def train_abol_cv(sents: List[str]) -> DecomposedTokenizer:
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write("\n".join(sents))
        tmp = f.name
    m = DecomposedTokenizer()
    m.train_from_corpus(tmp)
    os.unlink(tmp)
    return m


def train_abol_hybrid(sents: List[str]) -> HybridTokenizer:
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write("\n".join(sents))
        tmp = f.name
    m = HybridTokenizer()
    m.train_from_corpus(tmp)
    os.unlink(tmp)
    return m


def train_bpe(sents: List[str], vocab_size: int = 13_000):
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write("\n".join(sents))
        tmp = f.name
    tok = HFTokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]"],
        show_progress=False,
    )
    tok.train([tmp], trainer)
    os.unlink(tmp)
    return tok


def tokenize_bpe(model, sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        enc = model.encode(s)
        out.extend(enc.tokens)
    return out


def train_sp(sents: List[str], model_type: str, vocab_size: int = 8_000):
    with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        f.write("\n".join(sents))
        tmp = f.name
    prefix = f"/tmp/spm_curve_{model_type}"
    base = dict(input=tmp, model_prefix=prefix, model_type=model_type,
                character_coverage=0.9999,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                pad_piece="<pad>", unk_piece="<unk>")
    for vs in [vocab_size, 8_000, 6_000, 4_000, 2_000, 1_000]:
        try:
            spm.SentencePieceTrainer.train(vocab_size=vs, **base)
            break
        except RuntimeError as e:
            if "Vocabulary size too high" in str(e):
                continue
            raise
    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    os.unlink(tmp)
    return sp


def tokenize_sp(sp, sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        out.extend(sp.encode(s, out_type=str))
    return out


def tokenize_tiktoken(enc, sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        ids = enc.encode(s, disallowed_special=())
        out.extend(enc.decode([i]) for i in ids)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tiktoken

    print("Loading corpus...")
    all_sents = load_sentences(CORPUS_PATH)
    train_sents, test_sents = split_train_test(all_sents, TEST_RATIO, SEED)
    all_train_tokens = sentences_to_tokens(train_sents)

    print(f"  Train tokens : {len(all_train_tokens):,}")
    print(f"  Test sents   : {len(test_sents):,}")

    # Pre-load fixed tiktoken encoders (no Amharic training — flat lines)
    print("Loading tiktoken encoders (GPT-2, cl100k)...")
    gpt2_enc    = tiktoken.get_encoding("gpt2")
    cl100k_enc  = tiktoken.get_encoding("cl100k_base")
    gpt2_test   = tokenize_tiktoken(gpt2_enc, test_sents)
    cl100k_test = tokenize_tiktoken(cl100k_enc, test_sents)

    results: Dict[str, List[Dict]] = {
        "Abol-GMS":              [],
        "Abol-CV":               [],
        "Abol-Hybrid":           [],
        "BPE (Amharic)":         [],
        "SentencePiece (unigram)": [],
        "SentencePiece (bpe)":   [],
        "GPT-2":                 [],
        "cl100k":                [],
    }

    for step in CHECKPOINTS:
        # Slice training data to `step` tokens
        subset_tokens = all_train_tokens[:step]
        subset_sents  = tokens_to_sentences(subset_tokens, train_sents)

        if not subset_sents:
            for alg in results:
                results[alg].append({"step": step, "ppl": None})
            continue

        print(f"\n── Step {step:,} ({len(subset_sents):,} sentences) ──")

        # ── Abol-GMS ──
        print("  [Abol-GMS]", end=" ", flush=True)
        try:
            m = train_abol_gms(subset_sents)
            train_t = tokenize_abol(m, subset_sents)
            test_t  = tokenize_abol(m, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["Abol-GMS"].append({"step": step, "ppl": ppl})

        # ── Abol-CV ──
        print("  [Abol-CV]", end=" ", flush=True)
        try:
            m = train_abol_cv(subset_sents)
            train_t = tokenize_abol(m, subset_sents)
            test_t  = tokenize_abol(m, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["Abol-CV"].append({"step": step, "ppl": ppl})

        # ── Abol-Hybrid ──
        print("  [Abol-Hybrid]", end=" ", flush=True)
        try:
            m = train_abol_hybrid(subset_sents)
            train_t = tokenize_abol(m, subset_sents)
            test_t  = tokenize_abol(m, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["Abol-Hybrid"].append({"step": step, "ppl": ppl})

        # ── BPE (Amharic) ──
        print("  [BPE (Amharic)]", end=" ", flush=True)
        try:
            m = train_bpe(subset_sents, vocab_size=min(13_000, len(subset_sents) * 3))
            train_t = tokenize_bpe(m, subset_sents)
            test_t  = tokenize_bpe(m, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["BPE (Amharic)"].append({"step": step, "ppl": ppl})

        # ── SentencePiece Unigram ──
        print("  [SP-Unigram]", end=" ", flush=True)
        try:
            sp = train_sp(subset_sents, "unigram", vocab_size=8_000)
            train_t = tokenize_sp(sp, subset_sents)
            test_t  = tokenize_sp(sp, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["SentencePiece (unigram)"].append({"step": step, "ppl": ppl})

        # ── SentencePiece BPE ──
        print("  [SP-BPE]", end=" ", flush=True)
        try:
            sp = train_sp(subset_sents, "bpe", vocab_size=min(13_000, len(subset_sents) * 3))
            train_t = tokenize_sp(sp, subset_sents)
            test_t  = tokenize_sp(sp, test_sents)
            ppl = compute_ppl(train_t, test_t)
        except Exception as e:
            ppl = None
        print(f"PPL={ppl}")
        results["SentencePiece (bpe)"].append({"step": step, "ppl": ppl})

        # ── GPT-2 (fixed) — retokenize test with a unigram built on CURRENT train tokens ──
        # GPT-2 doesn't retrain, but we measure PPL of its fixed tokenization
        # using the train-set frequency distribution at this step.
        print("  [GPT-2]", end=" ", flush=True)
        try:
            step_gpt2_train = tokenize_tiktoken(gpt2_enc, subset_sents)
            ppl = compute_ppl(step_gpt2_train, gpt2_test)
        except Exception:
            ppl = None
        print(f"PPL={ppl}")
        results["GPT-2"].append({"step": step, "ppl": ppl})

        # ── cl100k (fixed) ──
        print("  [cl100k]", end=" ", flush=True)
        try:
            step_cl_train = tokenize_tiktoken(cl100k_enc, subset_sents)
            ppl = compute_ppl(step_cl_train, cl100k_test)
        except Exception:
            ppl = None
        print(f"PPL={ppl}")
        results["cl100k"].append({"step": step, "ppl": ppl})

    # ── Save ──
    out_path = "ppl_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"checkpoints": CHECKPOINTS, "curves": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n✓ Saved → {out_path}")

    # ── Print summary table ──
    print("\n" + "=" * 75)
    print(f"{'Algorithm':<28}", end="")
    for s in CHECKPOINTS:
        print(f"{s//1000:>6}K", end="")
    print()
    print("=" * 75)
    for alg, curve in results.items():
        print(f"{alg:<28}", end="")
        for pt in curve:
            v = pt["ppl"]
            if v is None:
                print(f"{'N/A':>7}", end="")
            elif v > 99_999:
                print(f"{'>>99K':>7}", end="")
            else:
                print(f"{v:>7.0f}", end="")
        print()
    print("=" * 75)
