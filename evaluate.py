#!/usr/bin/env python3
"""
Evaluation script for Abol Tokenizer algorithms vs. standard tokenizers.

Algorithms compared:
  - Abol-GMS           (Amharic-specific, Greedy Meaningful Subword)
  - Abol-Decomposed    (Amharic-specific, Fidel CV + morphological)
  - BPE                (Byte Pair Encoding trained on Amharic corpus)
  - GPT-2              (OpenAI BPE, English-optimised)
  - cl100k             (OpenAI tiktoken cl100k_base, used in GPT-4)

Metrics:
  PPL          — Unigram perplexity on held-out corpus       (lower  = better)
  Steps Thr.   — Encoding throughput in tokens/second        (higher = better)
  BLEU         — Reconstruction BLEU original → decode       (higher = better)
  chrF         — Reconstruction character F-score            (higher = better)
  Vocab        — Vocabulary size
  Tok/w        — Average tokens per word                     (lower  = more compact)
  UNK (%)      — Out-of-vocabulary token rate                (lower  = better)
  Fidel (%)    — % of tokens that are valid Ethiopic syllables (higher = better)
"""

import time
import math
import json
import os
import tempfile
import random
from collections import Counter
from typing import List, Tuple, Dict, Any

import sacrebleu
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from amharic_tokenizer.tokenizer import Tokenizer
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Ethiopic character set
# ──────────────────────────────────────────────────────────────────────────────

ETHIOPIC_CHARS = set(chr(c) for c in range(0x1200, 0x1380))


def is_ethiopic_token(tok: str) -> bool:
    return bool(tok) and all(c in ETHIOPIC_CHARS for c in tok)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer wrappers  (uniform interface)
# ──────────────────────────────────────────────────────────────────────────────

class AbolWrapper:
    """Wraps Abol Tokenizer / HybridTokenizer."""
    def __init__(self, name: str, model):
        self.name = name
        self._model = model
        self.vocab_size = len(model.tokens)
        unk = getattr(model.tokenizer_obj, 'unk_token', '<unk>')
        self._unk_id = model.tokenizer_obj.token2id.get(unk, -1)

    def encode(self, text: str) -> Tuple[List[str], List[bool]]:
        try:
            ids, spans = self._model.encode_with_spans(text)
            tokens = [s['token'] for s in spans]
            is_unk = [s['id'] == self._unk_id for s in spans]
            return tokens, is_unk
        except Exception:
            return [], []

    def decode(self, text: str) -> str:
        try:
            ids, _ = self._model.encode_with_spans(text)
            return self._model.decode(ids)
        except Exception:
            return text


class BPEWrapper:
    """Wraps a HuggingFace BPE tokenizer trained on the Amharic corpus."""
    def __init__(self, corpus_path: str, vocab_size: int = 13_000):
        self.name = "BPE"
        tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]"],
            show_progress=False,
        )
        tokenizer.train([corpus_path], trainer)
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self._unk_str = "[UNK]"

    def encode(self, text: str) -> Tuple[List[str], List[bool]]:
        enc = self._tok.encode(text)
        tokens = enc.tokens
        is_unk = [t == self._unk_str for t in tokens]
        return tokens, is_unk

    def decode(self, text: str) -> str:
        enc = self._tok.encode(text)
        return self._tok.decode(enc.ids)


class TiktokenWrapper:
    """Wraps a tiktoken encoder (GPT-2 or cl100k_base)."""
    def __init__(self, encoding_name: str):
        import tiktoken
        self.name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> Tuple[List[str], List[bool]]:
        try:
            ids = self._enc.encode(text, disallowed_special=())
            tokens = [self._enc.decode([i]) for i in ids]
            is_unk = [False] * len(tokens)   # tiktoken never produces UNK
            return tokens, is_unk
        except Exception:
            return [], []

    def decode(self, text: str) -> str:
        try:
            ids = self._enc.encode(text, disallowed_special=())
            return self._enc.decode(ids)
        except Exception:
            return text


# ──────────────────────────────────────────────────────────────────────────────
# Corpus helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def split_train_test(lines: List[str], test_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    shuffled = lines[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def count_words(sentences: List[str]) -> int:
    return sum(len(s.split()) for s in sentences)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_all(wrapper, sentences: List[str]) -> Tuple[List[str], List[bool]]:
    all_tokens, all_unk = [], []
    for sent in sentences:
        toks, unk = wrapper.encode(sent)
        all_tokens.extend(toks)
        all_unk.extend(unk)
    return all_tokens, all_unk


def compute_unigram_ppl(
    train_tokens: List[str], test_tokens: List[str], smoothing: float = 1e-10
) -> float:
    counts = Counter(train_tokens)
    total = sum(counts.values())
    V = len(counts)
    log_sum = 0.0
    for tok in test_tokens:
        p = (counts.get(tok, 0) + smoothing) / (total + smoothing * V)
        log_sum += math.log(p)
    if not test_tokens:
        return float("inf")
    return math.exp(-log_sum / len(test_tokens))


def compute_throughput(wrapper, sentences: List[str]) -> float:
    start = time.perf_counter()
    total = 0
    for sent in sentences:
        toks, _ = wrapper.encode(sent)
        total += len(toks)
    elapsed = time.perf_counter() - start
    return total / elapsed if elapsed > 0 else 0.0


def compute_reconstruction_scores(wrapper, sentences: List[str]) -> Tuple[float, float]:
    originals, reconstructed = [], []
    for sent in sentences:
        decoded = wrapper.decode(sent)
        originals.append(sent)
        reconstructed.append(decoded)

    bleu = sacrebleu.corpus_bleu(
        reconstructed, [originals],
        tokenize="char", smooth_method="exp", force=True,
    ).score
    chrf = sacrebleu.corpus_chrf(
        reconstructed, [originals], char_order=6, beta=2,
    ).score
    return round(bleu, 2), round(chrf, 2)


def compute_tok_per_word(tokens: List[str], n_words: int) -> float:
    n_real = sum(1 for t in tokens if t.strip() != "")
    return round(n_real / n_words, 3) if n_words else 0.0


def compute_unk_pct(is_unk: List[bool]) -> float:
    if not is_unk:
        return 0.0
    return round(100.0 * sum(is_unk) / len(is_unk), 3)


def compute_fidel_pct(tokens: List[str]) -> float:
    real = [t for t in tokens if t.strip() != ""]
    if not real:
        return 0.0
    return round(100.0 * sum(1 for t in real if is_ethiopic_token(t)) / len(real), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate one wrapper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(wrapper, train_sents: List[str], test_sents: List[str]) -> Dict[str, Any]:
    print(f"  [{wrapper.name}] tokenising train set...")
    train_tokens, _ = tokenize_all(wrapper, train_sents)

    print(f"  [{wrapper.name}] tokenising test set...")
    test_tokens, test_unk = tokenize_all(wrapper, test_sents)

    n_words = count_words(test_sents)

    print(f"  [{wrapper.name}] computing PPL...")
    ppl = compute_unigram_ppl(train_tokens, test_tokens)

    print(f"  [{wrapper.name}] measuring throughput...")
    tps = compute_throughput(wrapper, test_sents)

    print(f"  [{wrapper.name}] computing BLEU / chrF...")
    bleu, chrf = compute_reconstruction_scores(wrapper, test_sents)

    return {
        "Algorithm":   wrapper.name,
        "PPL":         round(ppl, 1),
        "Steps Thr.":  f"{tps:,.0f} tok/s",
        "BLEU":        bleu,
        "chrF":        chrf,
        "Vocab":       f"{wrapper.vocab_size:,}",
        "Tok/w":       compute_tok_per_word(test_tokens, n_words),
        "UNK (%)":     compute_unk_pct(test_unk),
        "Fidel (%)":   compute_fidel_pct(test_tokens),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty table
# ──────────────────────────────────────────────────────────────────────────────

def print_table(results: List[Dict[str, Any]]):
    cols = ["Algorithm", "PPL", "Steps Thr.", "BLEU", "chrF",
            "Vocab", "Tok/w", "UNK (%)", "Fidel (%)"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    hdr = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
    div = sep.replace("-", "=")

    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)
    print(sep)
    print(hdr)
    print(div)

    # Group: Abol first, then standard
    abol  = [r for r in results if r["Algorithm"].startswith("Abol")]
    other = [r for r in results if not r["Algorithm"].startswith("Abol")]

    for r in abol:
        print("|" + "|".join(f" {str(r[c]):<{widths[c]}} " for c in cols) + "|")
    print(sep)
    for r in other:
        print("|" + "|".join(f" {str(r[c]):<{widths[c]}} " for c in cols) + "|")
    print(sep)

    print()
    print("Metric legend:")
    print("  PPL        — Unigram perplexity on held-out corpus         (↓ better)")
    print("  Steps Thr. — Encoding throughput tokens/sec                (↑ better)")
    print("  BLEU       — Reconstruction fidelity 0-100                 (↑ better)")
    print("  chrF       — Character F-score reconstruction 0-100        (↑ better)")
    print("  Vocab      — Total vocabulary size")
    print("  Tok/w      — Average tokens per word                       (↓ more compact)")
    print("  UNK (%)    — Out-of-vocabulary token rate                  (↓ better)")
    print("  Fidel (%)  — % tokens that are valid Ethiopic syllables    (↑ better)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CORPUS_PATH  = "./ahun_corpus.txt"
    GMS_MODEL    = "./model_dir"
    DECOMP_MODEL = "./model_hybrid"

    print("=" * 90)
    print("ABOL TOKENIZER — EVALUATION (Abol-GMS · Abol-Decomposed · BPE · GPT-2 · cl100k)")
    print("=" * 90)

    # Load corpus
    print("\nLoading corpus...")
    lines = load_corpus(CORPUS_PATH)
    train_sents, test_sents = split_train_test(lines, test_ratio=0.15)
    print(f"  Total : {len(lines):,} lines   Train : {len(train_sents):,}   Test : {len(test_sents):,}")
    print(f"  Test words : {count_words(test_sents):,}")

    # ── Load / train tokenizers ───────────────────────────────────────────────
    print("\nPreparing tokenizers...")

    print("  [Abol-GMS] loading model_dir...")
    gms_wrapper = AbolWrapper("Abol-GMS", Tokenizer.load_pretrained(GMS_MODEL))

    print("  [Abol-Decomposed] loading model_hybrid...")
    decomp_wrapper = AbolWrapper("Abol-Decomposed", HybridTokenizer.load_pretrained(DECOMP_MODEL))

    print("  [BPE] training on corpus (vocab ≈ 13 K)...")
    bpe_wrapper = BPEWrapper(CORPUS_PATH, vocab_size=13_000)
    print(f"    → vocab size: {bpe_wrapper.vocab_size:,}")

    print("  [GPT-2] loading tiktoken gpt2 encoding...")
    gpt2_wrapper = TiktokenWrapper("gpt2")

    print("  [cl100k] loading tiktoken cl100k_base encoding...")
    cl100k_wrapper = TiktokenWrapper("cl100k_base")

    wrappers = [gms_wrapper, decomp_wrapper, bpe_wrapper, gpt2_wrapper, cl100k_wrapper]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning evaluation...\n")
    results = []
    for w in wrappers:
        results.append(evaluate(w, train_sents, test_sents))
        print()

    # ── Print & save ──────────────────────────────────────────────────────────
    print_table(results)

    out_path = "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved → {out_path}")
