#!/usr/bin/env python3
"""
Evaluation script for Abol Tokenizer algorithms.

Metrics:
  PPL          - Unigram perplexity on held-out corpus (lower = better)
  Steps Thr.   - Encoding throughput in tokens/second (higher = better)
  BLEU         - Reconstruction BLEU: original vs decoded text (higher = better)
  chrF         - Reconstruction character F-score (higher = better)
  Vocab        - Vocabulary size
  Tok/w        - Average tokens per word (lower = more compact)
  UNK (%)      - Percentage of unknown tokens (lower = better)
  Fidel (%)    - Percentage of tokens that are valid Ethiopic syllables (higher = better)
"""

import time
import math
import re
import unicodedata
from collections import Counter
from typing import List, Tuple

import sacrebleu

from amharic_tokenizer.tokenizer import Tokenizer
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

ETHIOPIC_RANGE = range(0x1200, 0x1380)
ETHIOPIC_CHARS = set(chr(c) for c in ETHIOPIC_RANGE)


def is_ethiopic_token(tok: str) -> bool:
    """True if every character in tok is an Ethiopic syllable."""
    return bool(tok) and all(c in ETHIOPIC_CHARS for c in tok)


def load_corpus(path: str) -> List[str]:
    """Load corpus as a list of non-empty lines."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def split_train_test(lines: List[str], test_ratio: float = 0.15, seed: int = 42):
    import random
    random.seed(seed)
    shuffled = lines[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


def tokenize_corpus(tokenizer, sentences: List[str]):
    """Return flat list of (token_str, is_unk) pairs for all sentences."""
    results = []
    for sent in sentences:
        try:
            ids, spans = tokenizer.encode_with_spans(sent)
            for span in spans:
                tok = span["token"]
                tid = span["id"]
                # Both tokenizers use unk_token for OOV
                unk_id = tokenizer.tokenizer_obj.token2id.get(
                    tokenizer.tokenizer_obj.unk_token, -999
                )
                results.append((tok, tid == unk_id))
        except Exception:
            pass
    return results


def count_words(sentences: List[str]) -> int:
    """Count whitespace-separated words."""
    return sum(len(s.split()) for s in sentences)


# ──────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_unigram_ppl(
    train_tokens: List[str], test_tokens: List[str], smoothing: float = 1e-10
) -> float:
    """
    Unigram perplexity: fit a unigram distribution on training tokens,
    evaluate cross-entropy on test tokens.
    PPL = exp(H)  where H = -1/N * Σ log P(token)
    """
    counts = Counter(train_tokens)
    total = sum(counts.values())
    log_prob_sum = 0.0
    N = 0
    for tok in test_tokens:
        p = (counts.get(tok, 0) + smoothing) / (total + smoothing * len(counts))
        log_prob_sum += math.log(p)
        N += 1
    if N == 0:
        return float("inf")
    entropy = -log_prob_sum / N
    return math.exp(entropy)


def compute_throughput(tokenizer, sentences: List[str]) -> Tuple[float, int]:
    """
    Measure encoding throughput.
    Returns (tokens_per_second, total_tokens).
    """
    start = time.perf_counter()
    total_tokens = 0
    for sent in sentences:
        try:
            ids, spans = tokenizer.encode_with_spans(sent)
            total_tokens += len(spans)
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    return tps, total_tokens


def compute_reconstruction_bleu_chrf(
    tokenizer, sentences: List[str]
) -> Tuple[float, float]:
    """
    Encode then decode each sentence, compare to original.
    Returns (BLEU, chrF).
    """
    originals = []
    reconstructed = []
    for sent in sentences:
        try:
            ids, spans = tokenizer.encode_with_spans(sent)
            decoded = tokenizer.decode(ids)
            originals.append(sent)
            reconstructed.append(decoded)
        except Exception:
            pass

    if not originals:
        return 0.0, 0.0

    bleu = sacrebleu.corpus_bleu(
        reconstructed,
        [originals],
        tokenize="char",        # character-level is more meaningful for Amharic
        smooth_method="exp",
        force=True,
    ).score

    chrf = sacrebleu.corpus_chrf(
        reconstructed,
        [originals],
        char_order=6,
        beta=2,
    ).score

    return round(bleu, 2), round(chrf, 2)


def compute_vocab_stats(tokenizer) -> int:
    return len(tokenizer.tokens)


def compute_tok_per_word(token_pairs: List[Tuple[str, bool]], n_words: int) -> float:
    if n_words == 0:
        return 0.0
    # Exclude space tokens when counting
    n_real = sum(1 for tok, _ in token_pairs if tok.strip() != "")
    return round(n_real / n_words, 3)


def compute_unk_pct(token_pairs: List[Tuple[str, bool]]) -> float:
    if not token_pairs:
        return 0.0
    n_unk = sum(1 for _, is_unk in token_pairs if is_unk)
    return round(100.0 * n_unk / len(token_pairs), 3)


def compute_fidel_pct(token_pairs: List[Tuple[str, bool]]) -> float:
    """% of tokens (excluding spaces/punct) that are purely Ethiopic syllables."""
    real_toks = [(tok, iu) for tok, iu in token_pairs if tok.strip() != ""]
    if not real_toks:
        return 0.0
    n_fidel = sum(1 for tok, _ in real_toks if is_ethiopic_token(tok))
    return round(100.0 * n_fidel / len(real_toks), 2)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(name: str, tokenizer, train_sents: List[str], test_sents: List[str]):
    print(f"\n  Evaluating {name}...")

    # Tokenize train + test
    train_pairs = tokenize_corpus(tokenizer, train_sents)
    test_pairs  = tokenize_corpus(tokenizer, test_sents)

    train_tokens = [tok for tok, _ in train_pairs]
    test_tokens  = [tok for tok, _ in test_pairs]

    n_words = count_words(test_sents)

    # Metrics
    ppl          = compute_unigram_ppl(train_tokens, test_tokens)
    tps, _       = compute_throughput(tokenizer, test_sents)
    bleu, chrf   = compute_reconstruction_bleu_chrf(tokenizer, test_sents)
    vocab        = compute_vocab_stats(tokenizer)
    tok_per_word = compute_tok_per_word(test_pairs, n_words)
    unk_pct      = compute_unk_pct(test_pairs)
    fidel_pct    = compute_fidel_pct(test_pairs)

    return {
        "Algorithm":   name,
        "PPL":         round(ppl, 1),
        "Steps Thr.":  f"{tps:,.0f} tok/s",
        "BLEU":        bleu,
        "chrF":        chrf,
        "Vocab":       f"{vocab:,}",
        "Tok/w":       tok_per_word,
        "UNK (%)":     unk_pct,
        "Fidel (%)":   fidel_pct,
    }


def print_table(results: list):
    cols = ["Algorithm", "PPL", "Steps Thr.", "BLEU", "chrF",
            "Vocab", "Tok/w", "UNK (%)", "Fidel (%)"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in results)) for c in cols}

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    hdr = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(sep)
    print(hdr)
    print(sep.replace("-", "="))
    for r in results:
        row = "|" + "|".join(f" {str(r[c]):<{widths[c]}} " for c in cols) + "|"
        print(row)
    print(sep)
    print()

    print("Metric descriptions:")
    print("  PPL        — Unigram perplexity on held-out corpus (lower = better)")
    print("  Steps Thr. — Encoding throughput in tokens/second (higher = better)")
    print("  BLEU       — Reconstruction fidelity score 0-100 (higher = better)")
    print("  chrF       — Character F-score reconstruction 0-100 (higher = better)")
    print("  Vocab      — Total vocabulary size")
    print("  Tok/w      — Average tokens per word (lower = more compact)")
    print("  UNK (%)    — Out-of-vocabulary token rate (lower = better)")
    print("  Fidel (%)  — % of tokens that are valid Ethiopic syllables (higher = better)")


if __name__ == "__main__":
    CORPUS_PATH  = "./ahun_corpus.txt"
    GMS_MODEL    = "./model_dir"
    DECOMP_MODEL = "./model_hybrid"

    print("=" * 80)
    print("ABOL TOKENIZER — EVALUATION")
    print("=" * 80)

    # Load corpus
    print("\nLoading corpus...")
    lines = load_corpus(CORPUS_PATH)
    train_sents, test_sents = split_train_test(lines, test_ratio=0.15)
    print(f"  Total lines  : {len(lines):,}")
    print(f"  Train lines  : {len(train_sents):,}")
    print(f"  Test lines   : {len(test_sents):,}")
    print(f"  Test words   : {count_words(test_sents):,}")

    # Load models
    print("\nLoading models...")
    print("  Loading Abol-GMS (model_dir)...")
    gms = Tokenizer.load_pretrained(GMS_MODEL)

    print("  Loading Abol-Decomposed (model_hybrid)...")
    decomp = HybridTokenizer.load_pretrained(DECOMP_MODEL)

    # Evaluate
    results = []
    results.append(evaluate("Abol-GMS",         gms,   train_sents, test_sents))
    results.append(evaluate("Abol-Decomposed",   decomp, train_sents, test_sents))

    # Print table
    print_table(results)

    # Save to JSON
    import json
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Results saved to evaluation_results.json")
