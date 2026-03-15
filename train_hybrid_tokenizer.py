#!/usr/bin/env python3
"""
Train Hybrid Tokenizer

Uses decomposition logic BUT keeps whole words from dictionary as single tokens
"""

import sys
import os
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset/training_corpus.txt"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "./model_hybrid"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: Corpus not found: {corpus_path}")
        return
    
    print("=" * 80)
    print("TRAINING HYBRID TOKENIZER")
    print("=" * 80)
    print()
    print("Logic:")
    print("  ✓ Works in CV (decomposed) space")
    print("  ✓ BUT keeps whole words if found in dictionary")
    print("  ✓ Example: 'ይጫወታሉ' in corpus → ONE token")
    print()
    
    # Train
    tok = HybridTokenizer()
    tok.train_from_corpus(corpus_path)
    
    # Save
    print()
    tok.save_pretrained(outdir)
    
    # Test
    print()
    print("=" * 80)
    print("🧪 TESTING")
    print("=" * 80)
    
    test_cases = [
        "ልጆች ይጫወታሉ",
        "ሰዎች",
        "በሰው",
        "የልጆች",
        "እንዴት ነህ?",
        "ሰላም",
    ]
    
    print()
    for text in test_cases:
        ids, spans = tok.encode_with_spans(text)
        tokens = [s['token'] for s in spans]
        decoded = tok.decode(ids, spans=spans)
        status = "✓" if decoded == text else "✗"
        
        print(f"'{text}':")
        print(f"  → Tokens: {tokens}")
        print(f"  → Count: {len(tokens)} token(s)")
        print(f"  → Decoded: '{decoded}' {status}")
        print()
    
    print("=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {outdir}/")
    print(f"Vocabulary: {len(tok.tokens):,} tokens")
    print(f"Corpus words: {len(tok.original_corpus_words):,} unique words")
    print()


if __name__ == "__main__":
    main()
