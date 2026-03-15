#!/usr/bin/env python3
"""
Training Script for Decomposed Tokenizer

This script trains the new fidel-decomposition based tokenizer.
"""

import os
import argparse
from amharic_tokenizer.decomposed_tokenizer import DecomposedTokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Decomposed Tokenizer')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to training corpus file')
    parser.add_argument('--outdir', type=str, default='./model_decomposed',
                        help='Output directory for trained model')
    parser.add_argument('--use-old-algorithm', action='store_true',
                        help='Use old Suffix Automaton algorithm instead of Greedy')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING DECOMPOSED TOKENIZER")
    print("=" * 80)
    print()
    
    # Check corpus exists
    if not os.path.exists(args.corpus):
        print(f"❌ Error: Corpus file not found: {args.corpus}")
        return
    
    corpus_size = os.path.getsize(args.corpus)
    print(f"📂 Corpus: {args.corpus}")
    print(f"   Size: {corpus_size:,} bytes")
    print()
    
    # Train
    print("🔄 Training tokenizer...")
    print(f"   Algorithm: {'OLD (Suffix Automaton)' if args.use_old_algorithm else 'NEW (Greedy Meaningful Subword)'}")
    print(f"   Method: Fidel Decomposition (CV space)")
    print()
    
    tokenizer = DecomposedTokenizer()
    tokenizer.train_from_corpus(
        args.corpus, 
        use_new_algorithm=not args.use_old_algorithm
    )
    
    print()
    print("💾 Saving model...")
    tokenizer.save_pretrained(args.outdir)
    
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Model saved to: {args.outdir}/")
    print(f"Vocabulary size: {len(tokenizer.tokens):,} tokens")
    print()
    
    # Quick test
    print("🧪 Quick Test:")
    test_texts = ["ሰላም", "እንዴት ነህ", "መጽሐፍ"]
    for text in test_texts:
        ids, spans = tokenizer.encode_with_spans(text)
        tokens = [s['token'] for s in spans]
        decoded = tokenizer.decode(ids, spans=spans)
        status = "✓" if decoded == text else "✗"
        print(f"  '{text}' → {tokens} → '{decoded}' {status}")
    
    print()
    print("📚 Next Steps:")
    print(f"  1. Test the model: python demo_decomposed_tokenizer.py")
    print(f"  2. Compare with original: python compare_tokenizers.py")
    print(f"  3. Use in your code:")
    print(f"     from amharic_tokenizer.decomposed_tokenizer import DecomposedTokenizer")
    print(f"     tok = DecomposedTokenizer.load_pretrained('{args.outdir}')")
    print()


if __name__ == "__main__":
    main()

