"""
Decomposed Tokenizer

This tokenizer works in the decomposed consonant-vowel space for better
morphological awareness and linguistic consistency.

Process:
1. Training: Decompose corpus → Run greedy algorithm → Build vocab
2. Encoding: Decompose input → Tokenize → Return IDs
3. Decoding: IDs → Tokens → Recompose → Return original text
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import os

from .fidel_decomposer import FidelDecomposer, AMHARIC_FIDEL_MAP
from .automata import split_corpus_greedy_subword, split_corpus_ac
from .utils import simple_tokenize_keep_punct, load_text_file, map_non_ethiopic_to_placeholders


class DecomposedVocabTokenizer:
    """
    Vocabulary-based tokenizer that operates in decomposed CV space.
    """
    
    def __init__(self, tokens: Optional[List[str]] = None, 
                 decomposer: Optional[FidelDecomposer] = None,
                 pad_token: str = "<pad>", unk_token: str = "<unk>", 
                 start_ids_from: int = 0):
        """
        Initialize tokenizer with decomposed vocabulary.
        
        Args:
            tokens: List of tokens (in decomposed form)
            decomposer: FidelDecomposer instance
            pad_token: Padding token
            unk_token: Unknown token
            start_ids_from: Starting ID for tokens
        """
        tokens = tokens or []
        self.decomposer = decomposer if decomposer is not None else FidelDecomposer()
        
        # Ensure special tokens are present
        if pad_token not in tokens:
            tokens = [pad_token] + tokens
        if unk_token not in tokens:
            tokens = [unk_token] + tokens
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2id = {t: i + start_ids_from for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.max_token_len = max((len(t) for t in tokens), default=1)
        
        # Composite-id scheme for leading-space variants
        self.COMPOSITE_FACTOR = 2
        max_base_id = max(self.token2id.values(), default=-1)
        self.COMPOSITE_OFFSET = (max_base_id + 1) * self.COMPOSITE_FACTOR + 1
    
    def tokenize_with_spans(self, text: str, oov_strategy: str = "unk") -> List[Dict[str, Any]]:
        """
        Tokenize text and return spans with original text preserved.
        
        Process:
        1. Decompose input text
        2. Tokenize in decomposed space
        3. Track original positions
        4. Return spans with both decomposed and original forms
        
        Args:
            text: Original Amharic text
            oov_strategy: How to handle OOV tokens ("unk" or "char")
            
        Returns:
            List of span dictionaries with token info
        """
        # Step 1: Decompose the input text
        decomposed_text = self.decomposer.decompose_word(text)
        
        # Step 2: Tokenize in decomposed space
        cp = list(decomposed_text)
        n = len(cp)
        i = 0
        records = []
        
        # Track original position for reconstruction
        original_pos = 0
        
        while i < n:
            # Handle leading whitespace
            had_leading_space = False
            start_pos = i
            while i < n and cp[i].isspace():
                had_leading_space = True
                i += 1
            if i >= n:
                break
            
            # Greedy longest-match in decomposed space
            longest_tok = None
            longest_len = 0
            for L in range(self.max_token_len, 0, -1):
                if i + L > n:
                    continue
                cand = ''.join(cp[i:i+L])
                if cand in self.token2id:
                    longest_tok = cand
                    longest_len = L
                    break
            
            if longest_tok is not None:
                base_tid = self.token2id[longest_tok]
                comp_id = self.COMPOSITE_OFFSET + base_tid * self.COMPOSITE_FACTOR + (1 if had_leading_space else 0)
                
                # Recompose the token to original form for display
                original_token = self.decomposer.recompose_word(longest_tok)
                text_piece = (" " if had_leading_space else "") + original_token
                
                records.append({
                    "id": comp_id,
                    "token": original_token,  # Store ORIGINAL form for user
                    "decomposed_token": longest_tok,  # Store decomposed for internal use
                    "start": start_pos,
                    "end": i + longest_len,
                    "text": text_piece
                })
                i += longest_len
                continue
            
            # OOV handling
            ch = cp[i]
            if oov_strategy == 'unk':
                if self.unk_token is None:
                    raise KeyError('OOV and no unk_token set')
                base_tid = self.token2id[self.unk_token]
                comp_id = self.COMPOSITE_OFFSET + base_tid * self.COMPOSITE_FACTOR + (1 if had_leading_space else 0)
                text_piece = (" " if had_leading_space else "") + ch
                records.append({
                    "id": comp_id, 
                    "token": self.unk_token, 
                    "decomposed_token": ch,
                    "start": start_pos, 
                    "end": i+1, 
                    "text": text_piece
                })
                i += 1
            else:
                # char fallback
                if ch in self.token2id:
                    base_tid = self.token2id[ch]
                else:
                    base_tid = self.token2id[self.unk_token]
                comp_id = self.COMPOSITE_OFFSET + base_tid * self.COMPOSITE_FACTOR + (1 if had_leading_space else 0)
                text_piece = (" " if had_leading_space else "") + ch
                records.append({
                    "id": comp_id, 
                    "token": ch if ch in self.token2id else self.unk_token,
                    "decomposed_token": ch,
                    "start": start_pos, 
                    "end": i+1, 
                    "text": text_piece
                })
                i += 1
        
        return records
    
    def detokenize_from_spans(self, spans: List[Dict[str, Any]]) -> str:
        """Reconstruct text from spans (already in original form)"""
        return ''.join(rec['text'] for rec in spans)
    
    def tokenize(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Tokenize text and return only IDs"""
        spans = self.tokenize_with_spans(text, oov_strategy=oov_strategy)
        return [r['id'] for r in spans]
    
    def detokenize(self, ids: List[int], unk_placeholder: Optional[str] = None) -> str:
        """
        Detokenize IDs back to original text.
        
        Process:
        1. Convert IDs to decomposed tokens
        2. Join decomposed tokens
        3. Recompose to original Amharic
        """
        if unk_placeholder is None:
            unk_placeholder = self.unk_token if self.unk_token is not None else "<unk>"
        
        # Collect decomposed tokens
        decomposed_parts = []
        id2t = self.id2token
        
        for i in ids:
            # Detect composite leading-space ids
            if isinstance(i, int) and i >= self.COMPOSITE_OFFSET:
                rel = i - self.COMPOSITE_OFFSET
                base_id = rel // self.COMPOSITE_FACTOR
                flag = rel % self.COMPOSITE_FACTOR
                tok = id2t.get(base_id)
                if tok is None:
                    decomposed_parts.append(unk_placeholder)
                else:
                    piece = (" " if flag == 1 else "") + (unk_placeholder if (self.unk_token is not None and tok == self.unk_token) else tok)
                    decomposed_parts.append(piece)
            else:
                tok = id2t.get(i)
                if tok is None:
                    decomposed_parts.append(unk_placeholder)
                else:
                    if self.unk_token is not None and tok == self.unk_token:
                        decomposed_parts.append(unk_placeholder)
                    else:
                        decomposed_parts.append(tok)
        
        # Join and recompose
        decomposed_text = ''.join(decomposed_parts)
        original_text = self.decomposer.recompose_word(decomposed_text)
        
        return original_text


@dataclass
class DecomposedTokenizer:
    """
    Main tokenizer class that uses fidel decomposition.
    
    User always sees original Amharic text, but tokenization
    happens in the decomposed consonant-vowel space.
    """
    tokens: List[str] = field(default_factory=list)
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    tokenizer_obj: Optional[DecomposedVocabTokenizer] = None
    decomposer: Optional[FidelDecomposer] = None
    
    def __post_init__(self):
        if self.decomposer is None:
            self.decomposer = FidelDecomposer()
    
    def train_from_corpus(self, corpus_path: str, use_new_algorithm: bool = True) -> None:
        """
        Train tokenizer from corpus using decomposition.
        
        Process:
        1. Load corpus
        2. Decompose entire corpus to CV space
        3. Run greedy/AC algorithm on decomposed corpus
        4. Build vocabulary in decomposed space
        5. Add all CV pairs as fallback
        
        Args:
            corpus_path: Path to training corpus
            use_new_algorithm: Use greedy (True) or suffix automaton (False)
        """
        # Load and preprocess corpus
        text = load_text_file(corpus_path)
        if text is None:
            raise FileNotFoundError(corpus_path)
        text = map_non_ethiopic_to_placeholders(text)
        corpus = simple_tokenize_keep_punct(text)
        
        print(f"Loaded corpus: {len(corpus):,} words")
        
        # Step 1: Decompose entire corpus
        print("Decomposing corpus to CV space...")
        decomposed_corpus = self.decomposer.decompose_corpus(corpus)
        
        # Step 2: Run morphological analysis on decomposed corpus
        print(f"Running {'greedy' if use_new_algorithm else 'suffix automaton'} algorithm...")
        if use_new_algorithm:
            prefixes, stems, suffixes = split_corpus_greedy_subword(decomposed_corpus)
        else:
            prefixes, stems, suffixes = split_corpus_ac(decomposed_corpus)
        
        print(f"  Found {len(prefixes)} prefixes, {len(stems)} stems, {len(suffixes)} suffixes")
        
        # Step 3: Build vocabulary in decomposed space
        tokenset = set()
        tokenset.update(prefixes)
        tokenset.update(stems)
        tokenset.update(suffixes)
        
        # Add all CV pairs from fidel map as fallback
        for cv_pair in self.decomposer.fidel_to_cv.values():
            tokenset.add(cv_pair)
        
        # Add space
        tokenset.add(" ")
        
        # Remove empty strings
        tokenset.discard('')
        tokenset.discard(None)
        
        # Sort by length (longest first)
        tokens = sorted(tokenset, key=lambda x: (-len(x), x))
        self.tokens = tokens
        
        print(f"Built vocabulary: {len(tokens):,} tokens (in decomposed space)")
        
        # Create tokenizer object
        self.tokenizer_obj = DecomposedVocabTokenizer(
            tokens=tokens, 
            decomposer=self.decomposer,
            pad_token=self.pad_token, 
            unk_token=self.unk_token
        )
    
    def save_pretrained(self, outdir: str) -> None:
        """Save model to directory"""
        os.makedirs(outdir, exist_ok=True)
        
        # Save vocabulary (decomposed tokens)
        vocab = {t: i for t, i in self.tokenizer_obj.token2id.items()}
        with open(os.path.join(outdir, 'vocab.json'), 'w', encoding='utf-8') as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        
        # Save config
        cfg = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'tokens': sorted(self.tokens, key=lambda x: self.tokenizer_obj.token2id.get(x, 0)),
            'algorithm': 'decomposed',
            'fidel_map': AMHARIC_FIDEL_MAP
        }
        with open(os.path.join(outdir, 'tokenizer.json'), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        
        print(f"Saved decomposed tokenizer to {outdir}/")
    
    @classmethod
    def load_pretrained(cls, indir: str) -> 'DecomposedTokenizer':
        """Load model from directory"""
        with open(os.path.join(indir, 'tokenizer.json'), 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        
        t = cls()
        t.pad_token = cfg.get('pad_token', '<pad>')
        t.unk_token = cfg.get('unk_token', '<unk>')
        t.tokens = cfg.get('tokens', [])
        
        # Load fidel map if saved
        fidel_map = cfg.get('fidel_map', AMHARIC_FIDEL_MAP)
        t.decomposer = FidelDecomposer(fidel_map)
        
        # Create tokenizer object
        t.tokenizer_obj = DecomposedVocabTokenizer(
            tokens=t.tokens,
            decomposer=t.decomposer,
            pad_token=t.pad_token,
            unk_token=t.unk_token
        )
        
        return t
    
    def encode_with_spans(self, text: str, oov_strategy: str = 'unk') -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Encode text and return IDs + spans.
        
        User sees original tokens, but encoding happens in decomposed space.
        """
        spans = self.tokenizer_obj.tokenize_with_spans(text, oov_strategy=oov_strategy)
        ids = [r['id'] for r in spans]
        return ids, spans
    
    def encode(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Encode text to IDs"""
        return self.tokenizer_obj.tokenize(text, oov_strategy=oov_strategy)
    
    def decode(self, ids: List[int], spans: Optional[List[Dict[str, Any]]] = None, unk_placeholder: str = '<unk>') -> str:
        """
        Decode IDs back to original text.
        
        User sees original Amharic, recomposed from decomposed tokens.
        """
        if spans is not None:
            return self.tokenizer_obj.detokenize_from_spans(spans)
        return self.tokenizer_obj.detokenize(ids, unk_placeholder=unk_placeholder)


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("DECOMPOSED TOKENIZER DEMO")
    print("=" * 80)
    
    # Create small test corpus
    import tempfile
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as fh:
        fh.write('ሰላም ልጅ\nእንዴት ነህ\nመጽሐፍ ቤት')
        corpus_file = fh.name
    
    # Train
    print("\n1. Training tokenizer...")
    tok = DecomposedTokenizer()
    tok.train_from_corpus(corpus_file, use_new_algorithm=True)
    
    # Test
    print("\n2. Testing encoding/decoding...")
    test_texts = ["ሰላም", "ልጅ", "እንዴት ነህ?"]
    
    for text in test_texts:
        ids, spans = tok.encode_with_spans(text)
        tokens = [s['token'] for s in spans]  # User sees original tokens
        decoded = tok.decode(ids, spans=spans)
        
        print(f"\n  Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs: {ids}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {'✓' if decoded == text else '✗'}")
    
    # Cleanup
    os.unlink(corpus_file)
    
    print("\n" + "=" * 80)

