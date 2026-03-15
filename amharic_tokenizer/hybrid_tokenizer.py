"""
Hybrid Tokenizer - Smart Morphological + Whole Word

Logic:
1. Decompose word to CV space
2. Check for morphological patterns (suffix/prefix) → SPLIT them
3. For remaining words: if in dictionary → ONE token
4. Otherwise → use greedy algorithm

Example:
- "ልጆች" → ልጅ + ኦች (morphological split)
- "ይጫወታሉ" in corpus → ይጫወታሉ (keep whole)
- "ልጆች ይጫወታሉ" → ['ልጅ', 'ኦች', 'ይጫወታሉ']
"""

from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import json
import os

from .fidel_decomposer import FidelDecomposer, AMHARIC_FIDEL_MAP
from .automata import split_corpus_greedy_subword
from .utils import simple_tokenize_keep_punct, load_text_file, map_non_ethiopic_to_placeholders


# Known morphological suffixes (ORDERED: longest first!)
MORPHOLOGICAL_SUFFIXES = [
    "ዎቹን",   # plural definite + accusative
    "ዎቹ",    # plural definite
    "ኦች",    # plural (KEY!)
    "ዎች",    # plural variant
    "ዋን",    # definite + accusative
    "ቱን",    # definite masc + accusative
    "ቱ",     # definite masculine
    "ዋ",     # definite feminine
    "ች",     # feminine/plural
    "ን",     # accusative
]

# Known morphological prefixes
MORPHOLOGICAL_PREFIXES = [
    "እንደ",   # like, as
    "እስከ",   # until
    "ወደ",    # to, towards
    "ስለ",    # about
    "የ",     # of, possessive
    "ለ",     # for, to
    "በ",     # by, in, at
    "ከ",     # from
]


class HybridVocabTokenizer:
    """
    Tokenizer that works in CV space but respects whole-word boundaries.
    """
    
    def __init__(self, tokens: Optional[List[str]] = None,
                 original_corpus_words: Optional[Set[str]] = None,
                 decomposer: Optional[FidelDecomposer] = None,
                 pad_token: str = "<pad>", unk_token: str = "<unk>",
                 start_ids_from: int = 0):
        """
        Initialize with vocabulary and original corpus words.
        
        Args:
            tokens: Decomposed tokens for vocabulary
            original_corpus_words: Set of original words from corpus
            decomposer: FidelDecomposer instance
        """
        tokens = tokens or []
        self.decomposer = decomposer if decomposer is not None else FidelDecomposer()
        self.original_corpus_words = original_corpus_words or set()
        
        # Ensure special tokens
        if pad_token not in tokens:
            tokens = [pad_token] + tokens
        if unk_token not in tokens:
            tokens = [unk_token] + tokens
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2id = {t: i + start_ids_from for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.max_token_len = max((len(t) for t in tokens), default=1)
        
        # Composite-id scheme
        self.COMPOSITE_FACTOR = 2
        max_base_id = max(self.token2id.values(), default=-1)
        self.COMPOSITE_OFFSET = (max_base_id + 1) * self.COMPOSITE_FACTOR + 1
    
    def tokenize_with_spans(self, text: str, oov_strategy: str = "unk") -> List[Dict[str, Any]]:
        """
        Tokenize with word-boundary awareness.
        
        Process:
        1. Split text into words
        2. For each word:
           a. Check if word exists in original corpus → keep as one token
           b. Otherwise, decompose and use greedy algorithm
        3. Track spans and recompose
        """
        # Split into words and spaces
        words_and_spaces = []
        current_word = []
        
        for char in text:
            if char.isspace() or char in '.,;:!?።፡፤፣':
                if current_word:
                    words_and_spaces.append(('word', ''.join(current_word)))
                    current_word = []
                words_and_spaces.append(('space', char))
            else:
                current_word.append(char)
        
        if current_word:
            words_and_spaces.append(('word', ''.join(current_word)))
        
        # Tokenize each part
        records = []
        position = 0
        
        for part_type, part_text in words_and_spaces:
            if part_type == 'space':
                # Keep spaces as-is
                records.append({
                    "id": self.token2id.get(part_text, self.token2id[self.unk_token]),
                    "token": part_text,
                    "decomposed_token": part_text,
                    "start": position,
                    "end": position + len(part_text),
                    "text": part_text
                })
                position += len(part_text)
            else:
                # Process word
                word = part_text
                word_tokens = self._split_word_smart(word)
                
                # Add each token
                for original_tok in word_tokens:
                    decomp_tok = self.decomposer.decompose_word(original_tok)
                    token_id = self.token2id.get(decomp_tok, self.token2id[self.unk_token])
                    
                    records.append({
                        "id": token_id,
                        "token": original_tok,
                        "decomposed_token": decomp_tok,
                        "start": position,
                        "end": position + len(original_tok),
                        "text": original_tok
                    })
                    position += len(original_tok)
        
        return records
    
    def _split_word_smart(self, word: str) -> List[str]:
        """
        Smart word splitting with morphological awareness in CV space.
        
        Priority:
        1. Check for morphological suffixes in CV space (ኦችእ, etc.) → SPLIT
        2. Check if whole word is in dictionary → KEEP
        3. Otherwise use greedy algorithm
        
        Example:
        - "ልጆች" (ልእጅእኦችእ) → "ልጅ" + "ኦች" (morphological split)
        - "ይጫወታሉ" → "ይጫወታሉ" (whole word if in dictionary)
        """
        # Decompose word to CV space
        decomposed_word = self.decomposer.decompose_word(word)
        
        # STEP 1: Check for morphological suffixes IN CV SPACE
        for suffix_pattern in MORPHOLOGICAL_SUFFIXES:
            decomposed_suffix = self.decomposer.decompose_word(suffix_pattern)
            
            if decomposed_word.endswith(decomposed_suffix) and len(decomposed_word) > len(decomposed_suffix):
                # Found a morphological suffix in CV space
                decomposed_stem = decomposed_word[:-len(decomposed_suffix)]
                
                # Recompose to get original forms
                stem = self.decomposer.recompose_word(decomposed_stem)
                
                # Check if stem is meaningful (in dictionary or long enough)
                if stem in self.original_corpus_words or len(stem) >= 1:
                    # SPLIT: stem + suffix
                    return [stem, suffix_pattern]
        
        # STEP 2: Check if whole word is in dictionary
        if word in self.original_corpus_words:
            # Keep as ONE token
            return [word]
        
        # STEP 3: Check for morphological prefixes IN CV SPACE
        for prefix_pattern in MORPHOLOGICAL_PREFIXES:
            decomposed_prefix = self.decomposer.decompose_word(prefix_pattern)
            
            if decomposed_word.startswith(decomposed_prefix) and len(decomposed_word) > len(decomposed_prefix):
                # Found a morphological prefix in CV space
                decomposed_stem = decomposed_word[len(decomposed_prefix):]
                stem = self.decomposer.recompose_word(decomposed_stem)
                
                # Check if stem is meaningful
                if stem in self.original_corpus_words or len(stem) >= 2:
                    # Recursively split the stem
                    stem_tokens = self._split_word_smart(stem)
                    return [prefix_pattern] + stem_tokens
        
        # STEP 4: No morphological pattern found, use greedy algorithm
        decomposed_tokens = self._greedy_tokenize(decomposed_word)
        
        # Recompose each token
        return [self.decomposer.recompose_word(tok) for tok in decomposed_tokens]
    
    def _greedy_tokenize(self, decomposed_word: str) -> List[str]:
        """
        Greedy tokenization in CV space.
        Finds longest matches in vocabulary.
        """
        tokens = []
        i = 0
        
        while i < len(decomposed_word):
            # Skip spaces
            if decomposed_word[i].isspace():
                tokens.append(decomposed_word[i])
                i += 1
                continue
            
            # Find longest match
            longest_match = None
            longest_len = 0
            
            for length in range(min(self.max_token_len, len(decomposed_word) - i), 0, -1):
                candidate = decomposed_word[i:i+length]
                if candidate in self.token2id:
                    longest_match = candidate
                    longest_len = length
                    break
            
            if longest_match:
                tokens.append(longest_match)
                i += longest_len
            else:
                # Fallback: single character
                tokens.append(decomposed_word[i])
                i += 1
        
        return tokens
    
    def tokenize(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Tokenize and return IDs"""
        spans = self.tokenize_with_spans(text, oov_strategy=oov_strategy)
        return [r['id'] for r in spans]
    
    def _merge_cv_morpheme_boundaries(self, cv_parts: List[str]) -> str:
        """
        Merge morpheme boundaries in CV space with morphophonemic rules.
        
        Key rule: ውእ + ኦ... → ውኦ... (e.g., ልእጅእ + ኦችእ → ልእጅኦችእ)
        Then recompose: ልእጅኦችእ → ልጆች
        """
        if len(cv_parts) <= 1:
            merged_cv = ''.join(cv_parts)
            return self.decomposer.recompose_word(merged_cv)
        
        merged_cv_parts = []
        i = 0
        
        while i < len(cv_parts):
            current_cv = cv_parts[i]
            
            # Check if we can merge with next token
            if i + 1 < len(cv_parts):
                next_cv = cv_parts[i + 1]
                
                # Rule: if current ends with እ, and next starts with vowel
                # Merge them: ውእ + ኦችእ → ውኦችእ
                if len(current_cv) >= 1 and len(next_cv) >= 1:
                    if current_cv.endswith('እ') and next_cv[0] in 'አኡኢኣእኤኦ':
                        # Morphophonemic fusion in CV space
                        # Remove the እ, merge
                        merged_cv_parts.append(current_cv[:-1] + next_cv)
                        i += 2  # Skip both tokens
                        continue
            
            # No merging, add current as-is
            merged_cv_parts.append(current_cv)
            i += 1
        
        # Join all CV parts and recompose
        full_cv = ''.join(merged_cv_parts)
        return self.decomposer.recompose_word(full_cv)
    
    def detokenize(self, ids: List[int], unk_placeholder: Optional[str] = None) -> str:
        """Detokenize IDs to original text with morpheme boundary merging"""
        if unk_placeholder is None:
            unk_placeholder = self.unk_token if self.unk_token is not None else "<unk>"
        
        # Get CV tokens
        cv_parts = []
        for i in ids:
            if isinstance(i, int) and i >= self.COMPOSITE_OFFSET:
                rel = i - self.COMPOSITE_OFFSET
                base_id = rel // self.COMPOSITE_FACTOR
                tok = self.id2token.get(base_id, unk_placeholder)
            else:
                tok = self.id2token.get(i, unk_placeholder)
            
            cv_parts.append(tok)
        
        # Merge morpheme boundaries in CV space, then recompose
        return self._merge_cv_morpheme_boundaries(cv_parts)


@dataclass
class HybridTokenizer:
    """
    Hybrid tokenizer: Works in CV space BUT keeps whole words when found in corpus.
    
    Logic:
    - If word exists in corpus → keep as ONE token
    - Otherwise → decompose and use greedy algorithm
    
    Example:
    - "ይጫወታሉ" in corpus → ['ይጫወታሉ'] (1 token)
    - "unknown" not in corpus → ['un', 'kn', 'own'] (multiple tokens)
    """
    tokens: List[str] = field(default_factory=list)
    original_corpus_words: Set[str] = field(default_factory=set)
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    tokenizer_obj: Optional[HybridVocabTokenizer] = None
    decomposer: Optional[FidelDecomposer] = None
    
    def __post_init__(self):
        if self.decomposer is None:
            self.decomposer = FidelDecomposer()
    
    def train_from_corpus(self, corpus_path: str, use_new_algorithm: bool = True) -> None:
        """
        Train hybrid tokenizer.
        
        Strategy:
        1. Store all original corpus words
        2. Decompose corpus and build morpheme vocabulary
        3. During tokenization: check if word in corpus → keep whole
        """
        # Load corpus
        text = load_text_file(corpus_path)
        if text is None:
            raise FileNotFoundError(corpus_path)
        text = map_non_ethiopic_to_placeholders(text)
        corpus = simple_tokenize_keep_punct(text)
        
        print(f"Loaded corpus: {len(corpus):,} words")
        
        # Store original corpus words for whole-word matching
        self.original_corpus_words = set(corpus)
        print(f"  Stored {len(self.original_corpus_words):,} unique words from corpus")
        
        # Decompose corpus
        print("Decomposing corpus to CV space...")
        decomposed_corpus = self.decomposer.decompose_corpus(corpus)
        
        # Run greedy algorithm to extract morphemes
        print(f"Extracting morphemes...")
        prefixes, stems, suffixes = split_corpus_greedy_subword(decomposed_corpus)
        
        print(f"  Found {len(prefixes):,} prefixes")
        print(f"  Found {len(stems):,} stems")
        print(f"  Found {len(suffixes):,} suffixes")
        
        # Build vocabulary
        tokenset = set()
        tokenset.update(prefixes)
        tokenset.update(stems)
        tokenset.update(suffixes)
        
        # Add all complete decomposed words from corpus
        for decomposed_word in decomposed_corpus:
            tokenset.add(decomposed_word)
        
        # CRITICAL: Add morphological suffixes and prefixes in CV space
        print(f"  Adding {len(MORPHOLOGICAL_SUFFIXES)} morphological suffixes...")
        for suffix in MORPHOLOGICAL_SUFFIXES:
            tokenset.add(self.decomposer.decompose_word(suffix))
        
        print(f"  Adding {len(MORPHOLOGICAL_PREFIXES)} morphological prefixes...")
        for prefix in MORPHOLOGICAL_PREFIXES:
            tokenset.add(self.decomposer.decompose_word(prefix))
        
        # Add CV pairs as fallback
        for cv_pair in self.decomposer.fidel_to_cv.values():
            tokenset.add(cv_pair)
        
        # Add space
        tokenset.add(" ")
        tokenset.discard('')
        tokenset.discard(None)
        
        # Sort: longest first for greedy matching
        tokens = sorted(tokenset, key=lambda x: (-len(x), x))
        self.tokens = tokens
        
        print(f"Built vocabulary: {len(tokens):,} tokens")
        
        # Create tokenizer
        self.tokenizer_obj = HybridVocabTokenizer(
            tokens=tokens,
            original_corpus_words=self.original_corpus_words,
            decomposer=self.decomposer,
            pad_token=self.pad_token,
            unk_token=self.unk_token
        )
    
    def encode_with_spans(self, text: str, oov_strategy: str = 'unk') -> Tuple[List[int], List[Dict[str, Any]]]:
        """Encode with word-boundary awareness"""
        spans = self.tokenizer_obj.tokenize_with_spans(text, oov_strategy=oov_strategy)
        ids = [r['id'] for r in spans]
        return ids, spans
    
    def encode(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Encode to IDs"""
        return self.tokenizer_obj.tokenize(text, oov_strategy=oov_strategy)
    
    def decode(self, ids: List[int], spans: Optional[List[Dict[str, Any]]] = None, unk_placeholder: str = '<unk>') -> str:
        """Decode to original text"""
        if spans is not None:
            return ''.join(s['text'] for s in spans)
        return self.tokenizer_obj.detokenize(ids, unk_placeholder=unk_placeholder)
    
    def save_pretrained(self, outdir: str) -> None:
        """Save model"""
        os.makedirs(outdir, exist_ok=True)
        
        vocab = {t: i for t, i in self.tokenizer_obj.token2id.items()}
        with open(os.path.join(outdir, 'vocab.json'), 'w', encoding='utf-8') as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        
        cfg = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'tokens': sorted(self.tokens, key=lambda x: self.tokenizer_obj.token2id.get(x, 0)),
            'original_corpus_words': sorted(list(self.original_corpus_words)),
            'algorithm': 'hybrid',
            'fidel_map': AMHARIC_FIDEL_MAP
        }
        with open(os.path.join(outdir, 'tokenizer.json'), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        
        print(f"Saved hybrid tokenizer to {outdir}/")
    
    @classmethod
    def load_pretrained(cls, indir: str) -> 'HybridTokenizer':
        """Load model"""
        with open(os.path.join(indir, 'tokenizer.json'), 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        
        t = cls()
        t.pad_token = cfg.get('pad_token', '<pad>')
        t.unk_token = cfg.get('unk_token', '<unk>')
        t.tokens = cfg.get('tokens', [])
        t.original_corpus_words = set(cfg.get('original_corpus_words', []))
        
        fidel_map = cfg.get('fidel_map', AMHARIC_FIDEL_MAP)
        t.decomposer = FidelDecomposer(fidel_map)
        
        t.tokenizer_obj = HybridVocabTokenizer(
            tokens=t.tokens,
            original_corpus_words=t.original_corpus_words,
            decomposer=t.decomposer,
            pad_token=t.pad_token,
            unk_token=t.unk_token
        )
        
        return t


if __name__ == "__main__":
    import tempfile
    
    print("=" * 80)
    print("HYBRID TOKENIZER DEMO")
    print("=" * 80)
    
    # Create test corpus
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as fh:
        fh.write('ልጅ ልጆች ይጫወታሉ ሰው ሰዎች\nበልጅ የልጆች')
        corpus_file = fh.name
    
    # Train
    print("\n1. Training hybrid tokenizer...")
    tok = HybridTokenizer()
    tok.train_from_corpus(corpus_file)
    
    # Test
    print("\n2. Testing...")
    test_cases = [
        "ልጆች ይጫወታሉ",        # Should be: ['ልጆች', 'ይጫወታሉ']
        "ሰዎች",                # Should be: ['ሰዎች'] (whole word in corpus)
        "በልጅ",                # Should be: ['በልጅ'] (in corpus)
    ]
    
    for text in test_cases:
        ids, spans = tok.encode_with_spans(text)
        tokens = [s['token'] for s in spans]
        decoded = tok.decode(ids, spans=spans)
        
        print(f"\n  '{text}'")
        print(f"    Tokens: {tokens}")
        print(f"    Count: {len(tokens)} token(s)")
        print(f"    Decoded: '{decoded}'")
        print(f"    Match: {'✓' if decoded == text else '✗'}")
    
    os.unlink(corpus_file)
    print("\n" + "=" * 80)
