"""
Morphologically-Aware Decomposed Tokenizer

This tokenizer prioritizes morphological boundaries (prefix-stem-suffix)
over whole-word matches for better linguistic analysis.

Example: ሰዎች → ሰው + ኦች (person + plural)
"""

from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import json
import os

from .fidel_decomposer import FidelDecomposer, AMHARIC_FIDEL_MAP
from .automata import split_corpus_greedy_subword
from .utils import simple_tokenize_keep_punct, load_text_file, map_non_ethiopic_to_placeholders


# Common Amharic morphological patterns (ORDERED: longest first!)
COMMON_SUFFIXES = [
    "ዎቹን",  # plural definite + accusative
    "ዎቹ",   # plural definite
    "ኦች",   # plural: ልጆች, ሰዎች, etc. (PRIORITY!)
    "ዎች",   # plural variant
    "ዋን",   # definite + accusative
    "ቱን",   # definite masc + accusative
    "ቱ",    # definite masculine
    "ዋ",    # definite feminine / possessive
    "ች",    # feminine/plural
    "ን",    # accusative/definiteness
    "ም",    # topic marker / emphasis
    "ስ",    # question particle
    "ና",    # and
    "ሽ",    # 2nd person feminine
    "ሁ",    # 2nd person masculine
    "ው",    # possessive/3rd person
]

COMMON_PREFIXES = [
    "እስከ",  # until (longest first)
    "እንደ",  # like, as
    "ወደ",   # to, towards
    "ስለ",   # about
    "ያል",   # negation variant
    "አል",   # negation
    "የ",    # of, possessive
    "ለ",    # for, to
    "በ",    # by, in, at
    "ከ",    # from
    "ብ",    # by (short form)
]


class MorphologicalVocabTokenizer:
    """
    Vocabulary-based tokenizer that prioritizes morphological boundaries.
    """
    
    def __init__(self, tokens: Optional[List[str]] = None,
                 stems: Optional[Set[str]] = None,
                 prefixes: Optional[Set[str]] = None,
                 suffixes: Optional[Set[str]] = None,
                 decomposer: Optional[FidelDecomposer] = None,
                 pad_token: str = "<pad>", unk_token: str = "<unk>",
                 start_ids_from: int = 0):
        """
        Initialize with morpheme-aware vocabulary.
        """
        tokens = tokens or []
        self.decomposer = decomposer if decomposer is not None else FidelDecomposer()
        
        # Store morpheme sets for morphological analysis
        self.stems = stems or set()
        self.prefixes = prefixes or set()
        self.suffixes = suffixes or set()
        
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
        
        # Composite-id scheme
        self.COMPOSITE_FACTOR = 2
        max_base_id = max(self.token2id.values(), default=-1)
        self.COMPOSITE_OFFSET = (max_base_id + 1) * self.COMPOSITE_FACTOR + 1
    
    def tokenize_morphological(self, decomposed_text: str) -> List[str]:
        """
        Tokenize with morphological awareness.
        
        Priority:
        1. Check for prefix-stem-suffix patterns
        2. Try to split known suffixes
        3. Try to split known prefixes
        4. Fallback to greedy longest match
        """
        tokens = []
        i = 0
        text_len = len(decomposed_text)
        
        while i < text_len:
            # Skip whitespace
            if decomposed_text[i].isspace():
                tokens.append(decomposed_text[i])
                i += 1
                continue
            
            # Find word boundary
            word_start = i
            while i < text_len and not decomposed_text[i].isspace():
                i += 1
            
            word = decomposed_text[word_start:i]
            
            # Try morphological splitting
            morphemes = self._split_morphologically(word)
            tokens.extend(morphemes)
        
        return tokens
    
    def _split_morphologically(self, word: str) -> List[str]:
        """
        Split a word into morphemes using linguistic knowledge.
        
        Key improvement: Check for EXACT suffix matches first
        Example: "ስአውኦችእ" (ሰዎች) → "ስአውእ" (ሰው) + "ኦችእ" (ኦች)
        """
        if len(word) <= 2:
            return [word]
        
        # STEP 1: Try to find EXACT suffix match from known suffixes
        # This is critical for patterns like "ኦች" (plural)
        best_suffix = None
        best_stem = None
        
        # Check each known suffix (decomposed form)
        for suffix_orig in COMMON_SUFFIXES:
            suffix_decomp = self.decomposer.decompose_word(suffix_orig)
            
            if word.endswith(suffix_decomp):
                # Extract stem (what remains after removing suffix)
                stem_cand = word[:-len(suffix_decomp)]
                
                # Verify stem is valid (length >= 2 chars in CV space = 1+ chars in original)
                if len(stem_cand) >= 2:
                    best_suffix = suffix_decomp
                    best_stem = stem_cand
                    break  # Take first match (suffixes sorted by priority)
        
        # STEP 2: If exact suffix not found, try vocabulary-based suffix
        if not best_suffix:
            for suf_len in range(min(len(word) - 2, 10), 0, -1):
                suffix_cand = word[-suf_len:]
                
                # Only accept if it's a recognized suffix in vocab
                if suffix_cand in self.suffixes:
                    stem_cand = word[:-suf_len]
                    
                    # Verify stem
                    if len(stem_cand) >= 2 and (stem_cand in self.stems or stem_cand in self.token2id):
                        best_suffix = suffix_cand
                        best_stem = stem_cand
                        break
        
        # STEP 3: If we found suffix, now check for prefix in stem
        if best_suffix and best_stem:
            # Check for prefix in the stem
            best_prefix = None
            
            for prefix_orig in COMMON_PREFIXES:
                prefix_decomp = self.decomposer.decompose_word(prefix_orig)
                
                if best_stem.startswith(prefix_decomp):
                    # Extract remaining stem
                    remaining_stem = best_stem[len(prefix_decomp):]
                    
                    # Verify remaining stem is valid
                    if len(remaining_stem) >= 2:
                        best_prefix = prefix_decomp
                        best_stem = remaining_stem
                        break
            
            # Return the split
            if best_prefix:
                return [best_prefix, best_stem, best_suffix]
            else:
                return [best_stem, best_suffix]
        
        # STEP 4: No suffix found, try prefix only
        for prefix_orig in COMMON_PREFIXES:
            prefix_decomp = self.decomposer.decompose_word(prefix_orig)
            
            if word.startswith(prefix_decomp):
                stem = word[len(prefix_decomp):]
                if len(stem) >= 2 and (stem in self.stems or stem in self.token2id):
                    return [prefix_decomp, stem]
        
        # STEP 5: Check if whole word is a known stem
        if word in self.stems:
            return [word]
        
        # STEP 6: Fallback to greedy matching
        return self._greedy_match(word)
    
    def _greedy_match(self, word: str) -> List[str]:
        """Greedy longest-match tokenization"""
        tokens = []
        i = 0
        
        while i < len(word):
            longest_match = None
            longest_len = 0
            
            for length in range(min(self.max_token_len, len(word) - i), 0, -1):
                candidate = word[i:i+length]
                if candidate in self.token2id:
                    longest_match = candidate
                    longest_len = length
                    break
            
            if longest_match:
                tokens.append(longest_match)
                i += longest_len
            else:
                # Fallback to single character
                tokens.append(word[i])
                i += 1
        
        return tokens
    
    def tokenize_with_spans(self, text: str, oov_strategy: str = "unk") -> List[Dict[str, Any]]:
        """
        Tokenize with morphological awareness and span tracking.
        """
        # Decompose input
        decomposed_text = self.decomposer.decompose_word(text)
        
        # Tokenize morphologically
        decomposed_tokens = self.tokenize_morphological(decomposed_text)
        
        # Build spans
        records = []
        decomposed_pos = 0
        original_pos = 0
        
        for decomposed_token in decomposed_tokens:
            # Find position in decomposed text
            start_decomp = decomposed_pos
            end_decomp = decomposed_pos + len(decomposed_token)
            
            # Recompose to original
            original_token = self.decomposer.recompose_word(decomposed_token)
            
            # Get token ID
            base_tid = self.token2id.get(decomposed_token, self.token2id.get(self.unk_token, 0))
            comp_id = self.COMPOSITE_OFFSET + base_tid * self.COMPOSITE_FACTOR
            
            records.append({
                "id": comp_id,
                "token": original_token,
                "decomposed_token": decomposed_token,
                "start": original_pos,
                "end": original_pos + len(original_token),
                "text": original_token
            })
            
            decomposed_pos = end_decomp
            original_pos += len(original_token)
        
        return records
    
    def tokenize(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Tokenize text and return only IDs"""
        spans = self.tokenize_with_spans(text, oov_strategy=oov_strategy)
        return [r['id'] for r in spans]
    
    def detokenize(self, ids: List[int], unk_placeholder: Optional[str] = None) -> str:
        """
        Detokenize IDs back to original text with morpheme boundary merging.
        
        Special handling: ው + ኦ → ዎ at morpheme boundaries
        """
        if unk_placeholder is None:
            unk_placeholder = self.unk_token if self.unk_token is not None else "<unk>"
        
        decomposed_parts = []
        id2t = self.id2token
        
        for i in ids:
            if isinstance(i, int) and i >= self.COMPOSITE_OFFSET:
                rel = i - self.COMPOSITE_OFFSET
                base_id = rel // self.COMPOSITE_FACTOR
                tok = id2t.get(base_id, unk_placeholder)
            else:
                tok = id2t.get(i, unk_placeholder)
            
            decomposed_parts.append(tok)
        
        # Join decomposed parts with morpheme boundary handling
        decomposed_text = self._merge_morpheme_boundaries(decomposed_parts)
        original_text = self.decomposer.recompose_word(decomposed_text)
        
        return original_text
    
    def _merge_morpheme_boundaries(self, parts: List[str]) -> str:
        """
        Merge decomposed tokens handling morpheme boundaries.
        
        Special morphophonemic rules:
        - ው + ኦ → ዎ (ውእ + ኦችእ → ውኦችእ)
        - ር + ኦ → ሮ (ርእ + ኦችእ → ርኦችእ)
        - Any consonant + vowel suffix merging
        """
        if not parts:
            return ""
        
        if len(parts) == 1:
            return parts[0]
        
        result = [parts[0]]
        
        for i in range(1, len(parts)):
            part = parts[i]
            
            # Check if current part is a vowel-starting suffix (ኦች, etc.)
            if part.startswith("ኦ") and result:
                prev = result[-1]
                
                # If previous part ends with consonant + እ pattern (like ውእ, ርእ, etc.)
                # Remove the እ and merge with ኦ to form vowel fusion
                if len(prev) >= 2 and prev.endswith("እ"):
                    # Check if character before እ is a consonant
                    consonant_part = prev[:-1]  # Remove እ
                    
                    # Merge: remove እ, add vowel suffix directly
                    result[-1] = consonant_part + part
                else:
                    result.append(part)
            else:
                result.append(part)
        
        return ''.join(result)


@dataclass
class MorphologicalTokenizer:
    """
    Morphologically-aware tokenizer that splits words into meaningful components.
    
    Example: ሰዎች → ሰው + ኦች (person + plural)
    """
    tokens: List[str] = field(default_factory=list)
    stems: Set[str] = field(default_factory=set)
    prefixes: Set[str] = field(default_factory=set)
    suffixes: Set[str] = field(default_factory=set)
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    tokenizer_obj: Optional[MorphologicalVocabTokenizer] = None
    decomposer: Optional[FidelDecomposer] = None
    
    def __post_init__(self):
        if self.decomposer is None:
            self.decomposer = FidelDecomposer()
    
    def train_from_corpus(self, corpus_path: str, use_new_algorithm: bool = True) -> None:
        """
        Train with morphological awareness.
        
        Unlike the simple decomposed tokenizer, this one:
        1. Identifies stems, prefixes, suffixes separately
        2. Does NOT add whole words to vocabulary
        3. Prioritizes morphological splitting
        """
        # Load and preprocess corpus
        text = load_text_file(corpus_path)
        if text is None:
            raise FileNotFoundError(corpus_path)
        text = map_non_ethiopic_to_placeholders(text)
        corpus = simple_tokenize_keep_punct(text)
        
        print(f"Loaded corpus: {len(corpus):,} words")
        
        # Decompose entire corpus
        print("Decomposing corpus to CV space...")
        decomposed_corpus = self.decomposer.decompose_corpus(corpus)
        
        # Run morphological analysis on decomposed corpus
        print(f"Running morphological analysis...")
        prefixes, stems, suffixes = split_corpus_greedy_subword(decomposed_corpus)
        
        # Store morpheme sets
        self.prefixes = prefixes
        self.stems = stems
        self.suffixes = suffixes
        
        print(f"  Found {len(prefixes):,} prefixes")
        print(f"  Found {len(stems):,} stems")
        print(f"  Found {len(suffixes):,} suffixes")
        
        # Build vocabulary from morphemes ONLY (no whole words)
        tokenset = set()
        tokenset.update(prefixes)
        tokenset.update(stems)
        tokenset.update(suffixes)
        
        # Add common morphemes
        for suffix in COMMON_SUFFIXES:
            decomposed_suffix = self.decomposer.decompose_word(suffix)
            tokenset.add(decomposed_suffix)
            self.suffixes.add(decomposed_suffix)
        
        for prefix in COMMON_PREFIXES:
            decomposed_prefix = self.decomposer.decompose_word(prefix)
            tokenset.add(decomposed_prefix)
            self.prefixes.add(decomposed_prefix)
        
        # Add all CV pairs as fallback
        for cv_pair in self.decomposer.fidel_to_cv.values():
            tokenset.add(cv_pair)
        
        # Add space
        tokenset.add(" ")
        
        # Remove empty
        tokenset.discard('')
        tokenset.discard(None)
        
        # Sort: suffixes/prefixes first (shorter), then stems (longer)
        # This encourages morphological splitting
        def sort_key(x):
            x_original = self.decomposer.recompose_word(x)
            
            # Prioritize known morphemes
            if x in self.suffixes or x_original in COMMON_SUFFIXES:
                return (0, -len(x), x)  # Suffixes first
            elif x in self.prefixes or x_original in COMMON_PREFIXES:
                return (1, -len(x), x)  # Prefixes second
            elif x in self.stems:
                return (2, -len(x), x)  # Stems third
            else:
                return (3, -len(x), x)  # Others last
        
        tokens = sorted(tokenset, key=sort_key)
        self.tokens = tokens
        
        print(f"Built vocabulary: {len(tokens):,} tokens (morpheme-focused)")
        
        # Create tokenizer object
        self.tokenizer_obj = MorphologicalVocabTokenizer(
            tokens=tokens,
            stems=self.stems,
            prefixes=self.prefixes,
            suffixes=self.suffixes,
            decomposer=self.decomposer,
            pad_token=self.pad_token,
            unk_token=self.unk_token
        )
    
    def save_pretrained(self, outdir: str) -> None:
        """Save model to directory"""
        os.makedirs(outdir, exist_ok=True)
        
        # Save vocabulary
        vocab = {t: i for t, i in self.tokenizer_obj.token2id.items()}
        with open(os.path.join(outdir, 'vocab.json'), 'w', encoding='utf-8') as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        
        # Save config with morpheme sets
        cfg = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'tokens': sorted(self.tokens, key=lambda x: self.tokenizer_obj.token2id.get(x, 0)),
            'stems': sorted(list(self.stems)),
            'prefixes': sorted(list(self.prefixes)),
            'suffixes': sorted(list(self.suffixes)),
            'algorithm': 'morphological',
            'fidel_map': AMHARIC_FIDEL_MAP
        }
        with open(os.path.join(outdir, 'tokenizer.json'), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
        
        print(f"Saved morphological tokenizer to {outdir}/")
    
    @classmethod
    def load_pretrained(cls, indir: str) -> 'MorphologicalTokenizer':
        """Load model from directory"""
        with open(os.path.join(indir, 'tokenizer.json'), 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        
        t = cls()
        t.pad_token = cfg.get('pad_token', '<pad>')
        t.unk_token = cfg.get('unk_token', '<unk>')
        t.tokens = cfg.get('tokens', [])
        t.stems = set(cfg.get('stems', []))
        t.prefixes = set(cfg.get('prefixes', []))
        t.suffixes = set(cfg.get('suffixes', []))
        
        # Load fidel map
        fidel_map = cfg.get('fidel_map', AMHARIC_FIDEL_MAP)
        t.decomposer = FidelDecomposer(fidel_map)
        
        # Create tokenizer object
        t.tokenizer_obj = MorphologicalVocabTokenizer(
            tokens=t.tokens,
            stems=t.stems,
            prefixes=t.prefixes,
            suffixes=t.suffixes,
            decomposer=t.decomposer,
            pad_token=t.pad_token,
            unk_token=t.unk_token
        )
        
        return t
    
    def encode_with_spans(self, text: str, oov_strategy: str = 'unk') -> Tuple[List[int], List[Dict[str, Any]]]:
        """Encode with morphological awareness"""
        spans = self.tokenizer_obj.tokenize_with_spans(text, oov_strategy=oov_strategy)
        ids = [r['id'] for r in spans]
        return ids, spans
    
    def encode(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        """Encode text to IDs"""
        return self.tokenizer_obj.tokenize(text, oov_strategy=oov_strategy)
    
    def decode(self, ids: List[int], spans: Optional[List[Dict[str, Any]]] = None, unk_placeholder: str = '<unk>') -> str:
        """Decode IDs back to original text"""
        if spans is not None:
            return ''.join(s['text'] for s in spans)
        return self.tokenizer_obj.detokenize(ids, unk_placeholder=unk_placeholder)


if __name__ == "__main__":
    # Demo
    print("=" * 80)
    print("MORPHOLOGICAL TOKENIZER DEMO")
    print("=" * 80)
    
    import tempfile
    with tempfile.NamedTemporaryFile('w+', encoding='utf-8', delete=False) as fh:
        fh.write('ሰው ሰዎች ልጅ ልጆች መጽሐፍ መጻሕፍት\nበሰው የሰው ለሰው ከሰው')
        corpus_file = fh.name
    
    # Train
    print("\n1. Training...")
    tok = MorphologicalTokenizer()
    tok.train_from_corpus(corpus_file)
    
    # Test
    print("\n2. Testing morphological splitting...")
    test_cases = [
        ("ሰዎች", "ሰው + ኦች (person + plural)"),
        ("ልጆች", "ልጅ + ኦች (child + plural)"),
        ("በሰው", "በ + ሰው (by + person)"),
        ("የልጆች", "የ + ልጅ + ኦች (of + child + plural)")
    ]
    
    for text, expected in test_cases:
        ids, spans = tok.encode_with_spans(text)
        tokens = [s['token'] for s in spans]
        decoded = tok.decode(ids, spans=spans)
        
        print(f"\n  '{text}' ({expected})")
        print(f"    Tokens: {tokens}")
        print(f"    Decoded: '{decoded}'")
        print(f"    Match: {'✓' if decoded == text else '✗'}")
    
    os.unlink(corpus_file)
    print("\n" + "=" * 80)
