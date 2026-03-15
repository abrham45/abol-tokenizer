from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import json, os

from .utils import simple_tokenize_keep_punct, load_text_file, map_non_ethiopic_to_placeholders
from .automata import split_corpus_ac, split_corpus_greedy_subword

class VocabTokenizer:
    def __init__(self, tokens: Optional[List[str]] = None, pad_token: str = "<pad>", unk_token: str = "<unk>", start_ids_from: int = 0):
        tokens = tokens or []
        # Ensure the special tokens are present (preserve order deterministically)
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
        # Use a disjoint id range above base ids:
        # composite_id = COMPOSITE_OFFSET + base_id * COMPOSITE_FACTOR + leading_space_flag
        # where leading_space_flag in {0,1}
        self.COMPOSITE_FACTOR = 2
        max_base_id = max(self.token2id.values(), default=-1)
        # Keep offset strictly above any base id to avoid overlap
        self.COMPOSITE_OFFSET = (max_base_id + 1) * self.COMPOSITE_FACTOR + 1

    def tokenize_with_spans(self, text: str, oov_strategy: str = "unk") -> List[Dict[str, Any]]:
        cp = list(text)
        n = len(cp)
        i = 0
        records = []
        while i < n:
            # Fold any leading whitespace(s) into a binary flag for the next token
            had_leading_space = False
            start_pos = i
            while i < n and cp[i].isspace():
                had_leading_space = True
                i += 1
            if i >= n:
                break

            longest_tok = None
            longest_len = 0
            # Greedy longest-match using vocabulary
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
                text_piece = (" " if had_leading_space else "") + longest_tok
                records.append({
                    "id": comp_id,
                    "token": longest_tok,
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
                records.append({"id": comp_id, "token": self.unk_token, "start": start_pos, "end": i+1, "text": text_piece})
                i += 1
            else:
                # char fallback
                if ch in self.token2id:
                    base_tid = self.token2id[ch]
                else:
                    base_tid = self.token2id[self.unk_token]
                comp_id = self.COMPOSITE_OFFSET + base_tid * self.COMPOSITE_FACTOR + (1 if had_leading_space else 0)
                text_piece = (" " if had_leading_space else "") + ch
                records.append({"id": comp_id, "token": ch if ch in self.token2id else self.unk_token, "start": start_pos, "end": i+1, "text": text_piece})
                i += 1
        return records

    def detokenize_from_spans(self, spans: List[Dict[str, Any]]) -> str:
        # spans contain reconstructed substrings (original text slices) -> join them
        return ''.join(rec['text'] for rec in spans)

    def tokenize(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        spans = self.tokenize_with_spans(text, oov_strategy=oov_strategy)
        return [r['id'] for r in spans]

    def detokenize(self, ids: List[int], unk_placeholder: Optional[str] = None) -> str:
        """
        Detokenize a list of ids into a string.

        If `unk_placeholder` is None, the method returns the literal token string
        from the vocabulary (which for the unk token is often '<unk>'). If you want
        unknown ids to be rendered as an empty string or as some other placeholder,
        pass unk_placeholder (e.g. unk_placeholder=' ' or unk_placeholder='').
        """
        if unk_placeholder is None:
            # default to the literal unk token string (preserves original behavior)
            unk_placeholder = self.unk_token if self.unk_token is not None else "<unk>"

        parts = []
        id2t = self.id2token
        for i in ids:
            # Detect composite leading-space ids
            if isinstance(i, int) and i >= self.COMPOSITE_OFFSET:
                rel = i - self.COMPOSITE_OFFSET
                base_id = rel // self.COMPOSITE_FACTOR
                flag = rel % self.COMPOSITE_FACTOR
                tok = id2t.get(base_id)
                if tok is None:
                    parts.append(unk_placeholder)
                else:
                    # If base token represents unk, use the placeholder
                    piece = (" " if flag == 1 else "") + (unk_placeholder if (self.unk_token is not None and tok == self.unk_token) else tok)
                    parts.append(piece)
            else:
                tok = id2t.get(i)
                if tok is None:
                    parts.append(unk_placeholder)
                else:
                    if self.unk_token is not None and tok == self.unk_token:
                        parts.append(unk_placeholder)
                    else:
                        parts.append(tok)
        return ''.join(parts)


@dataclass
class Tokenizer:
    tokens: List[str] = field(default_factory=list)
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'
    tokenizer_obj: Optional[VocabTokenizer] = None

    def extract_custom_tokens(self, corpus: List[str]) -> set:
        """
        Extract all Amharic letters (glyphs) and punctuation/symbols from Unicode ranges.
        
        Args:
            corpus: List of words from the training corpus (not used, but kept for interface compatibility)
            
        Returns:
            Set of all Amharic characters and punctuation/symbols from Unicode ranges
        """
        custom_tokens = set()
        
        # Add all characters from the complete Ethiopic Unicode ranges
        # Ethiopic block (0x1200-0x137F) - Main Amharic characters
        for code_point in range(0x1200, 0x1380):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Ethiopic Supplement (0x1380-0x139F) - Additional characters
        for code_point in range(0x1380, 0x13A0):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Ethiopic Extended (0x2D80-0x2DDF) - Extended characters
        for code_point in range(0x2D80, 0x2DE0):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Add common punctuation marks and symbols
        punctuation_and_symbols = [
            # Basic punctuation
            '.', ',', ';', ':', '!', '?', '"', "'", '(', ')', '[', ']', '{', '}',
            # Mathematical symbols
            '+', '-', '*', '/', '=', '<', '>', '%', '&', '|', '^', '~', '`',
            # Currency and other symbols
            '$', '€', '£', '¥', '¢', '@', '#', '§', '©', '®', '™',
            # Quotation marks and dashes
            '"', '"', ''', ''', '–', '—', '…',
            # Other common symbols
            '°', '±', '×', '÷', '√', '∞', '∑', '∏', '∫', '∆', '∇',
            # Arrows
            '←', '→', '↑', '↓', '↔', '↕', '↖', '↗', '↘', '↙',
            # Greek letters (common in math/scientific text)
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
            # Numbers (0-9)
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Additional Unicode punctuation ranges
        ]
        
        # Add punctuation and symbols
        custom_tokens.update(punctuation_and_symbols)
        
        # Add characters from Unicode punctuation ranges
        # General Punctuation (U+2000-U+206F)
        for code_point in range(0x2000, 0x2070):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Supplemental Punctuation (U+2E00-U+2E7F)
        for code_point in range(0x2E00, 0x2E80):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Currency Symbols (U+20A0-U+20CF)
        for code_point in range(0x20A0, 0x20D0):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Mathematical Operators (U+2200-U+22FF)
        for code_point in range(0x2200, 0x2300):
            char = chr(code_point)
            custom_tokens.add(char)
        
        # Miscellaneous Symbols (U+2600-U+26FF)
        for code_point in range(0x2600, 0x2700):
            char = chr(code_point)
            custom_tokens.add(char)
        
        return custom_tokens

    def train_from_corpus(self, corpus_path: str, build_maps: bool = True, use_new_algorithm: bool = True) -> None:
        """
        Train tokenizer from a plain-text corpus file.

        Key change: ensure the SPACE character ' ' is explicitly included in the
        vocabulary so that encoding/decoding preserves spaces when using id-only decode.
        
        Args:
            corpus_path: Path to the training corpus file
            build_maps: Whether to build character maps (for compatibility)
            use_new_algorithm: If True, use greedy meaningful subword detection.
                              If False, use original Suffix Automaton approach.
        """
        text = load_text_file(corpus_path)
        if text is None:
            raise FileNotFoundError(corpus_path)
        text = map_non_ethiopic_to_placeholders(text)
        corpus = simple_tokenize_keep_punct(text)
        
        # Choose algorithm based on flag
        if use_new_algorithm:
            prefixes, stems, suffixes = split_corpus_greedy_subword(corpus)
        else:
            prefixes, stems, suffixes = split_corpus_ac(corpus)
        
        # Extract custom tokens
        custom_tokens = self.extract_custom_tokens(corpus)

        tokenset = set()
        tokenset.update(prefixes)
        tokenset.update(stems)
        tokenset.update(suffixes)
        tokenset.update(custom_tokens)  # Add the new custom tokens
        #tokenset.update(corpus)

        # IMPORTANT: ensure space (and optionally newline) are present as tokens
        tokenset.add(" ")
        # If you want to preserve explicit newlines as tokens, uncomment:
        # tokenset.add("\n")

        # Remove empty strings and ensure uniqueness
        tokenset.discard('')  # Remove empty strings
        tokenset.discard(None)  # Remove None values
        
        # deterministic ordering: prefer longer tokens first (greedy matching)
        tokens = sorted(tokenset, key=lambda x: (-len(x), x))
        self.tokens = tokens
        self.tokenizer_obj = VocabTokenizer(tokens=tokens, pad_token=self.pad_token, unk_token=self.unk_token)

    def merge_additional_corpus(self, additional_corpus_path: str, use_new_algorithm: bool = True) -> None:
        """
        Merge additional corpus with existing tokenizer vocabulary.
        
        This method:
        1. Loads and preprocesses the additional corpus
        2. Extracts morphological components (prefixes, stems, suffixes)
        3. Merges them with existing tokens
        4. Rebuilds the tokenizer with the combined vocabulary
        
        Args:
            additional_corpus_path: Path to the additional corpus file
            use_new_algorithm: If True, use greedy meaningful subword detection.
                              If False, use original Suffix Automaton approach.
        """
        # Load and preprocess the additional corpus
        text = load_text_file(additional_corpus_path)
        if text is None:
            raise FileNotFoundError(additional_corpus_path)
        text = map_non_ethiopic_to_placeholders(text)
        additional_corpus = simple_tokenize_keep_punct(text)
        
        # Extract morphological components from additional corpus
        if use_new_algorithm:
            additional_prefixes, additional_stems, additional_suffixes = split_corpus_greedy_subword(additional_corpus)
        else:
            additional_prefixes, additional_stems, additional_suffixes = split_corpus_ac(additional_corpus)
        
        # Get current tokens as a set
        current_tokens = set(self.tokens) if self.tokens else set()
        
        # Merge with additional morphological components
        current_tokens.update(additional_prefixes)
        current_tokens.update(additional_stems)
        current_tokens.update(additional_suffixes)
        
        # Ensure space is still present
        current_tokens.add(" ")
        
        # Remove empty strings and ensure uniqueness
        current_tokens.discard('')  # Remove empty strings
        current_tokens.discard(None)  # Remove None values
        
        # Rebuild tokenizer with merged vocabulary
        tokens = sorted(current_tokens, key=lambda x: (-len(x), x))
        self.tokens = tokens
        self.tokenizer_obj = VocabTokenizer(tokens=tokens, pad_token=self.pad_token, unk_token=self.unk_token)
        
        print(f"Merged additional corpus. Total tokens: {len(tokens)}")
        print(f"Added {len(additional_prefixes)} prefixes, {len(additional_stems)} stems, {len(additional_suffixes)} suffixes")

    def train_from_multiple_corpora(self, corpus_paths: List[str], build_maps: bool = True, use_new_algorithm: bool = True) -> None:
        """
        Train tokenizer from multiple corpus files.
        
        This method:
        1. Trains on the first corpus (base corpus)
        2. Merges additional corpora one by one
        
        Args:
            corpus_paths: List of paths to corpus files
            build_maps: Whether to build character maps (for compatibility)
            use_new_algorithm: If True, use greedy meaningful subword detection.
                              If False, use original Suffix Automaton approach.
        """
        if not corpus_paths:
            raise ValueError("At least one corpus path must be provided")
        
        # Train on the first corpus (base)
        print(f"Training on base corpus: {corpus_paths[0]}")
        self.train_from_corpus(corpus_paths[0], build_maps, use_new_algorithm)
        
        # Merge additional corpora
        for i, corpus_path in enumerate(corpus_paths[1:], 1):
            print(f"Merging additional corpus {i}: {corpus_path}")
            self.merge_additional_corpus(corpus_path, use_new_algorithm)

    def save_pretrained(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        vocab = {t: i for t, i in self.tokenizer_obj.token2id.items()}
        with open(os.path.join(outdir, 'vocab.json'), 'w', encoding='utf-8') as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        cfg = {
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'tokens': sorted(self.tokens, key=lambda x: self.tokenizer_obj.token2id.get(x, 0))
        }
        with open(os.path.join(outdir, 'tokenizer.json'), 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load_pretrained(cls, indir: str) -> 'Tokenizer':
        with open(os.path.join(indir, 'tokenizer.json'), 'r', encoding='utf-8') as fh:
            cfg = json.load(fh)
        t = cls()
        t.pad_token = cfg.get('pad_token', '<pad>')
        t.unk_token = cfg.get('unk_token', '<unk>')
        t.tokens = cfg.get('tokens', [])
        t.tokenizer_obj = VocabTokenizer(tokens=t.tokens, pad_token=t.pad_token, unk_token=t.unk_token)
        return t

    def encode_with_spans(self, text: str, oov_strategy: str = 'unk') -> Tuple[List[int], List[Dict[str, Any]]]:
        spans = self.tokenizer_obj.tokenize_with_spans(text, oov_strategy=oov_strategy)
        ids = [r['id'] for r in spans]
        return ids, spans

    def encode(self, text: str, oov_strategy: str = 'unk') -> List[int]:
        return self.tokenizer_obj.tokenize(text, oov_strategy=oov_strategy)

    def decode(self, ids: List[int], spans: Optional[List[Dict[str, Any]]] = None, unk_placeholder: str = '<unk>') -> str:
        # If spans are present we can reconstruct exact original substrings
        if spans is not None:
            return self.tokenizer_obj.detokenize_from_spans(spans)
        # else, detokenize using id2token; supply an unk_placeholder so that
        # literal "<unk>" tokens aren't inserted when id corresponds to unk token.
        return self.tokenizer_obj.detokenize(ids, unk_placeholder=unk_placeholder)