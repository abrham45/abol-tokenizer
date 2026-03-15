# Abol Tokenizer — Algorithm Documentation

This document describes the two tokenization algorithms available in the Abol Tokenizer project:

1. **Abol-GMS** — Greedy Meaningful Subword tokenizer (the original algorithm)
2. **Abol-Decomposed** — Fidel Decomposition + Morphological tokenizer (the new algorithm)

---

## Background: Amharic Script

Amharic is written in the **Ethiopic script (Fidel)**, where each character is a syllable representing a **consonant + vowel** pair. For example:

| Fidel | Consonant | Vowel | Pronunciation |
|-------|-----------|-------|---------------|
| ሰ     | ስ         | አ     | /sə/          |
| ላ     | ል         | ኣ     | /la/          |
| ም     | ምእ        | (neutral) | /m/       |

This means "ሰላም" (peace) is not three letters but three **syllables**, each encoding phonological information. Standard tokenizers that treat characters as atomic units lose this structure.

---

## Algorithm 1: Abol-GMS

### Overview

Abol-GMS (Greedy Meaningful Subword) tokenizes Amharic text by identifying **meaningful subwords** directly in the original Fidel script. It finds substrings that appear as standalone words in the training corpus and uses those as tokens, relying on the principle that if a substring is a real word in the corpus, it is semantically meaningful.

### Training

1. **Load corpus** — A plain-text Amharic file is read and tokenized into words.
2. **Build a corpus word set** — All unique words are stored in a hash set for O(1) lookup.
3. **Greedy subword extraction** — For every word in the corpus, the algorithm scans position by position and finds the **longest substring that exists as a standalone word** in the corpus:
   - If the current substring is in the corpus but extending it by one more character is not → take it as a token.
   - If both are in the corpus → keep extending.
   - If neither is in the corpus → fall back to the single character.
4. **Morpheme classification** — Each extracted token is classified as a prefix, stem, or suffix based on its position in the word.
5. **Character fallback** — All Ethiopic Unicode characters (U+1200–U+137F, U+1380–U+139F, U+2D80–U+2DDF) plus punctuation and symbols are added individually as a guaranteed fallback so no input character is ever truly OOV.
6. **Vocabulary build** — All extracted tokens and characters are sorted longest-first (greedy matching preference) and assigned integer IDs.

### Encoding

Encoding uses a **greedy longest-match** strategy over the sorted vocabulary:

```
Input: "ሰዎች"
→ Scan: try length 3 → "ሰዎች" in vocab? Yes → token
Output: ["ሰዎች"]  (ID: 12847)
```

Each token ID uses a **composite ID scheme** that encodes whether a leading space preceded the token, enabling lossless reconstruction without storing the whitespace as a separate token.

### Decoding

Decoding reverses the composite ID scheme: extract the base token, prepend a space if the leading-space flag is set, and concatenate.

### Example

```
Input:   "ልጅ ይጫወታሉ"
Tokens:  ["ልጅ", " ", "ይጫወታሉ"]
IDs:     [4823, 1, 9217]
Decoded: "ልጅ ይጫወታሉ"
```

### Properties

| Property | Value |
|---|---|
| Vocabulary size | ~13,000–15,000 tokens |
| Granularity | Word-level and subword-level |
| OOV handling | Character-level fallback (all Ethiopic chars covered) |
| Space encoding | Composite ID flag (lossless) |
| Script awareness | Direct Fidel matching only |

### Underlying Data Structures

- **SuffixAutomaton** — Used to check whether a string is a substring of any corpus word in O(n) time during the original split_corpus_ac pass.
- **Aho-Corasick** — Multi-pattern matching automaton used to find all known prefix/suffix patterns simultaneously when splitting a new text piece.
- **TokenIndex** — Wrapper around Aho-Corasick that lazily rebuilds the automaton when the vocabulary changes.

---

## Algorithm 2: Abol-Decomposed

### Overview

Abol-Decomposed is a two-layer algorithm. It first **decomposes each Fidel syllable into its constituent consonant+vowel (CV) parts**, then tokenizes in that decomposed CV space, and finally **recomposes** the output back to readable Amharic Fidel. Crucially, it also applies **morphological splitting** to identify stems and suffixes (e.g. ልጆች → ልጅ + ኦች) and **whole-word preservation** to keep corpus words intact as single tokens (e.g. ይጫወታሉ → one token).

### Fidel Decomposition

The `FidelDecomposer` uses a lookup table (`AMHARIC_FIDEL_MAP`) mapping every Fidel character to its CV representation:

```
ሰ → ስአ    (consonant ስ + vowel አ)
ዎ → ውኦ    (consonant ው + vowel ኦ)
ጆ → ጅኦ    (consonant ጅ + vowel ኦ)
```

The neutral/6th-order form (e.g. ም, ስ, ል) maps to consonant + **እ** (neutral vowel marker):

```
ም → ምእ
ስ → ስእ
ል → ልእ
```

Full word decomposition example:

```
ሰላም  →  ስአ + ልኣ + ምእ  =  "ስአልኣምእ"
ልጆች  →  ልእ + ጅኦ + ችእ  =  "ልእጅኦችእ"
```

Recomposition reverses this by consuming 1, 2, or 3 CV characters at a time and looking them up in the reverse map.

### Training

1. **Load corpus** and store all unique original words in a set (`original_corpus_words`).
2. **Decompose the entire corpus** to CV space.
3. **Run greedy subword extraction** on the decomposed corpus to find meaningful CV-level subwords (prefixes, stems, suffixes).
4. **Add all complete decomposed words** from the corpus to the vocabulary — this ensures full words are always tokenizable as single units.
5. **Add all morphological suffixes and prefixes** (see table below) in their decomposed forms to ensure they are always in vocabulary.
6. **Add all individual CV pairs** from the Fidel map as a character-level fallback.
7. **Sort and build vocabulary** — tokens sorted longest-first.

### Morphological Patterns

The algorithm has built-in knowledge of common Amharic morphological markers:

**Suffixes** (ordered longest-first to ensure greedy priority):

| Suffix | Meaning |
|--------|---------|
| ዎቹን   | plural + definite + accusative |
| ዎቹ    | plural + definite |
| ኦች    | plural |
| ዎች    | plural variant |
| ዋን    | definite + accusative |
| ቱን    | definite masculine + accusative |
| ቱ     | definite masculine |
| ዋ     | definite feminine |
| ች     | feminine / light plural |
| ን     | accusative |

**Prefixes** (ordered longest-first):

| Prefix | Meaning |
|--------|---------|
| እንደ   | like, as |
| እስከ   | until |
| ወደ    | to, towards |
| ስለ    | about |
| የ     | of, possessive |
| ለ     | for, to |
| በ     | by, in, at |
| ከ     | from |

### Encoding: Smart Word Splitting

For every word in the input, `_split_word_smart` applies the following priority logic:

```
STEP 1 — Morphological suffix check (in CV space):
  Decompose the word. Check if the decomposed form ends
  with the decomposed form of any known suffix (longest first).
  If yes AND the remaining stem has ≥ 1 character → SPLIT.

  Example:
    "ልጆች" → CV: "ልእጅኦችእ"
    Suffix "ኦች" → CV: "ኦችእ"
    "ልእጅኦችእ" ends with "ኦችእ" → YES
    Stem CV: "ልእጅእ" → recompose → "ልጅ"
    Result: ["ልጅ", "ኦች"]

STEP 2 — Whole-word dictionary check:
  If the word exists in original_corpus_words → ONE token.

  Example:
    "ይጫወታሉ" in corpus → ["ይጫወታሁ"]

STEP 3 — Morphological prefix check (in CV space):
  Check if decomposed form starts with the decomposed
  form of any known prefix. If yes → SPLIT and recurse
  on the stem.

  Example:
    "ለልጆች" → prefix "ለ" (CV: "ልአ") found
    Stem: "ልጆች" → recurse → ["ልጅ", "ኦች"]
    Result: ["ለ", "ልጅ", "ኦች"]

STEP 4 — Greedy CV-space tokenization:
  If none of the above matched, decompose the word and
  run greedy longest-match over the CV vocabulary.
  Recompose each CV token back to Fidel for output.
```

### Decoding: Morphophonemic Boundary Merging

When decoding token IDs back to text, adjacent morphemes may need to be fused back together due to **morphophonemic alternation** — the process by which sounds change at morpheme boundaries.

The key rule is: **if a CV token ends with `እ` (neutral vowel) and the next CV token starts with a vowel, remove the `እ` and concatenate**.

This reconstructs the original fused syllable:

```
ልእጅእ  +  ኦችእ
  ends with እ, next starts with ኦ
→ merge: ልእጅ + ኦችእ = ልእጅኦችእ
→ recompose: ልጆች  ✓

ስአውእ  +  ኦችእ
  ends with እ, next starts with ኦ
→ merge: ስአው + ኦችእ = ስአውኦችእ
→ recompose: ሰዎች  ✓
```

### Full Example

```
Input:   "ልጆች ይጫወታሉ"

Tokenization:
  Word "ልጆች":
    CV: ልእጅኦችእ
    Suffix "ኦች" (CV: ኦችእ) found at end
    Stem CV: ልእጅእ → recompose → "ልጅ"
    → tokens: ["ልጅ", "ኦች"]

  Space: [" "]

  Word "ይጫወታሉ":
    In corpus_words → keep whole
    → tokens: ["ይጫወታሉ"]

Output tokens: ["ልጅ", "ኦች", " ", "ይጫወታሉ"]
IDs:           [233369, 237459, 237791, 141962]

Decoding:
  CV parts: ["ልእጅእ", "ኦችእ", " ", "ይእጭኣውአትኣልኡ"]
  Merge ልእጅእ + ኦችእ → ልእጅኦችእ → "ልጆች"
  " " stays as " "
  "ይእጭኣውአትኣልኡ" → "ይጫወታሉ"

Decoded: "ልጆች ይጫወታሉ"  ✓
```

### Properties

| Property | Value |
|---|---|
| Vocabulary size | ~237,000–238,000 tokens |
| Granularity | Morpheme-level (stem + suffix/prefix) |
| OOV handling | CV-pair fallback (every syllable always covered) |
| Script awareness | Full Fidel decomposition via CV map |
| Morphological rules | Hardcoded common Amharic suffixes/prefixes |
| Decoding | Morphophonemic boundary merging |

---

## Side-by-Side Comparison

| Feature | Abol-GMS | Abol-Decomposed |
|---|---|---|
| **Operates on** | Original Fidel script | CV (consonant-vowel) space |
| **Splits "ልጆች"** | ["ልጆች"] (1 token, whole word) | ["ልጅ", "ኦች"] (stem + suffix) |
| **Handles "ይጫወታሉ"** | ["ይጫወታሉ"] | ["ይጫወታሉ"] (in corpus → whole) |
| **Morphological awareness** | No | Yes (suffixes, prefixes) |
| **Fidel decomposition** | No | Yes |
| **Vocabulary size** | ~13,000 | ~238,000 |
| **Token granularity** | Coarser (word-level) | Finer (morpheme-level) |
| **Decode fidelity** | Lossless via composite ID | Lossless via morphophonemic merging |
| **Best for** | Speed, compactness | Morphologically-rich NLP tasks |

---

## API Usage

### Abol-GMS

```python
from amharic_tokenizer.tokenizer import Tokenizer

# Train
tok = Tokenizer()
tok.train_from_corpus("ahun_corpus.txt")
tok.save_pretrained("model_dir/")

# Load and use
tok = Tokenizer.load_pretrained("model_dir/")
ids, spans = tok.encode_with_spans("ሰላም ዓለም")
text = tok.decode(ids)
```

### Abol-Decomposed

```python
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer

# Train
tok = HybridTokenizer()
tok.train_from_corpus("ahun_corpus.txt")
tok.save_pretrained("model_hybrid/")

# Load and use
tok = HybridTokenizer.load_pretrained("model_hybrid/")
ids, spans = tok.encode_with_spans("ልጆች ይጫወታሉ")
# spans[0]['token'] = 'ልጅ', spans[1]['token'] = 'ኦች'
text = tok.decode(ids)
# text = 'ልጆች ይጫወታሉ'
```

### Web API

Both algorithms are available via the FastAPI web app at `http://localhost:8000`:

```bash
# Tokenize with Abol-GMS
curl -X POST http://localhost:8000/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "ሰላም ዓለም", "algorithm": "original"}'

# Tokenize with Abol-Decomposed
curl -X POST http://localhost:8000/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "ልጆች ይጫወታሉ", "algorithm": "hybrid"}'

# Decode IDs
curl -X POST http://localhost:8000/api/decode \
  -H "Content-Type: application/json" \
  -d '{"ids": [233369, 237459, 237791, 141962], "algorithm": "hybrid"}'
```

### Docker

```bash
# Build and start
docker-compose up -d

# App available at http://localhost:8000
```
