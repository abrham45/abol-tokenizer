"""Utility functions for file IO and lightweight Amharic preprocessing."""
from typing import Optional, List
import re

# Ethiopic Unicode blocks; keep common punctuation and digits
ETHIOPIC_RE = re.compile(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF]+')
NON_ETHIOPIC_RE = re.compile(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF0-9\s\-\.,:?!\u1361\u1362\u1363\ufeff]+', re.U)

SPLITTER_RE = re.compile(r'(\s+|(?=[\.\,\:\?\!\-\u1361\u1362\u1363])|(?<=[\.\,\:\?\!\-\u1361\u1362\u1363]))')

def load_text_file(filename: str, encoding: str = 'utf-8') -> Optional[str]:
    try:
        with open(filename, 'r', encoding=encoding) as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        return None

def map_non_ethiopic_to_placeholders(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if re.search(r'[A-Za-z]', s):
            return ' '
        if re.search(r'[\u4E00-\u9FFF]', s):
            return ' '
        if re.search(r'\d', s):
            # keep digits separated
            return ' ' + ' '.join(list(re.sub(r'\D', '', s))) + ' '
        return ' '
    return NON_ETHIOPIC_RE.sub(repl, text)

def simple_tokenize_keep_punct(text: str) -> List[str]:
    parts = SPLITTER_RE.split(text)
    tokens: List[str] = []
    for p in parts:
        if not p:
            continue
        if p.isspace():
            continue
        tokens.append(p.strip())
    return tokens