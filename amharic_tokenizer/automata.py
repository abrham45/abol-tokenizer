from collections import deque

class SuffixAutomaton:
    def __init__(self, s: str):
        self.next = []
        self.link = []
        self.length = []
        self.last = 0
        self._build(s)

    def _new_state(self, length=0):
        self.next.append({})
        self.link.append(-1)
        self.length.append(length)
        return len(self.next) - 1

    def _build(self, s: str):
        self.next = []
        self.link = []
        self.length = []
        self._new_state(0)
        self.last = 0
        for ch in s:
            cur = self._new_state(self.length[self.last] + 1)
            p = self.last
            while p >= 0 and ch not in self.next[p]:
                self.next[p][ch] = cur
                p = self.link[p]
            if p == -1:
                self.link[cur] = 0
            else:
                q = self.next[p][ch]
                if self.length[p] + 1 == self.length[q]:
                    self.link[cur] = q
                else:
                    clone = self._new_state(self.length[p] + 1)
                    self.next[clone] = self.next[q].copy()
                    self.link[clone] = self.link[q]
                    while p >= 0 and self.next[p].get(ch) == q:
                        self.next[p][ch] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            self.last = cur

    def contains(self, substring: str) -> bool:
        p = 0
        for ch in substring:
            if ch not in self.next[p]:
                return False
            p = self.next[p][ch]
        return True


class AhoCorasick:
    def __init__(self, words=None):
        self.next = []
        self.fail = []
        self.out = []
        self._make_node()
        if words:
            for w in words:
                self.add_word(w)
            self.build_automaton()

    def _make_node(self):
        self.next.append({})
        self.fail.append(0)
        self.out.append([])
        return len(self.next) - 1

    def add_word(self, word):
        node = 0
        for ch in word:
            if ch not in self.next[node]:
                nxt = self._make_node()
                self.next[node][ch] = nxt
            node = self.next[node][ch]
        self.out[node].append(word)

    def build_automaton(self):
        from collections import deque
        q = deque()
        for ch, nxt in list(self.next[0].items()):
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in list(self.next[r].items()):
                q.append(s)
                state = self.fail[r]
                while state and ch not in self.next[state]:
                    state = self.fail[state]
                self.fail[s] = self.next[state].get(ch, 0)
                self.out[s].extend(self.out[self.fail[s]])

    def find_all(self, text):
        node = 0
        results = []
        for i, ch in enumerate(text):
            while node and ch not in self.next[node]:
                node = self.fail[node]
            node = self.next[node].get(ch, 0)
            if self.out[node]:
                for w in self.out[node]:
                    results.append((w, i))
        return results


class TokenIndex:
    def __init__(self, tokens=None):
        self.tokens = set(tokens) if tokens else set()
        self.ac = None
        self._last_build_size = -1

    def ensure_built(self):
        if self.ac is None or len(self.tokens) != self._last_build_size:
            self.ac = AhoCorasick(self.tokens)
            self._last_build_size = len(self.tokens)

    def add(self, tok: str):
        if tok in self.tokens:
            return
        self.tokens.add(tok)
        self._last_build_size = -1

    def find_matches(self, piece: str):
        if not self.tokens:
            return []
        self.ensure_built()
        matches = self.ac.find_all(piece)
        return [(w, end - len(w) + 1) for w, end in matches]

def add_and_split_ac(token_index: TokenIndex, new_text: str) -> None:
    q = deque([new_text])
    while q:
        piece = q.popleft()
        if piece in token_index.tokens:
            continue
        if not token_index.tokens:
            token_index.add(piece)
            continue
        matches = token_index.find_matches(piece)
        if not matches:
            token_index.add(piece)
            continue
        matches.sort(key=lambda t: (-len(t[0]), t[1]))
        tok, idx = matches[0]
        prefix = piece[:idx]
        suffix = piece[idx + len(tok):]
        if prefix:
            q.append(prefix)
        if suffix:
            q.append(suffix)

def split_corpus_greedy_subword(corpus):
    """
    New greedy meaningful subword approach for stem detection.
    
    For each word:
    1. Check character by character: a, ab, abc, abcd...
    2. Check if each substring is meaningful (exists in corpus)
    3. Take the subword when current is meaningful but next extension is not
    4. Continue from next position
    
    Returns: prefixes, stems, suffixes
    """
    # Build a set of all words in corpus for O(1) lookup
    corpus_set = set(corpus)
    
    prefixes = set()
    stems = set()
    suffixes = set()
    
    # Add all single characters as base tokens
    for word in corpus:
        for ch in word:
            prefixes.add(ch)
    
    # Process each word to find meaningful subwords
    for word in corpus:
        if len(word) <= 1:
            continue
            
        tokens_in_word = []
        i = 0
        word_len = len(word)
        
        while i < word_len:
            # Build subwords incrementally from position i
            longest_meaningful = None
            longest_len = 0
            
            for j in range(i + 1, word_len + 1):
                subword = word[i:j]
                
                # Check if this subword is meaningful (exists in corpus)
                if subword in corpus_set:
                    # Check if extending it further is also meaningful
                    if j < word_len:
                        extended = word[i:j+1]
                        if extended not in corpus_set:
                            # Current is meaningful, next is not - take current
                            longest_meaningful = subword
                            longest_len = j - i
                            break
                        else:
                            # Both are meaningful, keep checking
                            longest_meaningful = subword
                            longest_len = j - i
                    else:
                        # Reached end of word
                        longest_meaningful = subword
                        longest_len = j - i
                else:
                    # Not meaningful, stop if we had a previous match
                    if longest_meaningful is not None:
                        break
            
            # If we found a meaningful subword, add it
            if longest_meaningful is not None and longest_len > 0:
                tokens_in_word.append(longest_meaningful)
                i += longest_len
            else:
                # No meaningful subword found, take single character
                tokens_in_word.append(word[i])
                i += 1
        
        # Classify tokens from this word into prefixes, stems, suffixes
        if len(tokens_in_word) == 1:
            # Single token = stem
            stems.add(tokens_in_word[0])
        elif len(tokens_in_word) == 2:
            # Two tokens: prefix + stem or stem + suffix
            # Heuristic: if first is longer, it's likely stem + suffix
            if len(tokens_in_word[0]) >= len(tokens_in_word[1]):
                stems.add(tokens_in_word[0])
                suffixes.add(tokens_in_word[1])
            else:
                prefixes.add(tokens_in_word[0])
                stems.add(tokens_in_word[1])
        elif len(tokens_in_word) >= 3:
            # Three or more: prefix + stem + suffix
            prefixes.add(tokens_in_word[0])
            stems.add(tokens_in_word[1])
            for suffix_tok in tokens_in_word[2:]:
                suffixes.add(suffix_tok)
    
    return prefixes, stems, suffixes


def split_corpus_ac(corpus):
    """
    Original algorithm using Suffix Automaton and Aho-Corasick.
    Kept for backward compatibility.
    """
    prefixes = set()
    suffixes = set()
    stems = set()
    corpus = sorted(corpus, key=len)
    n = len(corpus)
    skips = set()

    char_to_indices = {}
    for idx, w in enumerate(corpus):
        seen = set()
        for ch in w:
            if ch in seen: continue
            seen.add(ch)
            char_to_indices.setdefault(ch, []).append(idx)

    sa_cache = {}
    pref_index = TokenIndex(prefixes)
    suf_index = TokenIndex(suffixes)

    for i in range(n):
        if len(corpus[i]) > 1:
            if i not in skips:
                stem = corpus[i]
                best_list = None
                seen_chars = set()
                for ch in stem:
                    if ch in seen_chars: continue
                    seen_chars.add(ch)
                    lst = char_to_indices.get(ch, [])
                    if best_list is None or len(lst) < len(best_list):
                        best_list = lst
                candidates = best_list if best_list is not None else range(i+1, n)
                stem_found_as_substring = False
                for j in candidates:
                    if j <= i:
                        continue
                    word = corpus[j]
                    if len(word) < len(stem):
                        continue
                    sa = sa_cache.get(word)
                    if sa is None:
                        sa = SuffixAutomaton(word)
                        sa_cache[word] = sa
                    if sa.contains(stem):
                        idx = word.find(stem)
                        prefix = word[:idx]
                        suffix = word[idx + len(stem):]
                        if prefix and prefix not in prefixes:
                            add_and_split_ac(pref_index, prefix)
                            prefixes = pref_index.tokens
                        if suffix and suffix not in suffixes:
                            add_and_split_ac(suf_index, suffix)
                            suffixes = suf_index.tokens
                        skips.add(j)
                        stem_found_as_substring = True
                # Add as stem if it was found as a substring in another word,
                # OR if it's a meaningful word that should be preserved as a stem
                if stem_found_as_substring or len(stem) >= 2:  # Add words of 2+ characters as stems
                    stems.add(stem)
        elif corpus[i] not in prefixes:
            prefixes.add(corpus[i])
            pref_index.tokens = prefixes
            pref_index._last_build_size = -1

    prefixes = pref_index.tokens
    suffixes = suf_index.tokens
    return prefixes, stems, suffixes