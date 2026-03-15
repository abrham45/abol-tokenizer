"""
Fidel Decomposition Module

This module handles decomposition and recomposition of Amharic syllabic characters (fidel)
into consonant-vowel pairs using the FIDEL_MAP.

Example:
    ሰላም → ስአልኣምእ (decompose)
    ስአልኣምእ → ሰላም (recompose)
"""

import json
from typing import Dict, Optional

# Fidel mapping: syllable → consonant+vowel
AMHARIC_FIDEL_MAP = {
    "ሀ":"ህአ","ሁ":"ህኡ","ሂ":"ህኢ","ሃ":"ህኣ","ሄ":"ህኤ","ህ":"ህእ","ሆ":"ህኦ","ሇ":"ህኡኣ",
    "ለ":"ልአ","ሉ":"ልኡ","ሊ":"ልኢ","ላ":"ልኣ","ሌ":"ልኤ","ል":"ልእ","ሎ":"ልኦ","ሏ":"ልኡኣ",
    "ሐ":"ሕአ","ሑ":"ሕኡ","ሒ":"ሕኢ","ሓ":"ሕኣ","ሔ":"ሕኤ","ሕ":"ሕእ","ሖ":"ሕኦ","ሗ":"ሕኡኣ",
    "መ":"ምአ","ሙ":"ምኡ","ሚ":"ምኢ","ማ":"ምኣ","ሜ":"ምኤ","ም":"ምእ","ሞ":"ምኦ","ሟ":"ምኡኣ",
    "ሠ":"ሥአ","ሡ":"ሥኡ","ሢ":"ሥኢ","ሣ":"ሥኣ","ሤ":"ሥኤ","ሥ":"ሥእ","ሦ":"ሥኦ","ሧ":"ሥኡኣ",
    "ረ":"ርአ","ሩ":"ርኡ","ሪ":"ርኢ","ራ":"ርኣ","ሬ":"ርኤ","ር":"ርእ","ሮ":"ርኦ","ሯ":"ርኡኣ",
    "ሰ":"ስአ","ሱ":"ስኡ","ሲ":"ስኢ","ሳ":"ስኣ","ሴ":"ስኤ","ስ":"ስእ","ሶ":"ስኦ","ሷ":"ስኡኣ",
    "ሸ":"ሽአ","ሹ":"ሽኡ","ሺ":"ሽኢ","ሻ":"ሽኣ","ሼ":"ሽኤ","ሽ":"ሽእ","ሾ":"ሽኦ","ሿ":"ሽኡኣ",
    "ቀ":"ቅአ","ቁ":"ቅኡ","ቂ":"ቅኢ","ቃ":"ቅኣ","ቄ":"ቅኤ","ቅ":"ቅእ","ቆ":"ቅኦ","ቋ":"ቅኡኣ",
    "በ":"ብአ","ቡ":"ብኡ","ቢ":"ብኢ","ባ":"ብኣ","ቤ":"ብኤ","ብ":"ብእ","ቦ":"ብኦ","ቧ":"ብኡኣ",
    "ቨ":"ቭአ","ቩ":"ቭኡ","ቪ":"ቭኢ","ቫ":"ቭኣ","ቬ":"ቭኤ","ቭ":"ቭእ","ቮ":"ቭኦ","ቯ":"ቭኡኣ",
    "ተ":"ትአ","ቱ":"ትኡ","ቲ":"ትኢ","ታ":"ትኣ","ቴ":"ትኤ","ት":"ትእ","ቶ":"ትኦ","ቷ":"ትኡኣ",
    "ቸ":"ችአ","ቹ":"ችኡ","ቺ":"ችኢ","ቻ":"ችኣ","ቼ":"ችኤ","ች":"ችእ","ቾ":"ችኦ","ቿ":"ችኡኣ",
    "ኀ":"ኅአ","ኁ":"ኅኡ","ኂ":"ኅኢ","ኃ":"ኅኣ","ኄ":"ኅኤ","ኅ":"ኅእ","ኆ":"ኅኦ","ኋ":"ኅኡኣ",
    "ነ":"ንአ","ኑ":"ንኡ","ኒ":"ንኢ","ና":"ንኣ","ኔ":"ንኤ","ን":"ንእ","ኖ":"ንኦ","ኗ":"ንኡኣ",
    "ኘ":"ኝአ","ኙ":"ኝኡ","ኚ":"ኝኢ","ኛ":"ኝኣ","ኜ":"ኝኤ","ኝ":"ኝእ","ኞ":"ኝኦ","ኟ":"ኝኡኣ",
    "አ":"አ","ኡ":"ኡ","ኢ":"ኢ","ኣ":"ኣ","ኤ":"ኤ","እ":"እ","ኦ":"ኦ","ኧ":"ኧ",
    "ከ":"ክአ","ኩ":"ክኡ","ኪ":"ክኢ","ካ":"ክኣ","ኬ":"ክኤ","ክ":"ክእ","ኮ":"ክኦ","ኳ":"ክኡኣ",
    "ኸ":"ኽአ","ኹ":"ኽኡ","ኺ":"ኽኢ","ኻ":"ኽኣ","ኼ":"ኽኤ","ኽ":"ኽእ","ኾ":"ኽኦ","ዃ":"ኽኡኣ",
    "ወ":"ውአ","ዉ":"ውኡ","ዊ":"ውኢ","ዋ":"ውኣ","ዌ":"ውኤ","ው":"ውእ","ዎ":"ውኦ","ዏ":"ውኡኣ",
    "ዐ":"ዕአ","ዑ":"ዕኡ","ዒ":"ዕኢ","ዓ":"ዕኣ","ዔ":"ዕኤ","ዕ":"ዕእ","ዖ":"ዕኦ",
    "ዘ":"ዝአ","ዙ":"ዝኡ","ዚ":"ዝኢ","ዛ":"ዝኣ","ዜ":"ዝኤ","ዝ":"ዝእ","ዞ":"ዝኦ","ዟ":"ዝኡኣ",
    "ዠ":"ዥአ","ዡ":"ዥኡ","ዢ":"ዥኢ","ዣ":"ዥኣ","ዤ":"ዥኤ","ዥ":"ዥእ","ዦ":"ዥኦ","ዧ":"ዥኡኣ",
    "የ":"ይአ","ዩ":"ይኡ","ዪ":"ይኢ","ያ":"ይኣ","ዬ":"ይኤ","ይ":"ይእ","ዮ":"ይኦ","ዯ":"ይኡኣ",
    "ደ":"ድአ","ዱ":"ድኡ","ዲ":"ድኢ","ዳ":"ድኣ","ዴ":"ድኤ","ድ":"ድእ","ዶ":"ድኦ","ዷ":"ድኡኣ",
    "ጀ":"ጅአ","ጁ":"ጅኡ","ጂ":"ጅኢ","ጃ":"ጅኣ","ጄ":"ጅኤ","ጅ":"ጅእ","ጆ":"ጅኦ","ጇ":"ጅኡኣ",
    "ገ":"ግአ","ጉ":"ግኡ","ጊ":"ግኢ","ጋ":"ግኣ","ጌ":"ግኤ","ግ":"ግእ","ጎ":"ግኦ","ጏ":"ግኡኣ",
    "ጠ":"ጥአ","ጡ":"ጥኡ","ጢ":"ጥኢ","ጣ":"ጥኣ","ጤ":"ጥኤ","ጥ":"ጥእ","ጦ":"ጥኦ","ጧ":"ጥኡኣ",
    "ጨ":"ጭአ","ጩ":"ጭኡ","ጪ":"ጭኢ","ጫ":"ጭኣ","ጬ":"ጭኤ","ጭ":"ጭእ","ጮ":"ጭኦ","ጯ":"ጭኡኣ",
    "ጰ":"ጵአ","ጱ":"ጵኡ","ጲ":"ጵኢ","ጳ":"ጵኣ","ጴ":"ጵኤ","ጵ":"ጵእ","ጶ":"ጵኦ","ጷ":"ጵኡኣ",
    "ጸ":"ጽአ","ጹ":"ጽኡ","ጺ":"ጽኢ","ጻ":"ጽኣ","ጼ":"ጽኤ","ጽ":"ጽእ","ጾ":"ጽኦ","ጿ":"ጽኡኣ",
    "ፀ":"ፅአ","ፁ":"ፅኡ","ፂ":"ፅኢ","ፃ":"ፅኣ","ፄ":"ፅኤ","ፅ":"ፅእ","ፆ":"ፅኦ","ፇ":"ፅኡኣ",
    "ፈ":"ፍአ","ፉ":"ፍኡ","ፊ":"ፍኢ","ፋ":"ፍኣ","ፌ":"ፍኤ","ፍ":"ፍእ","ፎ":"ፍኦ","ፏ":"ፍኡኣ",
    "ፐ":"ፕአ","ፑ":"ፕኡ","ፒ":"ፕኢ","ፓ":"ፕኣ","ፔ":"ፕኤ","ፕ":"ፕእ","ፖ":"ፕኦ","ፗ":"ፕኡኣ"
}


class FidelDecomposer:
    """
    Handles decomposition and recomposition of Amharic fidel characters.
    
    Decomposition: ሰላም → ስአልኣምእ
    Recomposition: ስአልኣምእ → ሰላም
    """
    
    def __init__(self, fidel_map: Optional[Dict[str, str]] = None):
        """
        Initialize the decomposer with a fidel map.
        
        Args:
            fidel_map: Optional custom fidel map. If None, uses default AMHARIC_FIDEL_MAP.
        """
        self.fidel_to_cv = fidel_map if fidel_map is not None else AMHARIC_FIDEL_MAP
        self.cv_to_fidel = self._build_reverse_map(self.fidel_to_cv)
        
    def _build_reverse_map(self, fidel_map: Dict[str, str]) -> Dict[str, str]:
        """
        Build reverse mapping: consonant+vowel → syllable
        
        Args:
            fidel_map: Forward mapping (syllable → CV)
            
        Returns:
            Reverse mapping (CV → syllable)
        """
        reverse = {}
        for syllable, cv in fidel_map.items():
            reverse[cv] = syllable
        return reverse
    
    def decompose_word(self, word: str) -> str:
        """
        Decompose an Amharic word into consonant-vowel representation.
        
        Characters not in the fidel map (punctuation, spaces, etc.) are kept as-is.
        
        Args:
            word: Original Amharic word (e.g., "ሰላም")
            
        Returns:
            Decomposed representation (e.g., "ስአልኣምእ")
            
        Example:
            >>> decomposer = FidelDecomposer()
            >>> decomposer.decompose_word("ሰላም")
            'ስአልኣምእ'
        """
        decomposed = []
        for char in word:
            if char in self.fidel_to_cv:
                decomposed.append(self.fidel_to_cv[char])
            else:
                # Keep non-fidel characters as-is (punctuation, spaces, digits, etc.)
                decomposed.append(char)
        return ''.join(decomposed)
    
    def recompose_word(self, decomposed: str) -> str:
        """
        Recompose a decomposed string back into original Amharic.
        
        Args:
            decomposed: Decomposed representation (e.g., "ስአልኣምእ")
            
        Returns:
            Original Amharic word (e.g., "ሰላም")
            
        Example:
            >>> decomposer = FidelDecomposer()
            >>> decomposer.recompose_word("ስአልኣምእ")
            'ሰላም'
        """
        original = []
        i = 0
        
        while i < len(decomposed):
            matched = False
            
            # Try 3-character match first (for ኡኣ vowels like "ህኡኣ" → "ሇ")
            if i + 3 <= len(decomposed):
                cv_triple = decomposed[i:i+3]
                if cv_triple in self.cv_to_fidel:
                    original.append(self.cv_to_fidel[cv_triple])
                    i += 3
                    matched = True
            
            # Try 2-character match (standard CV pairs like "ስአ" → "ሰ")
            if not matched and i + 2 <= len(decomposed):
                cv_pair = decomposed[i:i+2]
                if cv_pair in self.cv_to_fidel:
                    original.append(self.cv_to_fidel[cv_pair])
                    i += 2
                    matched = True
            
            # Try 1-character match (for standalone vowels like "አ", "እ")
            if not matched and i + 1 <= len(decomposed):
                single_char = decomposed[i]
                if single_char in self.cv_to_fidel:
                    original.append(self.cv_to_fidel[single_char])
                    i += 1
                    matched = True
            
            # Fallback: keep character as-is (for punctuation, spaces, etc.)
            if not matched:
                original.append(decomposed[i])
                i += 1
        
        return ''.join(original)
    
    def decompose_corpus(self, corpus: list) -> list:
        """
        Decompose an entire corpus.
        
        Args:
            corpus: List of original words
            
        Returns:
            List of decomposed words
        """
        return [self.decompose_word(word) for word in corpus]
    
    def get_mapping_for_word(self, word: str) -> Dict[str, str]:
        """
        Get the decomposition mapping for a specific word.
        
        Args:
            word: Original word
            
        Returns:
            Dictionary mapping original → decomposed
        """
        return {word: self.decompose_word(word)}


def load_fidel_map_from_file(filepath: str) -> Dict[str, str]:
    """
    Load fidel map from a JSON file.
    
    Args:
        filepath: Path to JSON file containing fidel map
        
    Returns:
        Fidel mapping dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the dictionary from the file
        # Handle both JSON and Python dict formats
        if 'AMHARIC_FIDEL_MAP' in content:
            # Python file format
            import ast
            start = content.find('{')
            end = content.rfind('}') + 1
            dict_str = content[start:end]
            return ast.literal_eval(dict_str)
        else:
            # Pure JSON format
            return json.load(f)


if __name__ == "__main__":
    # Demo usage
    decomposer = FidelDecomposer()
    
    print("=" * 70)
    print("FIDEL DECOMPOSER DEMO")
    print("=" * 70)
    
    test_words = ["ሰላም", "እንዴት ነህ?", "መጽሐፍ", "ልጆች"]
    
    print("\n📝 DECOMPOSITION:")
    for word in test_words:
        decomposed = decomposer.decompose_word(word)
        print(f"  '{word}' → '{decomposed}'")
    
    print("\n🔄 RECOMPOSITION (Round-trip test):")
    for word in test_words:
        decomposed = decomposer.decompose_word(word)
        recomposed = decomposer.recompose_word(decomposed)
        status = "✓" if word == recomposed else "✗"
        print(f"  '{word}' → '{decomposed}' → '{recomposed}' {status}")
    
    print("\n" + "=" * 70)

