from typing import Dict, Tuple, Optional
import unicodedata, json

COMMON_VOWEL_SUFFIXES = ['AA','A','U','I','E','O','EE','Ä','IH']
COMMON_VOWEL_SUFFIXES_SORTED = sorted(COMMON_VOWEL_SUFFIXES, key=lambda x: -len(x))

def _parse_unicode_name_to_cv(ch: str):
    try:
        name = unicodedata.name(ch)
    except Exception:
        return (ch,)
    if 'ETHIOPIC SYLLABLE' not in name:
        return (ch,)
    suffix = name.split()[-1]
    vowel_found = None
    for v in COMMON_VOWEL_SUFFIXES_SORTED:
        if suffix.endswith(v):
            vowel_found = v
            break
    if vowel_found is None:
        vowel_found = suffix[-1:]
    consonant_part = suffix[:len(suffix) - len(vowel_found)]
    if consonant_part == '':
        consonant_part = suffix
        vowel_found = 'A'
    return (f"C_{consonant_part}", f"V_{vowel_found}")

def _consonant_root_from_Ctok(Ctok: str) -> str:
    if not isinstance(Ctok, str) or not Ctok.startswith("C_"):
        return Ctok
    parts = Ctok.split("_", 2)
    return parts[1] if len(parts) >= 2 else "_".join(parts[1:])

def _rank_g(glyph: str):
    cp = ord(glyph)
    if 0x1200 <= cp <= 0x137F:
        blk = 0
    elif 0x1380 <= cp <= 0x139F:
        blk = 1
    elif 0x2D80 <= cp <= 0x2DDF:
        blk = 2
    else:
        blk = 99
    return (blk, cp)

def build_unique_cv_key_maps(precomputed_map: Optional[Dict[str, Tuple[str, str]]] = None):
    base_to_glyphs = {}
    def _add_glyph_to_base(glyph: str, dv):
        if not (isinstance(dv, (list, tuple)) and len(dv) == 2):
            return
        Ctok, Vtok = dv[0], dv[1]
        root = _consonant_root_from_Ctok(Ctok)
        base_key = (root, Vtok)
        base_to_glyphs.setdefault(base_key, []).append(glyph)

    if precomputed_map:
        for glyph, dv in precomputed_map.items():
            _add_glyph_to_base(glyph, dv)

    ranges = [ (0x1200, 0x1380), (0x1380, 0x13A0), (0x2D80, 0x2DE0) ]
    for start, end in ranges:
        for cp in range(start, end):
            ch = chr(cp)
            dv = _parse_unicode_name_to_cv(ch)
            if isinstance(dv, tuple) and len(dv) == 2 and isinstance(dv[0], str) and dv[0].startswith("C_"):
                _add_glyph_to_base(ch, dv)

    key_to_glyph = {}
    glyph_to_key = {}
    for base_key in sorted(base_to_glyphs.keys(), key=lambda x: (str(x[0]), str(x[1]))):
        glyphs = base_to_glyphs[base_key]
        if len(glyphs) == 1:
            g = glyphs[0]
            root, Vtok = base_key
            Ctok = f"C_{root}"
            token = (Ctok, Vtok)
            key_to_glyph[token] = g
            glyph_to_key[g] = token
        else:
            glyphs.sort(key=_rank_g)
            root, Vtok = base_key
            for g in glyphs:
                hexsuffix = f"{ord(g):04X}"
                Ctok = f"C_{root}_{hexsuffix}"
                token = (Ctok, Vtok)
                key_to_glyph[token] = g
                glyph_to_key[g] = token

    return key_to_glyph, glyph_to_key

def save_json_map(mapping, filename: str):
    serial = {}
    for k, v in mapping.items():
        serial_key = json.dumps(k, ensure_ascii=False) if not isinstance(k, str) else k
        serial[serial_key] = v
    with open(filename, 'w', encoding='utf-8') as fh:
        json.dump(serial, fh, ensure_ascii=False, indent=2)

def load_json_map(filename: str):
    with open(filename, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    out = {}
    for k, v in data.items():
        try:
            kk = json.loads(k)
            if isinstance(kk, list):
                kk = tuple(kk)
            out[kk] = v
        except Exception:
            out[k] = v
    return out
