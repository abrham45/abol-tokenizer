"""Top-level package for amharic_tokenizer"""
from .tokenizer import Tokenizer
from .maps import build_unique_cv_key_maps, save_json_map, load_json_map

__all__ = ["Tokenizer", "build_unique_cv_key_maps", "save_json_map", "load_json_map"]
