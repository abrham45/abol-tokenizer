"""
FastAPI Application for Amharic Tokenizer
Beautiful web UI similar to tiktoken.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from amharic_tokenizer.tokenizer import Tokenizer
from amharic_tokenizer.decomposed_tokenizer import DecomposedTokenizer
from amharic_tokenizer.morphological_tokenizer import MorphologicalTokenizer
from amharic_tokenizer.hybrid_tokenizer import HybridTokenizer

# Initialize FastAPI
app = FastAPI(
    title="Amharic Tokenizer",
    description="Tokenize Amharic text with multiple algorithms",
    version="1.0.0"
)

# Mount static files
if os.path.exists('static'):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Load tokenizers
tokenizers = {}

print("Loading tokenizers...")
if os.path.exists('model_dir/tokenizer.json'):
    print("  ✓ Loading Original tokenizer...")
    tokenizers['original'] = Tokenizer.load_pretrained('model_dir')

if os.path.exists('model_decomposed/tokenizer.json'):
    print("  ✓ Loading Decomposed tokenizer...")
    tokenizers['decomposed'] = DecomposedTokenizer.load_pretrained('model_decomposed')

if os.path.exists('model_morphological/tokenizer.json'):
    print("  ✓ Loading Morphological tokenizer...")
    tokenizers['morphological'] = MorphologicalTokenizer.load_pretrained('model_morphological')

if os.path.exists('model_hybrid/tokenizer.json'):
    print("  ✓ Loading Hybrid tokenizer...")
    tokenizers['hybrid'] = HybridTokenizer.load_pretrained('model_hybrid')

print(f"✓ Loaded {len(tokenizers)} tokenizer(s)")


# Request/Response models
class TokenizeRequest(BaseModel):
    text: str
    algorithm: str = "original"  # "original" or "decomposed"


class TokenSpan(BaseModel):
    token: str
    id: int
    start: int
    end: int
    text: str
    decomposed_token: Optional[str] = None


class TokenizeResponse(BaseModel):
    text: str
    algorithm: str
    tokens: List[str]
    ids: List[int]
    spans: List[TokenSpan]
    token_count: int
    vocab_size: int
    decomposed_text: Optional[str] = None


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main UI"""
    with open('static/index.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.get("/api/info")
async def get_info():
    """Get information about available tokenizers"""
    # Custom names for algorithms
    algorithm_names = {
        "original": "Abol-GMS",
        "decomposed": "Decomposed",
        "morphological": "Morphological",
        "hybrid": "Abol-Decomposed"
    }
    
    info = {}
    
    for name, tokenizer in tokenizers.items():
        info[name] = {
            "name": algorithm_names.get(name, name.title()),
            "vocab_size": len(tokenizer.tokens),
            "available": True
        }
    
    return {
        "tokenizers": info,
        "default": "original"
    }


@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize_text(request: TokenizeRequest):
    """Tokenize text using specified algorithm"""
    
    if request.algorithm not in tokenizers:
        raise HTTPException(
            status_code=400, 
            detail=f"Algorithm '{request.algorithm}' not available. Choose from: {list(tokenizers.keys())}"
        )
    
    tokenizer = tokenizers[request.algorithm]
    
    # Encode
    ids, spans = tokenizer.encode_with_spans(request.text)
    tokens = [s['token'] for s in spans]
    
    # Prepare response
    response = {
        "text": request.text,
        "algorithm": request.algorithm,
        "tokens": tokens,
        "ids": ids,
        "spans": [
            {
                "token": s['token'],
                "id": s['id'],
                "start": s['start'],
                "end": s['end'],
                "text": s['text'],
                "decomposed_token": s.get('decomposed_token')
            }
            for s in spans
        ],
        "token_count": len(tokens),
        "vocab_size": len(tokenizer.tokens)
    }
    
    # Add decomposed text if using decomposed, morphological, or hybrid tokenizer
    if (request.algorithm in ["decomposed", "morphological", "hybrid"]) and hasattr(tokenizer, 'decomposer'):
        response["decomposed_text"] = tokenizer.decomposer.decompose_word(request.text)
    
    return response


@app.post("/api/decode")
async def decode_tokens(request: dict):
    """Decode token IDs back to text"""
    
    algorithm = request.get('algorithm', 'original')
    ids = request.get('ids', [])
    
    if algorithm not in tokenizers:
        raise HTTPException(status_code=400, detail=f"Algorithm '{algorithm}' not available")
    
    tokenizer = tokenizers[algorithm]
    decoded = tokenizer.decode(ids)
    
    return {
        "ids": ids,
        "decoded": decoded,
        "algorithm": algorithm
    }


@app.get("/api/compare")
async def compare_algorithms():
    """Get comparison of all algorithms"""
    
    comparison = {}
    
    for name, tokenizer in tokenizers.items():
        comparison[name] = {
            "name": name.title(),
            "vocab_size": len(tokenizer.tokens),
            "type": "Decomposed (CV)" if name == "decomposed" else "Direct"
        }
    
    return comparison


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 Starting Amharic Tokenizer Web App")
    print("="*80)
    print(f"\n📍 Open in browser: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
