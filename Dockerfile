# Multi-stage build for Amharic Tokenizer Web App
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir --user -r requirements_api.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY amharic_tokenizer/ ./amharic_tokenizer/
COPY static/ ./static/
COPY app.py .
COPY maps.json .

# Copy training scripts and corpus (needed for model training)
COPY train_hybrid_tokenizer.py .
COPY train_decomposed_tokenizer.py .
COPY ahun_corpus.txt .

# Copy pre-trained models if they exist
COPY model_dir/ ./model_dir/
COPY model_decomposed/ ./model_decomposed/
COPY model_morphological/ ./model_morphological/
COPY model_hybrid/ ./model_hybrid/

# If models don't exist, train them (this will be skipped if models are present)
RUN python train_hybrid_tokenizer.py || echo "Hybrid model already exists or training skipped"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/info')"

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
