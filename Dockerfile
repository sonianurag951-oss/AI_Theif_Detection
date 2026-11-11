FROM python:3.11-slim

WORKDIR /app
# 1. Install system utilities and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy entire project code (models, src, dashboard.py, etc.)
COPY . .
# 4. Expose both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501
# 5. Healthcheck verifying that both services are alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8501/_stcore/health || exit 1

# 6. Start FastAPI in the background and Streamlit in the foreground
CMD sh -c "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0"