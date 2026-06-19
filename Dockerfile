# ============================================================
# Dockerfile — SC-RLOT Cancer Drug Resistance Predictor
# ============================================================
# Build:  docker build -t sc-rlot-app:1.0 .
# Run:    docker run -p 8501:8501 sc-rlot-app:1.0
# Open:   http://localhost:8501
# ============================================================

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY configs/   ./configs/
COPY src/       ./src/
COPY models/    ./models/
COPY app/       ./app/
COPY artifacts/ ./artifacts/

# Create empty data and mlruns dirs
RUN mkdir -p data mlruns

# Streamlit config to disable browser auto-open in Docker
RUN mkdir -p /root/.streamlit && \
    echo '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\n' \
    > /root/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
