# ============================================================
# Dockerfile — SC-RLOT Cancer Drug Resistance Predictor
# Build:  docker build -t sc-rlot-app:1.0 .
# Run:    docker run -p 8501:8501 sc-rlot-app:1.0
# Open:   http://localhost:8501
# ============================================================

FROM python:3.10-slim

# System dependencies — curl is REQUIRED for the healthcheck
RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use inference-only requirements (not the full training requirements.txt)
COPY requirements_app.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy application code and saved models
COPY configs/   ./configs/
COPY src/       ./src/
COPY models/    ./models/
COPY app/       ./app/
COPY artifacts/ ./artifacts/

COPY mlruns/    ./mlruns/ 

COPY data/gdsc_merged_ic50.csv ./data/
COPY data/pbmc3k_raw.h5ad       ./data/

# Create empty dirs
RUN mkdir -p data mlruns

# Streamlit config — use printf to get real newlines in sh
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
