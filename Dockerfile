FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and PyTorch Geometric CPU wheels (lightweight for container inference)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase and artifacts
COPY config.py model.py explain.py utils.py baselines.py trace_service.py server.py run_dashboard.py ./
COPY checkpoints/ ./checkpoints/
COPY outputs/ ./outputs/
COPY static/ ./static/
COPY data/processed/ ./data/processed/
COPY data/HI-Small_accounts.csv ./data/HI-Small_accounts.csv
# Note: Raw HI-Small_Trans.csv (475MB) is excluded from image via .dockerignore; 
# the processed aml_graph.pt (68MB) and enriched_cases.json provide full graph & transaction serving.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]
