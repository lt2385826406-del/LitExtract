FROM python:3.10-slim-bookworm

LABEL maintainer="Materials Informatics Lab"
LABEL description="LitExtract: Automated Literature Knowledge Extraction for Materials Science"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment.yml (optional, for reference)
COPY environment.yml .

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install --no-cache-dir -e .

# Expose port for Gradio UI (if used)
EXPOSE 7860

# Default command: show help
CMD ["python", "scripts/run_pipeline.py", "--help"]
