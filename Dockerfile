# Dockerfile — Crypto Trading Agent
# Python 3.11 slim — small image, stable, all our deps work on it

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (we want live logs in fly logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (Docker layer cache — only rebuilds if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# The DATA_DIR env var routes all state files to the persistent volume (/data)
# Set in fly.toml — no need to set here
ENV DATA_DIR=/data

# Run the agent
CMD ["python3", "main.py"]
