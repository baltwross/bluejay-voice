# =============================================================================
# Bluejay Terminator - Backend Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Stage 1: Builder - installs all dependencies
# Stage 2: Runtime - minimal image with only runtime requirements
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Silero VAD model during build (required by LiveKit agents)
# This runs the agent's download-files command to pre-cache models
RUN python -c "from livekit.plugins import silero; silero.VAD.load()"

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for SSL certificates
    ca-certificates \
    # Required for some audio processing
    libsndfile1 \
    # Required for healthcheck
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the Silero VAD model from builder (pre-downloaded)
COPY --from=builder /root/.cache /home/appuser/.cache

# Copy application code
COPY agent.py .
COPY token_server.py .
COPY config.py .
COPY prompts.py .
COPY rag/ ./rag/
COPY src/ ./src/

# Create directory for ChromaDB persistence
# In production, this should be mounted as a volume (EFS)
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app

# Create directory for reading state persistence
RUN mkdir -p /app && touch /app/.reading_state.json && \
    chown appuser:appuser /app/.reading_state.json

# Switch to non-root user
USER appuser

# Environment variables
# Note: Actual secrets should be injected via AWS Secrets Manager
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHROMA_PERSIST_DIRECTORY=/app/chroma_db

# Expose ports
# 8080: Token server (FastAPI)
# Note: LiveKit agent connects outbound to LiveKit Cloud, no inbound port needed
EXPOSE 8080

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: Run both token server and agent
# For production, you may want to run these as separate containers
# and use a process manager like supervisord
CMD ["sh", "-c", "python token_server.py & python agent.py start"]

