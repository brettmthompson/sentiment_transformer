FROM python:3.12-slim-bookworm

WORKDIR /app

# Create cache directory for Hugging Face models with proper permissions
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Set environment variables for Hugging Face cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HUB_CACHE=/app/.cache/hub

# Copy dependency files first for better layer caching
COPY requirements.txt ./

# Copy application code
COPY sentiment_transformer/ ./sentiment_transformer/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Expose port
EXPOSE 8080

# Run the application
ENTRYPOINT ["python", "-m", "sentiment_transformer.transformer"]
