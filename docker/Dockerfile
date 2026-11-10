FROM python:3.11-slim

# Install OpenJDK (required by OpenDataLoader PDF) and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code
COPY app /app/app

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/odl_output /app/data/chroma

ENV PYTHONPATH=/app
ENV UPLOAD_DIR=/app/data/uploads
ENV ODL_OUTPUT_DIR=/app/data/odl_output
ENV CHROMA_DIR=/app/data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.ingest_api:app", "--host", "0.0.0.0", "--port", "8000"]
