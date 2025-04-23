FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/videos

# Expose port
EXPOSE 5050

# Run with gunicorn
CMD ["gunicorn", "--workers=2", "--timeout=300", "--threads=4", "--bind=0.0.0.0:5050", "app:app"]
