FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Copy processing script
COPY process_pdfs.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the command to run the processing script
CMD ["python", "process_pdfs.py"]
