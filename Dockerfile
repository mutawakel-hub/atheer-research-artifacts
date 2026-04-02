FROM python:3.10-slim

WORKDIR /app

# Upgrade pip and set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the simulation code
COPY . .

# Ensure outputs directory exists
RUN mkdir -p /app/outputs

CMD ["python", "atheer_sim.py"]
