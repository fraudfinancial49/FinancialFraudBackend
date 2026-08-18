FROM python:3.11-slim

# Set the working directory
WORKDIR /srv

# Install system dependencies (needed for math/ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# Note: Ensure gdown is included in your requirements.txt
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app code
COPY backend/app ./app

# Set env variables
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/srv

EXPOSE 8000

# Run the app
# Use Uvicorn directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
