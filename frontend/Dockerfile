# Minimal Dockerfile to containerize the Streamlit UI
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY frontend /app/frontend
COPY backend /app/backend
COPY .env.sample /app/.env.sample

EXPOSE 8501

CMD ["streamlit", "run", "frontend/rag_gui.py", "--server.port=8501", "--server.runOnSave=false"]
