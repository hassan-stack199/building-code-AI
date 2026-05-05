FROM python:3.11-slim

WORKDIR /app

# System deps (none beyond base; pypdf is pure python)
RUN pip install --no-cache-dir --upgrade pip

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /app/cache && \
    chown -R user:user /app
USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
