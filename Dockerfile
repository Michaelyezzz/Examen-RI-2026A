FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Hugging Face Spaces ejecuta el contenedor con el usuario UID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

# Solo lo que necesita el servicio
COPY --chown=user app.py corpus.jsonl ./

USER user
EXPOSE 7860

# Shell form para expandir $PORT (7860 por defecto)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
