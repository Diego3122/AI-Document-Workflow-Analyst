FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    WEBSITES_PORT=8501 \
    HOME=/home/appuser

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends supervisor \
    && useradd --create-home --home-dir /home/appuser --uid 1000 --gid 0 appuser \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh \
    && mkdir -p /home/site/data/uploads /home/site/data/chroma /home/site/data/logs /home/site/data/evals \
    && chgrp -R 0 /app /home/site /home/appuser \
    && chmod -R g=u /app /home/site /home/appuser

USER appuser

EXPOSE 8501

CMD ["/app/start.sh"]
