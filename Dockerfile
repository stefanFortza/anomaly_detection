FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv runpod && \
    uv pip install --system --no-cache \
    combo \
    matplotlib \
    pyod \
    scikit-learn \
    scipy

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
