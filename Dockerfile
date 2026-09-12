# A Dockerfile is a recipe for a computer. Each line adds a layer, and
# Docker caches every layer -- which is why the dependencies get installed
# BEFORE the code is copied. Change main.py and only the last layers rebuild.

FROM python:3.11-slim

# HOME=/tmp because Keras wants a writable home directory and the container
# may not run as root. PORT is a default for local runs; Render injects its
# own value at runtime and that one wins.
ENV HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

WORKDIR /app

# Dependencies first, so this heavy layer is cached between builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the application: code, the trained model, the knowledge document.
COPY . .

EXPOSE 7860

# Two things matter here and both are Render requirements:
#   --host 0.0.0.0   a service that binds 127.0.0.1 is invisible outside the
#                    container, and Render will report "no open ports detected"
#   --port $PORT     Render picks the port and passes it in. Shell form (sh -c)
#                    is what lets $PORT expand; the JSON exec form would pass
#                    the literal string "$PORT" to uvicorn.
# One worker on purpose: each worker loads its own copy of TensorFlow and the
# model, roughly 800 MB. Two workers on a 2 GB instance is an OOM kill.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
