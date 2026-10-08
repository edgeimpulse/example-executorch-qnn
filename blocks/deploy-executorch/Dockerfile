# ExecuTorch deployment block.
# python:3.11-slim keeps the image small. ExecuTorch wheels are large; if the
# build times out, pre-bake dependencies into a custom base image.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY build.py ./
COPY app ./app

# Edge Impulse passes --metadata <path to deployment-metadata.json>.
ENTRYPOINT ["python3", "build.py"]
