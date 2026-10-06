# A clean Linux environment that runs every check in the repository:
#   docker build -t heurics-cert .
#   docker run --rm heurics-cert
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ENV JAX_PLATFORMS=cpu
RUN pip install --no-cache-dir -e ".[test]" && heurics-cert build-checker
CMD ["sh", "-c", "heurics-cert info && python -m pytest -q && python tools/adversarial.py && python tools/cross_check.py --out /tmp/cross_check.md && python examples/quickstart.py"]
