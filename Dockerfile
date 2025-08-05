# Dockerfile
FROM python:3.13-slim

# 1. Disable pyc & unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Tambah ~/.local/bin ke PATH supaya uv bisa dipakai
ENV PATH="/root/.local/bin:${PATH}"

# 3. Install curl → unduh & pasang uv → symlink ke /usr/local/bin
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && ln -s /root/.local/bin/uv /usr/local/bin/uv

# 4. Set working dir
WORKDIR /projectwise_mcpserver

# 5. Copy file dependency
COPY pyproject.toml ./
COPY uv.lock ./

# 6. Sync deps (non-dev)
RUN uv sync --no-dev

# 7. Copy seluruh kode aplikasi
COPY . .

# 8. Expose port yang dipakai app
EXPOSE 5000

# 9. Jalankan Flask via uv
CMD ["uv", "run", "main.py"]
