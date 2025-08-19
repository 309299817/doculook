
FROM python:3.12-slim

WORKDIR /app

# Install runtime libs (headless OpenCV, Pillow, pdfium)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip install uv -i https://mirrors.aliyun.com/pypi/simple

# Copy project files
COPY pyproject.toml ./
COPY doculook/ ./doculook/

# Install project dependencies (API + pipeline only)
ENV UV_HTTP_TIMEOUT=300
RUN uv pip install -e .[pipeline,api] --system -i https://mirrors.aliyun.com/pypi/simple

# Expose port
EXPOSE 8000

# Set environment variable
ENV DOCULOOK_MODEL_SOURCE=modelscope

# Run the API service
ENTRYPOINT ["doculook-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
