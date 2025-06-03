# Dockerfile для vLLM на Sliplane (облачное развертывание)
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем PyTorch (CPU версия для совместимости с разными облачными провайдерами)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем vLLM
RUN pip3 install vllm

# Устанавливаем дополнительные зависимости
RUN pip3 install \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    requests \
    numpy \
    aiofiles \
    psutil

# Копируем код приложения
COPY vllm_cloud_app.py /app/
COPY static/ /app/static/

# Создаем requirements.txt
COPY requirements.txt /app/

# Переменные окружения для облачного развертывания
ENV MODEL_NAME=microsoft/DialoGPT-small
ENV MAX_MODEL_LEN=512
ENV GPU_MEMORY_UTILIZATION=0.8
ENV PORT=8080
ENV HOST=0.0.0.0

# Открываем порт для Sliplane
EXPOSE 8080

# Health check для облачной платформы
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Команда запуска
CMD ["python3", "/app/vllm_cloud_app.py"]