# Dockerfile для vLLM на Sliplane (облачное развертывание)
FROM python:3.12-slim-bookworm

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Создаем непривилегированного пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Создаем рабочую директорию
WORKDIR /app

# Меняем владельца директории
RUN chown -R appuser:appuser /app

# Обновляем pip до последней версии
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Устанавливаем PyTorch CPU версию (совместимость с большинством облачных платформ)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем vLLM
RUN pip install --no-cache-dir vllm

# Устанавливаем дополнительные зависимости
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    requests \
    numpy \
    aiofiles \
    psutil

# Копируем код приложения
COPY --chown=appuser:appuser vllm_cloud_app.py /app/
COPY --chown=appuser:appuser static/ /app/static/
COPY --chown=appuser:appuser requirements.txt /app/

# Переключаемся на непривилегированного пользователя
USER appuser

# Переменные окружения для облачного развертывания
ENV MODEL_NAME=microsoft/DialoGPT-small \
    MAX_MODEL_LEN=512 \
    GPU_MEMORY_UTILIZATION=0.8 \
    PORT=8080 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Открываем порт для Sliplane
EXPOSE 8080

# Health check для облачной платформы
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Команда запуска
CMD ["python", "/app/vllm_cloud_app.py"]