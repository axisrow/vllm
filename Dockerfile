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

# Устанавливаем vLLM (для CPU)
# vLLM требует PyTorch, который будет установлен как зависимость
RUN pip install --no-cache-dir \
    vllm \
    fastapi \
    uvicorn \
    requests \
    numpy \
    aiofiles \
    psutil

# Копируем код приложения (только необходимые файлы)
COPY --chown=appuser:appuser vllm_cloud_app.py /app/
COPY --chown=appuser:appuser static/ /app/static/

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт для Sliplane
EXPOSE 8080

# Health check для облачной платформы (проверяем главную страницу)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8080/ || exit 1

# Команда запуска
CMD ["python", "/app/vllm_cloud_app.py"]
