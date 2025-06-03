# vLLM Docker Setup

Контейнер для запуска маленьких LLM моделей через vLLM с возможностью SSH туннелирования на Mac.

## 🚀 Быстрый запуск

### 1. Построение образа
```bash
docker build -t vllm-server .
```

### 2. Запуск контейнера
```bash
# Простой запуск
docker run -p 27003:27003 vllm-server

# С GPU поддержкой
docker run --gpus all -p 27003:27003 vllm-server

# Через docker-compose
docker-compose up
```

### 3. Доступ к серверу
- API: http://localhost:27003
- Web UI: http://localhost:27003/ui
- Health check: http://localhost:27003/health

## 🔗 SSH Туннель (для Mac)

### Вариант 1: Автоматический (через переменные)
```bash
docker run --gpus all \
  -e SSH_HOST=192.168.1.100 \
  -e SSH_USER=your_username \
  -e SSH_PORT=2222 \
  -e REMOTE_PORT=8080 \
  vllm-server
```

### Вариант 2: Ручной SSH туннель
```bash
# В контейнере выполните:
ssh -N -R 8080:localhost:27003 username@mac_ip -p 2222

# Затем на Mac откройте: http://localhost:8080
```

### Вариант 3: Docker Compose с SSH
```yaml
# В docker-compose.yml раскомментируйте:
environment:
  - SSH_HOST=192.168.1.100
  - SSH_USER=your_username
  - SSH_PORT=2222
```

## ⚙️ Настройки

### Переменные окружения
- `MODEL_NAME` - название модели (по умолчанию: microsoft/DialoGPT-small)
- `MAX_MODEL_LEN` - максимальная длина контекста (512)
- `GPU_MEMORY_UTILIZATION` - использование GPU памяти (0.8)
- `PORT` - порт сервера (27003)

### Поддерживаемые модели
```bash
# Очень маленькие (< 200MB)
MODEL_NAME=microsoft/DialoGPT-small
MODEL_NAME=distilgpt2

# Маленькие (< 1GB)  
MODEL_NAME=gpt2

# Средние (1-3GB)
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MODEL_NAME=microsoft/DialoGPT-medium
```

## 📡 API Использование

### Генерация текста
```bash
curl -X POST "http://localhost:27003/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Привет! Как дела?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Python клиент
```python
import requests

response = requests.post("http://localhost:27003/generate", json={
    "prompt": "Расскажи короткую историю",
    "max_tokens": 200,
    "temperature": 0.8
})

print(response.json()["text"])
```

## 🔧 Разработка

### Структура проекта
```
.
├── Dockerfile              # Основной образ
├── docker-compose.yml      # Compose конфигурация  
├── vllm_app.py            # Основное приложение (создается автоматически)
├── requirements.txt        # Python зависимости
├── models/                # Кэш моделей
└── static/
    └── index.html         # Web интерфейс
```

### Логи
```bash
# Просмотр логов
docker-compose logs -f vllm-server

# Подключение к контейнеру
docker exec -it vllm-container bash
```

## 🐛 Troubleshooting

### Проблемы с памятью
```bash
# Уменьшите использование GPU памяти
-e GPU_MEMORY_UTILIZATION=0.5

# Или используйте CPU
docker run --cpus=4 -p 27003:27003 vllm-server
```

### SSH туннель не работает
1. Проверьте SSH сервер на Mac (из вашего Python скрипта)
2. Убедитесь что порты не заняты
3. Проверьте firewall настройки

### Модель не загружается
```bash
# Попробуйте другую модель
-e MODEL_NAME=distilgpt2

# Проверьте доступную память
docker stats vllm-container
```

## 📝 Полезные команды

```bash
# Остановка всех контейнеров
docker-compose down

# Пересборка образа
docker-compose build --no-cache

# Просмотр использования ресурсов
docker stats

# Очистка
docker system prune -f
```