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
    aiofiles

# Создаем приложение vLLM для облачного развертывания
RUN cat > /app/vllm_cloud_app.py << 'EOF'
#!/usr/bin/env python3
"""
vLLM сервер для облачного развертывания на Sliplane
Оптимизирован для работы в контейнерной среде
"""

import os
import sys
import logging
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
import torch
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="vLLM Cloud Server",
    version="1.0.0",
    description="Облачный сервер для генерации текста с использованием vLLM"
)

# Глобальная переменная для модели
llm_model = None
model_info = {}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    model: str
    generation_time: float

class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None

def get_optimal_model():
    """Выбирает оптимальную модель в зависимости от доступных ресурсов"""
    
    # Проверяем переменную окружения
    model_name = os.getenv("MODEL_NAME")
    if model_name:
        return model_name
    
    # Автоматический выбор в зависимости от памяти
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 8:
            return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        elif memory_gb >= 4:
            return "microsoft/DialoGPT-medium"
        else:
            return "microsoft/DialoGPT-small"
    except:
        return "microsoft/DialoGPT-small"

def load_model():
    """Загрузка модели vLLM с оптимизацией для облака"""
    global llm_model, model_info
    
    model_name = get_optimal_model()
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "512"))
    
    logger.info(f"🤖 Загружаем модель: {model_name}")
    logger.info(f"📏 Максимальная длина контекста: {max_model_len}")
    
    # Определяем конфигурацию в зависимости от доступности GPU
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA доступна: {torch.cuda.get_device_name(0)}")
        device_config = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8")),
        }
    else:
        logger.info("💻 Используем CPU режим")
        device_config = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.0,
        }
    
    try:
        start_time = time.time()
        
        llm_model = LLM(
            model=model_name,
            max_model_len=max_model_len,
            enforce_eager=True,  # Улучшает совместимость
            trust_remote_code=True,  # Для некоторых моделей
            **device_config
        )
        
        load_time = time.time() - start_time
        
        model_info = {
            "name": model_name,
            "max_length": max_model_len,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "load_time": load_time,
            "status": "loaded"
        }
        
        logger.info(f"✅ Модель загружена за {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        model_info = {"status": "error", "error": str(e)}
        raise

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("🚀 Запуск vLLM Cloud Server")
    
    # Загружаем модель в отдельном потоке чтобы не блокировать startup
    try:
        load_model()
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")

@app.get("/")
async def root():
    """Главная страница"""
    return {
        "message": "vLLM Cloud Server",
        "status": "running",
        "model": model_info.get("name", "not loaded"),
        "endpoints": {
            "generate": "/generate",
            "batch": "/batch",
            "models": "/models",
            "health": "/health",
            "ui": "/ui"
        }
    }

@app.get("/health")
async def health():
    """Health check для Sliplane"""
    return {
        "status": "healthy" if llm_model is not None else "loading",
        "model_loaded": llm_model is not None,
        "model_info": model_info
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Генерация текста"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Модель еще загружается")
    
    try:
        start_time = time.time()
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or ["</s>", "<|endoftext|>", "\n\n"]
        )
        
        outputs = llm_model.generate([request.prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            text=generated_text.strip(),
            prompt=request.prompt,
            model=model_info.get("name", "unknown"),
            generation_time=generation_time
        )
    
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_generate(request: BatchGenerateRequest):
    """Батчевая генерация для нескольких промптов"""
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Модель еще загружается")
    
    try:
        start_time = time.time()
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or ["</s>", "<|endoftext|>", "\n\n"]
        )
        
        outputs = llm_model.generate(request.prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                "prompt": output.prompt,
                "text": output.outputs[0].text.strip(),
                "model": model_info.get("name", "unknown")
            })
        
        generation_time = time.time() - start_time
        
        return {
            "results": results,
            "total_prompts": len(request.prompts),
            "generation_time": generation_time
        }
    
    except Exception as e:
        logger.error(f"❌ Ошибка батчевой генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Информация о загруженной модели"""
    return {
        "current_model": model_info,
        "available_models": [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "distilgpt2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Статистика сервера"""
    try:
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "model_info": model_info
        }
    except ImportError:
        return {"error": "psutil not available"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))  # Sliplane обычно использует 8080
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Запуск сервера на {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
EOF

# Создаем простой веб-интерфейс
RUN mkdir -p /app/static && cat > /app/static/index.html << 'EOF'
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Cloud Server</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr auto;
            gap: 15px;
            margin: 20px 0;
            align-items: end;
        }
        label {
            font-weight: 600;
            color: #555;
        }
        input[type="number"], input[type="range"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            background: #ccc;
            transform: none;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
        .error {
            background: #ffe6e6;
            border-left-color: #dc3545;
            color: #721c24;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-card {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #495057;
        }
        .stat-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 vLLM Cloud Server</h1>
        
        <textarea id="prompt" placeholder="Введите ваш запрос для генерации текста...">Привет! Расскажи короткую историю о космических приключениях.</textarea>
        
        <div class="controls">
            <div>
                <label for="maxTokens">Макс. токенов:</label>
                <input type="number" id="maxTokens" value="150" min="10" max="500">
            </div>
            <div>
                <label for="temperature">Температура:</label>
                <input type="range" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
                <span id="tempValue">0.7</span>
            </div>
            <div>
                <label for="topP">Top-p:</label>
                <input type="range" id="topP" value="0.9" min="0.1" max="1.0" step="0.1">
                <span id="topPValue">0.9</span>
            </div>
            <button onclick="generate()" id="generateBtn">Генерировать</button>
        </div>
        
        <div id="result"></div>
        
        <div class="stats" id="stats"></div>
    </div>

    <script>
        // Обновление отображения значений слайдеров
        document.getElementById('temperature').oninput = function() {
            document.getElementById('tempValue').textContent = this.value;
        }
        
        document.getElementById('topP').oninput = function() {
            document.getElementById('topPValue').textContent = this.value;
        }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            const temperature = parseFloat(document.getElementById('temperature').value);
            const topP = parseFloat(document.getElementById('topP').value);
            
            const resultDiv = document.getElementById('result');
            const generateBtn = document.getElementById('generateBtn');
            
            if (!prompt) {
                alert('Введите запрос');
                return;
            }
            
            generateBtn.disabled = true;
            generateBtn.textContent = 'Генерирую...';
            resultDiv.innerHTML = '<div class="loading">⏳ Генерирую ответ...</div>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        max_tokens: maxTokens,
                        temperature: temperature,
                        top_p: topP
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h3>📝 Результат:</h3>
                            <p><strong>${data.text}</strong></p>
                            <small>Модель: ${data.model} | Время: ${data.generation_time.toFixed(2)}s</small>
                        </div>
                    `;
                } else {
                    throw new Error(data.detail || 'Ошибка генерации');
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>❌ Ошибка:</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = 'Генерировать';
            }
        }
        
        // Загрузка статистики при запуске
        async function loadStats() {
            try {
                const [modelsRes, statsRes] = await Promise.all([
                    fetch('/models'),
                    fetch('/stats')
                ]);
                
                const models = await modelsRes.json();
                const stats = await statsRes.json();
                
                const statsDiv = document.getElementById('stats');
                statsDiv.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${models.current_model.name || 'N/A'}</div>
                        <div class="stat-label">Текущая модель</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${models.current_model.device || 'N/A'}</div>
                        <div class="stat-label">Устройство</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.memory_percent ? stats.memory_percent.toFixed(1) + '%' : 'N/A'}</div>
                        <div class="stat-label">Использование памяти</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${models.current_model.status || 'unknown'}</div>
                        <div class="stat-label">Статус</div>
                    </div>
                `;
            } catch (error) {
                console.error('Ошибка загрузки статистики:', error);
            }
        }
        
        // Загружаем статистику при загрузке страницы
        loadStats();
        
        // Обработка Enter для отправки
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                generate();
            }
        });
    </script>
</body>
</html>
EOF

# Добавляем статические файлы к приложению
RUN cat >> /app/vllm_cloud_app.py << 'EOF'

# Подключаем статические файлы и UI
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Веб-интерфейс"""
    return FileResponse("/app/static/index.html")
EOF

# Устанавливаем psutil для мониторинга
RUN pip3 install psutil

# Создаем requirements.txt для документации
RUN cat > /app/requirements.txt << 'EOF'
vllm>=0.2.0
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
accelerate>=0.20.0
requests>=2.28.0
numpy>=1.24.0
aiofiles>=23.0.0
psutil>=5.9.0
EOF

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