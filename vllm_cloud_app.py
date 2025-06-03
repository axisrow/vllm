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

# Подключаем статические файлы и UI
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    """Веб-интерфейс"""
    return FileResponse("/app/static/index.html")

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