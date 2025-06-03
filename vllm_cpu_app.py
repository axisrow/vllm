#!/usr/bin/env python3
"""
vLLM сервер для облачного развертывания (CPU-only версия)
Оптимизирован для работы без GPU
"""

import os
import sys
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Пытаемся импортировать vLLM, если не получается - используем transformers
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    logging.info("✅ vLLM доступен")
except ImportError:
    logging.warning("⚠️ vLLM недоступен, используем transformers")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline
    import torch
    VLLM_AVAILABLE = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="vLLM Cloud Server",
    version="1.0.0",
    description="Облачный сервер для генерации текста"
)

# Глобальные переменные
model = None
tokenizer = None
generator = None
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

def get_optimal_model():
    """Выбирает оптимальную модель в зависимости от доступных ресурсов"""
    model_name = os.getenv("MODEL_NAME")
    if model_name:
        return model_name
    
    # Для CPU-only используем легкие модели
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if memory_gb >= 8:
            return "microsoft/DialoGPT-medium"
        elif memory_gb >= 4:
            return "distilgpt2"
        else:
            return "microsoft/DialoGPT-small"
    except:
        return "microsoft/DialoGPT-small"

def load_model_vllm():
    """Загрузка модели через vLLM"""
    global model, model_info
    
    model_name = get_optimal_model()
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "512"))
    
    logger.info(f"🤖 Загружаем модель через vLLM: {model_name}")
    
    try:
        start_time = time.time()
        
        model = LLM(
            model=model_name,
            max_model_len=max_model_len,
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.0,  # CPU режим
        )
        
        load_time = time.time() - start_time
        
        model_info = {
            "name": model_name,
            "max_length": max_model_len,
            "device": "cpu",
            "backend": "vllm",
            "load_time": load_time,
            "status": "loaded"
        }
        
        logger.info(f"✅ Модель загружена за {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки vLLM: {e}")
        raise

def load_model_transformers():
    """Загрузка модели через transformers (fallback)"""
    global tokenizer, generator, model_info
    
    model_name = get_optimal_model()
    max_length = int(os.getenv("MAX_MODEL_LEN", "512"))
    
    logger.info(f"🤖 Загружаем модель через transformers: {model_name}")
    
    try:
        start_time = time.time()
        
        # Загружаем токенизатор и модель
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Добавляем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Создаем pipeline для генерации
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device=-1,  # CPU
            torch_dtype=torch.float32,
            max_length=max_length,
            return_full_text=False  # Возвращать только сгенерированный текст
        )
        
        load_time = time.time() - start_time
        
        model_info = {
            "name": model_name,
            "max_length": max_length,
            "device": "cpu",
            "backend": "transformers",
            "load_time": load_time,
            "status": "loaded"
        }
        
        logger.info(f"✅ Модель загружена за {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки transformers: {e}")
        raise

def load_model():
    """Главная функция загрузки модели"""
    if VLLM_AVAILABLE:
        try:
            load_model_vllm()
            return
        except Exception as e:
            logger.warning(f"vLLM не сработал: {e}, переключаемся на transformers")
    
    load_model_transformers()

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("🚀 Запуск Cloud Server")
    
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
        "backend": model_info.get("backend", "unknown"),
        "endpoints": {
            "generate": "/generate",
            "models": "/models",
            "health": "/health",
            "ui": "/ui"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    is_loaded = (model is not None) or (generator is not None)
    return {
        "status": "healthy" if is_loaded else "loading",
        "model_loaded": is_loaded,
        "model_info": model_info
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Генерация текста"""
    if model is None and generator is None:
        raise HTTPException(status_code=503, detail="Модель еще загружается")
    
    try:
        start_time = time.time()
        
        if VLLM_AVAILABLE and model is not None:
            # Используем vLLM
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop or ["</s>", "<|endoftext|>"]
            )
            
            outputs = model.generate([request.prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
        elif generator is not None:
            # Используем transformers
            result = generator(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer else None
            )
            
            # Extracting generated text properly
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip()
            else:
                generated_text = ""
        
        else:
            raise HTTPException(status_code=500, detail="Модель не загружена корректно")
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            text=generated_text,
            prompt=request.prompt,
            model=model_info.get("name", "unknown"),
            generation_time=generation_time
        )
    
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {e}")
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
            "gpt2"
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
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Запуск сервера на {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )