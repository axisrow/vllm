#!/usr/bin/env python3
"""
vLLM сервер для облачного развертывания (упрощенная версия)
Убраны сложные типы для лучшей совместимости
"""

import os
import sys
import logging
import time
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Приоритет transformers для стабильности
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers доступен")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("❌ Transformers недоступен")

# vLLM как дополнительная опция
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    logger.info("✅ vLLM доступен")
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("⚠️ vLLM недоступен")

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
available_models = [
    "microsoft/DialoGPT-small",
    "microsoft/DialoGPT-medium", 
    "distilgpt2",
    "gpt2"
]

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    stop: List[str] = []

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    model: str
    generation_time: float

class ChangeModelRequest(BaseModel):
    model_name: str
    max_model_len: int = 512
    backend: str = "auto"

def get_optimal_model():
    """Выбирает оптимальную модель в зависимости от доступных ресурсов"""
    model_name = os.getenv("MODEL_NAME")
    if model_name:
        return model_name
    
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

def load_model_transformers(model_name=None, max_length=None):
    """Загрузка модели через transformers"""
    global tokenizer, generator, model_info
    
    if model_name is None:
        model_name = get_optimal_model()
    if max_length is None:
        max_length = int(os.getenv("MAX_MODEL_LEN", "512"))
    
    logger.info(f"🤖 Загружаем модель через transformers: {model_name}")
    
    try:
        start_time = time.time()
        
        # Загружаем токенизатор
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
            return_full_text=False,
            max_length=max_length
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
        model_info = {"status": "error", "error": str(e)}
        raise

def load_model_vllm(model_name=None, max_length=None):
    """Загрузка модели через vLLM"""
    global model, model_info
    
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM не установлен")
    
    if model_name is None:
        model_name = get_optimal_model()
    if max_length is None:
        max_length = int(os.getenv("MAX_MODEL_LEN", "512"))
    
    logger.info(f"🤖 Загружаем модель через vLLM: {model_name}")
    
    try:
        start_time = time.time()
        
        model = LLM(
            model=model_name,
            max_model_len=max_length,
            enforce_eager=True,
            tensor_parallel_size=1,
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        
        model_info = {
            "name": model_name,
            "max_length": max_length,
            "device": "cpu",
            "backend": "vllm",
            "load_time": load_time,
            "status": "loaded"
        }
        
        logger.info(f"✅ Модель загружена за {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки vLLM: {e}")
        raise

def load_model(model_name=None, backend="auto"):
    """Главная функция загрузки модели"""
    global model, tokenizer, generator, model_info
    
    # Очищаем предыдущую модель
    model = None
    tokenizer = None
    generator = None
    
    if backend == "transformers" or (backend == "auto" and TRANSFORMERS_AVAILABLE):
        try:
            load_model_transformers(model_name)
            return
        except Exception as e:
            logger.warning(f"Transformers не сработал: {e}")
            if backend == "transformers":
                raise
    
    if backend == "vllm" or (backend == "auto" and VLLM_AVAILABLE):
        try:
            load_model_vllm(model_name)
            return
        except Exception as e:
            logger.warning(f"vLLM не сработал: {e}")
            if backend == "vllm":
                raise
    
    # Fallback на transformers
    if TRANSFORMERS_AVAILABLE:
        load_model_transformers(model_name)
    else:
        raise RuntimeError("Ни vLLM, ни transformers не доступны")

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    logger.info("🚀 Запуск Cloud Server")
    
    try:
        load_model()
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        global model_info
        model_info = {"status": "error", "error": str(e)}

@app.get("/")
async def root():
    """Главная страница"""
    return {
        "message": "vLLM Cloud Server",
        "status": "running",
        "model": model_info.get("name", "not loaded"),
        "backend": model_info.get("backend", "unknown"),
        "backends_available": {
            "vllm": VLLM_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE
        },
        "endpoints": {
            "generate": "/generate",
            "change_model": "/change_model",
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
        raise HTTPException(status_code=503, detail="Модель еще загружается или не загружена")
    
    try:
        start_time = time.time()
        
        if model is not None and model_info.get("backend") == "vllm":
            # Используем vLLM
            stop_sequences = request.stop if request.stop else ["</s>", "<|endoftext|>"]
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=stop_sequences
            )
            
            outputs = model.generate([request.prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
        elif generator is not None:
            # Используем transformers
            pad_token_id = tokenizer.eos_token_id if tokenizer else None
            result = generator(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=pad_token_id
            )
            
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
        "available_models": available_models,
        "available_backends": {
            "vllm": VLLM_AVAILABLE,
            "transformers": TRANSFORMERS_AVAILABLE
        }
    }

@app.post("/change_model")
async def change_model(request: ChangeModelRequest):
    """Смена модели"""
    if request.model_name not in available_models:
        raise HTTPException(status_code=400, detail=f"Модель {request.model_name} не поддерживается")
    
    try:
        logger.info(f"🔄 Смена модели на {request.model_name} (backend: {request.backend})")
        
        # Устанавливаем статус загрузки
        global model_info
        model_info = {"status": "loading", "name": request.model_name}
        
        # Перезагружаем модель
        load_model(request.model_name, request.backend)
        
        return {
            "message": f"Модель успешно изменена на {request.model_name}",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка смены модели: {e}")
        model_info = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")

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