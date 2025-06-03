#!/usr/bin/env python3
"""
vLLM Cloud Server - простая рабочая версия (только transformers)
"""

import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio # Добавляем импорт asyncio

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорты
try:
    from transformers import AutoTokenizer
    from transformers.pipelines import pipeline
    import torch
    HAS_TRANSFORMERS = True
    logger.info("✅ Transformers доступен")
except ImportError:
    HAS_TRANSFORMERS = False
    logger.error("❌ Transformers недоступен")

app = FastAPI(title="vLLM Cloud Server")

# Глобальные переменные
generator = None
tokenizer = None
current_model_info = {"status": "loading", "name": "none"}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50 # Уменьшаем значение по умолчанию для снижения потребления памяти
    temperature: float = 0.7

class ChangeModelRequest(BaseModel):
    model_name: str

def load_model_transformers(model_name):
    """Загрузка через transformers"""
    global generator, tokenizer, current_model_info
    
    logger.info(f"Загружаем {model_name} через transformers")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            device="cpu",  # CPU
            return_full_text=False
        )
        
        current_model_info = {
            "status": "loaded",
            "name": model_name,
            "backend": "transformers"
        }
        logger.info(f"✅ Модель {model_name} загружена")
        
    except Exception as e:
        current_model_info = {"status": "error", "error": str(e)}
        logger.error(f"❌ Ошибка загрузки: {e}")
        raise

@app.on_event("startup")
async def startup():
    """Загрузка модели при старте (асинхронно)"""
    model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-small") # Изменено на меньшую модель
    
    if HAS_TRANSFORMERS:
        # Запускаем загрузку модели в фоновом режиме
        asyncio.create_task(load_model_async(model_name))
    else:
        current_model_info["status"] = "error"
        current_model_info["error"] = "Transformers не установлен"

async def load_model_async(model_name: str):
    """Асинхронная обертка для загрузки модели"""
    try:
        load_model_transformers(model_name)
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели в фоновом режиме: {e}") # Используем logger.exception для полной трассировки
        current_model_info["status"] = "error"
        current_model_info["error"] = str(e)

@app.get("/")
async def root():
    """Главная страница - для healthcheck"""
    return {
        "status": "ok",
        "service": "vLLM Cloud Server",
        "model": current_model_info
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "model": current_model_info}

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Генерация текста"""
    if current_model_info.get("status") != "loaded":
        raise HTTPException(503, "Модель не загружена")
    
    try:
        start_time = time.time()
        
        if generator is None:
            raise HTTPException(500, "Генератор не доступен")
            
        # Генерируем текст
        result = generator(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id if tokenizer else None
        )
        
        # Безопасно извлекаем текст
        text = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                text = str(result[0]["generated_text"])
        
        logger.info(f"Сгенерированный результат: {result}") # Добавляем логирование
        logger.info(f"Извлеченный текст: {text}") # Добавляем логирование
        
        return {
            "text": text.strip(),
            "prompt": request.prompt,
            "model": current_model_info["name"],
            "generation_time": time.time() - start_time # Изменено с "time" на "generation_time"
        }
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        raise HTTPException(500, str(e))

@app.get("/models")
async def get_models():
    """Список моделей"""
    return {
        "current": current_model_info,
        "available": [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "distilgpt2",
            "gpt2"
        ],
        "backends": {
            "transformers": HAS_TRANSFORMERS
        }
    }

@app.post("/change_model")
async def change_model(request: ChangeModelRequest):
    """Смена модели"""
    global current_model_info, generator
    
    current_model_info = {"status": "loading", "name": request.model_name}
    
    # Очистка
    generator = None
    
    try:
        if HAS_TRANSFORMERS:
            load_model_transformers(request.model_name)
        else:
            raise Exception("Transformers недоступен")
            
        return {"message": f"Модель изменена на {request.model_name}"}
        
    except Exception as e:
        current_model_info = {"status": "error", "error": str(e)}
        raise HTTPException(500, str(e))

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def ui():
    """Веб интерфейс"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
