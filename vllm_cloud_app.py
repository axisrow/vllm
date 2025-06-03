#!/usr/bin/env python3
"""
vLLM Cloud Server - простая рабочая версия (только transformers)
"""

import os
import logging
import time
from contextlib import asynccontextmanager # Импортируем asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import torch # Для проверки доступности MPS

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорты vLLM
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
    logger.info("✅ vLLM доступен")
except ImportError:
    HAS_VLLM = False
    logger.error("❌ vLLM недоступен")

# Глобальные переменные
llm = None
current_model_info = {"status": "loading", "name": "none"}

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150 # Возвращаем значение по умолчанию
    temperature: float = 0.7

class ChangeModelRequest(BaseModel):
    model_name: str

def load_model_vllm(model_name: str):
    """Загрузка модели через vLLM"""
    global llm, current_model_info
    
    logger.info(f"Загружаем {model_name} через vLLM")
    
    try:
        # Определяем целевое устройство: сначала проверяем MPS, затем CPU
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        logger.info(f"Используемое устройство: {device}")

        # Если устройство CPU, явно устанавливаем VLLM_TARGET_DEVICE
        if device == "cpu":
            os.environ["VLLM_TARGET_DEVICE"] = "cpu"
            logger.info("Установлена переменная окружения VLLM_TARGET_DEVICE=\"cpu\" для использования CPU.")

        llm = LLM(
            model=model_name,
            device=device, # Явно указываем устройство
            tensor_parallel_size=1,  # Для одного GPU/устройства
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", 0.8)),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", 512)),
            enforce_eager=True,  # Отключаем CUDA graphs для совместимости с CPU/MPS
            dtype="float16" if device == "mps" else "auto" # Используем float16 для MPS
        )
        
        current_model_info = {
            "status": "loaded",
            "name": model_name,
            "backend": "vLLM",
            "device": device
        }
        logger.info(f"✅ Модель {model_name} загружена")
        
    except Exception as e:
        current_model_info = {"status": "error", "error": str(e)}
        logger.error(f"❌ Ошибка загрузки: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Обработчик событий жизненного цикла приложения"""
    model_name = os.getenv("MODEL_NAME", "distilgpt2") # Используем distilgpt2 по умолчанию
    
    if HAS_VLLM:
        # Запускаем загрузку модели в фоновом режиме
        asyncio.create_task(load_model_async(model_name))
    else:
        current_model_info["status"] = "error"
        current_model_info["error"] = "vLLM не установлен"
    
    yield # Приложение запущено

    # Здесь можно добавить логику очистки при завершении работы приложения
    logger.info("Приложение завершает работу.")

async def load_model_async(model_name: str):
    """Асинхронная обертка для загрузки модели"""
    try:
        load_model_vllm(model_name)
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели в фоновом режиме: {e}")
        current_model_info["status"] = "error"
        current_model_info["error"] = str(e)

app = FastAPI(title="vLLM Cloud Server", lifespan=lifespan) # Добавляем lifespan

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
        
        if llm is None:
            raise HTTPException(500, "LLM не доступен")
            
        # Настройки генерации для vLLM
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=0.9, # Используем фиксированный top_p, так как он не передается из UI
            max_tokens=request.max_tokens,
            stop=["</s>", "<|endoftext|>"] # Общие токены остановки
        )

        # Генерируем ответы
        outputs = llm.generate([request.prompt], sampling_params)
        
        # Безопасно извлекаем текст
        text = ""
        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
        
        logger.info(f"Сгенерированный результат: {outputs}") # Добавляем логирование
        logger.info(f"Извлеченный текст: {text}") # Добавляем логирование
        
        return {
            "text": text.strip(),
            "prompt": request.prompt,
            "model": current_model_info["name"],
            "generation_time": time.time() - start_time
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
            "distilgpt2", # vLLM лучше работает с моделями, не требующими специфического токенизатора
            "gpt2",
            "microsoft/DialoGPT-small", # Добавляем обратно, если пользователь захочет попробовать
            "microsoft/DialoGPT-medium"
        ],
        "backends": {
            "vllm": HAS_VLLM
        }
    }

@app.post("/change_model")
async def change_model(request: ChangeModelRequest):
    """Смена модели"""
    global current_model_info, llm # Изменено generator на llm
    
    current_model_info = {"status": "loading", "name": request.model_name}
    
    # Очистка
    llm = None # Очищаем llm
    
    try:
        if HAS_VLLM: # Изменено HAS_TRANSFORMERS на HAS_VLLM
            load_model_vllm(request.model_name) # Изменено load_model_transformers на load_model_vllm
        else:
            raise Exception("vLLM недоступен") # Изменено "Transformers недоступен" на "vLLM недоступен"
            
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
