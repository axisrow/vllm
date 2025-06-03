#!/usr/bin/env python3
"""
Базовый код для запуска маленькой LLM локально через vLLM на M1 Mac
"""

from vllm import LLM, SamplingParams
import torch

model_name = "distilgpt2"  # ~117MB

def main():
    # Проверяем доступность MPS (Metal Performance Shaders) для M1
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    
    print(f"Загружаем модель: {model_name}")
    
    # Инициализируем LLM с настройками для M1
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # Для одного GPU/устройства
        gpu_memory_utilization=0.7,  # Используем 70% доступной памяти
        max_model_len=512,  # Ограничиваем длину контекста
        enforce_eager=True,  # Отключаем CUDA graphs для совместимости
    )
    
    # Настройки генерации
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
        stop=["</s>", "<|endoftext|>"]
    )
    
    # Тестовые промпты
    prompts = [
        "Привет! Как дела?",
        "Расскажи короткую историю о коте",
        "Что такое машинное обучение?"
    ]
    
    print("\nГенерируем ответы...")
    
    # Генерируем ответы
    outputs = llm.generate(prompts, sampling_params)
    
    # Выводим результаты
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n{'='*50}")
        print(f"Промпт {i+1}: {prompt}")
        print(f"Ответ: {generated_text}")
    
    print(f"\n{'='*50}")
    print("Готово!")

if __name__ == "__main__":
    # Основной режим
    main()