import csv
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# параметры эксперимента
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 160
REPEATS = 4

# путь для сохранения результатов
RESULTS_PATH = Path(__file__).resolve().parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# проверяем доступность GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print("Модель:", MODEL_NAME)

# загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# загружаем модель на GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to(device)

model.eval()

print("Модель загружена успешно")


# создаём одинаковый запрос для всех измерений
messages = [
    {
        "role": "user",
        "content": (
            "Объясни подробно архитектуру Transformer Decoder. "
            "Расскажи о механизме Self-Attention, матрицах "
            "Query, Key, Value и генерации следующих токенов."
        )
    }
]

# преобразуем запрос в токены
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

input_ids = encoded["input_ids"]

print("Входных токенов:", input_ids.shape[1])
print("Генерируемых токенов:", MAX_NEW_TOKENS)


# выполняем генерацию в выбранном режиме
@torch.inference_mode()
def generate_text(use_cache, max_new_tokens):
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tokenizer.eos_token_id
    )


# измеряем время выполнения
def measure_generation(use_cache):
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    output_ids = generate_text(
        use_cache=use_cache,
        max_new_tokens=MAX_NEW_TOKENS
    )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    speed = generated_tokens / elapsed

    return elapsed, speed, generated_tokens


# прогреваем оба режима перед тестированием
print("\nПрогрев GPU...")

generate_text(use_cache=False, max_new_tokens=8)
generate_text(use_cache=True, max_new_tokens=8)

torch.cuda.synchronize()


# сохраняем результаты отдельных запусков
results = []

for run in range(1, REPEATS + 1):
    print(f"\nПовторение {run}/{REPEATS}")

    # чередуем порядок режимов между повторениями
    modes = [False, True] if run % 2 else [True, False]

    for use_cache in modes:
        elapsed, speed, tokens = measure_generation(use_cache)

        result = {
            "run": run,
            "use_cache": use_cache,
            "time_seconds": elapsed,
            "tokens_per_second": speed,
            "generated_tokens": tokens
        }

        results.append(result)

        mode_name = "С KV-cache" if use_cache else "Без KV-cache"

        print(
            f"{mode_name}: {elapsed:.2f} сек, "
            f"{speed:.2f} токенов/сек"
        )


# отдельно собираем времена для двух режимов
without_cache = [
    item["time_seconds"]
    for item in results
    if not item["use_cache"]
]

with_cache = [
    item["time_seconds"]
    for item in results
    if item["use_cache"]
]


# вычисляем медианные значения
median_without = statistics.median(without_cache)
median_with = statistics.median(with_cache)

# вычисляем ускорение
speedup = median_without / median_with


# выводим итоговую статистику
print()
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ KV-CACHE")
print()


print(f"Количество повторений: {REPEATS}")
print(f"Без KV-cache: {median_without:.3f} сек")
print(f"С KV-cache: {median_with:.3f} сек")
print(f"Ускорение: {speedup:.2f}x")


# сохраняем подробные результаты в CSV
csv_file = RESULTS_PATH / "kv_cache_results.csv"

with csv_file.open(
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "run",
            "use_cache",
            "time_seconds",
            "tokens_per_second",
            "generated_tokens"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nРезультаты сохранены: {csv_file}")
