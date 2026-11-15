import csv
import statistics
import time
from pathlib import Path

import torch


# параметры эксперимента
MATRIX_SIZE = 2048
WARMUP_RUNS = 5
MEASURE_RUNS = 10

# папка с результатами
RESULTS_PATH = Path(__file__).resolve().parent / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# проверяем доступность GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна")

print("GPU:", torch.cuda.get_device_name(0))
print("Размер матриц:", MATRIX_SIZE, "x", MATRIX_SIZE)

# фиксируем случайность
torch.manual_seed(42)

# создаём матрицы на CPU
a_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)
b_cpu = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32)

# копируем те же матрицы на GPU
a_gpu = a_cpu.to("cuda")
b_gpu = b_cpu.to("cuda")


# измеряем время матричного умножения
@torch.inference_mode()
def measure_time(a, b, device_name):
    times = []

    # прогрев перед измерениями
    for _ in range(WARMUP_RUNS):
        result = torch.matmul(a, b)

    if device_name == "cuda":
        torch.cuda.synchronize()

    # повторные измерения
    for run in range(MEASURE_RUNS):
        if device_name == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = torch.matmul(a, b)

        if device_name == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        times.append(elapsed_ms)

        print(f"{device_name.upper()} | запуск {run + 1}: {elapsed_ms:.3f} мс")

    return times


# запускаем эксперимент
print("\nТестирование CPU...")
cpu_times = measure_time(a_cpu, b_cpu, "cpu")

print("\nТестирование GPU...")
gpu_times = measure_time(a_gpu, b_gpu, "cuda")

# вычисляем медианы
cpu_median = statistics.median(cpu_times)
gpu_median = statistics.median(gpu_times)

# рассчитываем ускорение
speedup = cpu_median / gpu_median

print("\nРЕЗУЛЬТАТЫ CPU VS GPU")
print(f"CPU: {cpu_median:.3f} мс")
print(f"GPU: {gpu_median:.3f} мс")
print(f"Ускорение GPU: {speedup:.2f}x")


# сохраняем все измерения в CSV
csv_file = RESULTS_PATH / "cpu_gpu_results.csv"

with csv_file.open(
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)

    writer.writerow(["device", "run", "time_ms", "matrix_size"])

    for run, elapsed in enumerate(cpu_times, start=1):
        writer.writerow(["cpu", run, elapsed, MATRIX_SIZE])

    for run, elapsed in enumerate(gpu_times, start=1):
        writer.writerow(["gpu", run, elapsed, MATRIX_SIZE])

print(f"\nРезультаты сохранены: {csv_file}")
