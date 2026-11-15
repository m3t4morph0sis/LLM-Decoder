import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


# папка с результатами экспериментов
RESULTS_PATH = Path(__file__).resolve().parent / "results"


# функция для чтения CSV-файлов
def read_results(filename):
    file_path = RESULTS_PATH / filename

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


# загружаем результаты KV-cache
kv_results = read_results("kv_cache_results.csv")

# разделяем измерения по режимам
without_cache = [
    float(row["time_seconds"])
    for row in kv_results
    if row["use_cache"].lower() == "false"
]

with_cache = [
    float(row["time_seconds"])
    for row in kv_results
    if row["use_cache"].lower() == "true"
]

# вычисляем медианное время генерации
median_without = statistics.median(without_cache)
median_with = statistics.median(with_cache)

# рассчитываем ускорение от кеширования
kv_speedup = median_without / median_with


# строим график KV-cache
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["Without KV-cache", "With KV-cache"],
    [median_without, median_with],
    color=["steelblue", "seagreen"]
)

ax.set_title("KV-cache: Generation Time")
ax.set_ylabel("Time (seconds)")
ax.bar_label(bars, fmt="%.3f s", padding=3)
ax.set_ylim(0, max(median_without, median_with) * 1.2)

plt.tight_layout()
plt.savefig(
    RESULTS_PATH / "kv_cache_comparison.png",
    dpi=160
)
plt.close()


# загружаем результаты CPU/GPU
benchmark_results = read_results("cpu_gpu_results.csv")

# разделяем измерения по устройствам
cpu_times = [
    float(row["time_ms"])
    for row in benchmark_results
    if row["device"].lower() == "cpu"
]

gpu_times = [
    float(row["time_ms"])
    for row in benchmark_results
    if row["device"].lower() == "gpu"
]

# вычисляем медианное время
cpu_median = statistics.median(cpu_times)
gpu_median = statistics.median(gpu_times)

# рассчитываем ускорение GPU
gpu_speedup = cpu_median / gpu_median

# получаем размер матриц из CSV
matrix_size = benchmark_results[0]["matrix_size"]


# строим график CPU/GPU
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["CPU", "GPU (RTX 3060 Ti)"],
    [cpu_median, gpu_median],
    color=["steelblue", "seagreen"]
)

ax.set_title(
    f"CPU vs GPU: Matrix Multiplication ({matrix_size} x {matrix_size})"
)

ax.set_ylabel("Time (milliseconds)")
ax.bar_label(bars, fmt="%.3f ms", padding=3)
ax.set_ylim(0, max(cpu_median, gpu_median) * 1.2)

plt.tight_layout()
plt.savefig(
    RESULTS_PATH / "cpu_gpu_comparison.png",
    dpi=160
)
plt.close()


# выводим итоговые результаты
print("KV-CACHE")
print(f"Без кеша: {median_without:.3f} сек")
print(f"С кешем: {median_with:.3f} сек")
print(f"Ускорение: {kv_speedup:.2f}x")

print("\nCPU VS GPU")
print(f"CPU: {cpu_median:.3f} мс")
print(f"GPU: {gpu_median:.3f} мс")
print(f"Ускорение: {gpu_speedup:.2f}x")

print("\nГрафики успешно созданы!")
print("1. kv_cache_comparison.png")
print("2. cpu_gpu_comparison.png")
