import torch


# выводим версию установленного PyTorch
print("PyTorch:", torch.__version__)

# проверяем версию CUDA, используемую PyTorch
print("CUDA:", torch.version.cuda)

# проверяем доступность видеокарты
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    # получаем название видеокарты
    print("GPU:", torch.cuda.get_device_name(0))

    # создаём матрицу на видеокарте
    matrix = torch.rand(1000, 1000, device="cuda")

    print("Matrix device:", matrix.device)
    print("Matrix shape:", matrix.shape)
else:
    print("GPU is not available")
