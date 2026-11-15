import math
import torch


# задаём исходную последовательность токенов
tokens = ["LLM", "предсказывает", "следующий", "токен"]

# задаём условные векторные представления токенов
embeddings = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

# фиксируем случайность для воспроизводимости
torch.manual_seed(42)

# определяем размерность векторов
d_model = embeddings.shape[1]
d_k = d_model

# создаём три матрицы проекций
w_q = torch.randn(d_model, d_k) / math.sqrt(d_model)
w_k = torch.randn(d_model, d_k) / math.sqrt(d_model)
w_v = torch.randn(d_model, d_model) / math.sqrt(d_model)

# вычисляем Query, Key и Value
q = embeddings @ w_q
k = embeddings @ w_k
v = embeddings @ w_v

# вычисляем оценки внимания между токенами
scores = q @ k.T / math.sqrt(d_k)

# создаём маску, запрещающую видеть будущие токены
n_tokens = len(tokens)

causal_mask = torch.triu(
    torch.ones(n_tokens, n_tokens, dtype=torch.bool),
    diagonal=1,
)

# скрываем будущие позиции
masked_scores = scores.masked_fill(
    causal_mask,
    float("-inf"),
)

# превращаем оценки внимания в вероятностные веса
attention_weights = torch.softmax(
    masked_scores,
    dim=-1,
)

# вычисляем выход механизма внимания
output = attention_weights @ v

# выводим результаты
torch.set_printoptions(precision=3, sci_mode=False)

print("Токены:", tokens)
print("\nМатрица Query:\n", q)
print("\nМатрица Key:\n", k)
print("\nМатрица Value:\n", v)

print("\nОценки внимания:\n", scores)
print("\nCausal Mask:\n", causal_mask)
print("\nВеса внимания:\n", attention_weights)
print("\nРезультат Self-Attention:\n", output)

# проверяем корректность вычислений
assert torch.allclose(
    attention_weights.sum(dim=-1),
    torch.ones(n_tokens),
)

assert torch.all(attention_weights[causal_mask] == 0)

print("\nПроверки Self-Attention пройдены")
