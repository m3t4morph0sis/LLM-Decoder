import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# название модели на Hugging Face
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# выбираем GPU, если он доступен
device = "cuda" if torch.cuda.is_available() else "cpu"

# используем FP16 на GPU для экономии видеопамяти
dtype = torch.float16 if device == "cuda" else torch.float32

print("Устройство:", device)
print("Загружаем модель:", MODEL_NAME)

# загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# загружаем готовую языковую модель
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype
)

# переносим модель на выбранное устройство
model = model.to(device)

# переключаем модель в режим инференса
model.eval()

print("Модель загружена успешно!")


# задаём вопрос модели
messages = [
    {
        "role": "user",
        "content": "Объясни простыми словами, что такое Transformer Decoder."
    }
]

# превращаем диалог в последовательность токенов

# превращаем диалог в токены
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

# извлекаем тензор с идентификаторами токенов
input_ids = encoded["input_ids"]


print("Количество входных токенов:", input_ids.shape[1])

# генерируем ответ без вычисления градиентов
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

# отделяем новые токены от исходного запроса
new_tokens = output_ids[0][input_ids.shape[1]:]

# переводим токены обратно в обычный текст
answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\nОтвет модели:\n")
print(answer)
