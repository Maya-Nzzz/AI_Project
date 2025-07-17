from tokenizers import ByteLevelBPETokenizer
import json

# Загружаем очищенные сообщения
with open("data/cleaned_messages.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# Сохраняем в временный файл для обучения
with open("data/temp_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(texts))

# Обучаем токенизатор
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/temp_corpus.txt"],
    vocab_size=10_000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
)

test_text = "Привет, как дела?"
tokens = tokenizer.encode(test_text).tokens
corrected_tokens = []
for t in tokens:
    try:
        # Преобразуем строку в байты по Latin-1
        bytes_seq = t.encode('latin1')
        # Декодируем байты обратно в utf-8
        corrected = bytes_seq.decode('utf-8')
        corrected_tokens.append(corrected)
    except Exception as e:
        # В случае ошибки оставляем исходный токен
        corrected_tokens.append(t)
print(corrected_tokens)

# Сохраняем токенизатор
tokenizer.save_model("tokenizer")