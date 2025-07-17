from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import torch
from torch.utils.data import Dataset
import json
import os

# Загрузка токенизатора
tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename="tokenizer/vocab.json",
    merges_filename="tokenizer/merges.txt"
)

# Подготовка датасета 
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])
        input_ids = encoding.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids.copy(), dtype=torch.long)
        }

# Загрузка данных
with open("data/cleaned_messages.json", "r", encoding="utf-8") as f:
    texts = json.load(f)
dataset = TextDataset(texts, tokenizer)

# Загрузка модели
model = GPT2LMHeadModel.from_pretrained("model")
    
# Настройка обучения
training_args = TrainingArguments(
    output_dir="results",
    overwrite_output_dir=False, 
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=3,
    logging_dir='logs'
)

# Обучение с обработкой ошибок
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Начинаем обучение...")
    trainer.train()
    
except Exception as e:
    print(f"Ошибка обучения: {str(e)}")
    # Сохраняем модель в случае ошибки
    model.save_pretrained("model_interrupted")
    raise

# Сохранение результатов
model.save_pretrained("model")
print("Обучение завершено. Модель сохранена в model")