import json
import re
import pandas as pd
from typing import List, Dict, Set, Optional


INPUT_FILE = "result.json"  # Файл экспорта из Telegram
OUTPUT_FILE = "cleaned_messages.json"  # Выходной файл
TARGET_USER = "Laert"  # Имя пользователя (как в Telegram) или его user_id
LANGUAGE = "ru"  # 'ru' или 'en' (для списка матов)

# словарь ругательств (можно пополнить)
BAD_WORDS: Dict[str, Set[str]] = {
    "ru": {
        
    },
    "en": {
        
    }
}

def load_chats(file_path: str) -> List[Dict]:
    """Загружает JSON-экспорт и возвращает список чатов."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chats", {}).get("list", [])

def filter_user_messages(chat: Dict, target_user: str) -> List[str]:
    """Извлекает сообщения target_user из чата."""
    user_messages = []
    for msg in chat.get("messages", []):
        if isinstance(msg, dict) and msg.get("from") == target_user:
            text = msg.get("text", "")
            if isinstance(text, str):
                user_messages.append(text)
            elif isinstance(text, list): 
                text_parts = []
                for part in text:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                user_messages.append("".join(text_parts))
    return user_messages

def clean_text(text: str, bad_words: Set[str]) -> str:
    """Очищает текст от мусора и матов."""
    # Удаляем ссылки, хэштеги, email, юзернеймы
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+|\S+@\S+", "", text, flags=re.IGNORECASE)
    # Удаляем спецсимволы (кроме букв, цифр и базовой пунктуации)
    text = re.sub(r"[^\w\s.,!?а-яА-ЯёЁ]", "", text)
    # Удаляем маты (регистронезависимо)
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in bad_words]
    return " ".join(cleaned_words)

def main():
    # Загружаем чаты
    chats = load_chats(INPUT_FILE)
    print(f"Найдено чатов: {len(chats)}")
    
    # Собираем все сообщения пользователя
    bad_words = BAD_WORDS.get(LANGUAGE, set())
    all_cleaned_messages = []
    
    for chat in chats:
        chat_name = chat.get("name", "Без названия")
        user_messages = filter_user_messages(chat, TARGET_USER)
        cleaned_messages = [clean_text(msg, bad_words) for msg in user_messages]
        cleaned_messages = [msg for msg in cleaned_messages if msg.strip()]
        cleaned_messages = [msg for msg in cleaned_messages if len(msg.split()) >= 3]
        cleaned_messages = list(set(cleaned_messages))
        
        if cleaned_messages:
            print(f"Чат: {chat_name} | Сообщений: {len(cleaned_messages)}")
            all_cleaned_messages.extend(cleaned_messages)
    
    # Сохраняем результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_cleaned_messages, f, ensure_ascii=False, indent=2)
    
    print(f"\nИтого сохранено сообщений от {TARGET_USER}: {len(all_cleaned_messages)}")

if __name__ == "__main__":
    main()