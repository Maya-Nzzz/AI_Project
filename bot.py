import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from generate import TextGenerator

# Логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация генератора
generator = TextGenerator(
    model_path="model",
    tokenizer_path="tokenizer"
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id

    # Генерируем ответ
    response = generator.generate(
        user_text,
        max_length=30,
        temperature=0.1,
        top_k=10,
        repetition_penalty=1.2,
        stop_sequences=["\n", "---", '.', '!', '?']
    )

    # Отправляем ответ пользователю
    await context.bot.send_message(chat_id=chat_id, text=response)

def main():
    TOKEN = "7631944822:AAFrO1b-cKOD4c4t_pkTxKCKE9Ytv-kFLOY"

    app = ApplicationBuilder().token(TOKEN).build()

    # Обработчик всех текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
