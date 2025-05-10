import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Импортируем необходимые функции и переменные из rag_agent
import rag_agent 

# Загрузка переменных окружения
load_dotenv()

# --- Конфигурация --- 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # ЗАГРУЖАЕМ ИЗ .ENV
GIGACHAT_CLIENT_ID = os.getenv("GIGACHAT_CLIENT_ID")
GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")

# Словарь для хранения истории диалогов для каждого пользователя
user_dialog_histories = {}
MAX_HISTORY_TURNS_TELEGRAM = 3 

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные для RAG компонентов ---
qdrant_client_global = None
sbert_model_global = None
known_person_names_global = [] # <--- Глобальный список для имен персон

def load_known_person_names(source_dir: str) -> list[str]:
    """Сканирует директорию с текстовыми файлами и извлекает имена персон (теги)."""
    person_names = []
    if not os.path.exists(source_dir):
        logger.error(f"Директория с исходными текстами '{source_dir}' не найдена для загрузки имен персон.")
        return person_names
    try:
        for filename_with_ext in os.listdir(source_dir):
            if filename_with_ext.endswith(".txt"):
                person_tag = os.path.splitext(filename_with_ext)[0]
                person_names.append(person_tag)
        logger.info(f"Загружено {len(person_names)} имен персон: {person_names}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке имен персон из директории '{source_dir}': {e}")
    return person_names

def initialize_all_components_for_bot():
    """Инициализирует все компоненты: RAG и список имен персон."""
    global qdrant_client_global, sbert_model_global, known_person_names_global
    try:
        logger.info("Инициализация RAG компонентов для Telegram бота...")
        qdrant_client_global = rag_agent.initialize_qdrant_client()
        sbert_model_global = rag_agent.initialize_sbert_model()
        logger.info("RAG компоненты успешно инициализированы.")
        
        # Загружаем имена персон, используя TEXT_SOURCE_DIR из rag_agent, если он там определен
        # Или можно жестко задать путь к raw_text здесь
        # Предполагаем, что rag_agent.TEXT_SOURCE_DIR существует и указывает на правильную папку
        # Если такого атрибута нет, нужно будет передать путь явно, например, "raw_text"
        # Для безопасности, проверим наличие атрибута или используем константу, если она есть в telegram_bot
        source_dir_for_names = getattr(rag_agent, 'TEXT_SOURCE_DIR', "raw_text") 
        known_person_names_global = load_known_person_names(source_dir_for_names)
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при полной инициализации компонентов бота: {e}")
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение, когда пользователь вводит команду /start."""
    user = update.effective_user
    user_dialog_histories[user.id] = [] # Очищаем историю для нового старта
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я Архивариус. Задавай мне вопросы по истории кафедры.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение с помощью, когда пользователь вводит команду /help."""
    user_dialog_histories[update.effective_user.id] = [] # Очищаем историю и для help
    help_text = (
        "Просто напиши мне свой вопрос, и я постараюсь найти на него ответ в архивах.\n"
        "Я помню контекст нашего с тобой разговора (последние несколько сообщений).\n"
        "Чтобы начать разговор заново (сбросить контекст), можешь использовать команду /start."
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовое сообщение от пользователя и отвечает с помощью RAG-системы."""
    user_id = update.effective_user.id
    user_query = update.message.text
    logger.info(f"Получен вопрос от user {user_id}: {user_query}")

    if not qdrant_client_global or not sbert_model_global:
        logger.error("RAG компоненты не инициализированы. Невозможно обработать запрос.")
        await update.message.reply_text("Извините, произошла ошибка на сервере. Попробуйте позже.")
        return
    if not GIGACHAT_CLIENT_ID or not GIGACHAT_CLIENT_SECRET:
        logger.error("Учетные данные GigaChat не настроены.")
        await update.message.reply_text("Извините, сервис временно недоступен (ошибка конфигурации). Попробуйте позже.")
        return

    # Получаем или инициализируем историю диалога для этого пользователя
    if user_id not in user_dialog_histories:
        user_dialog_histories[user_id] = []
    
    current_dialog_history = user_dialog_histories[user_id]

    # Определение тега персоны из запроса
    current_person_tag_filter = None
    if known_person_names_global: # Если список имен загружен
        for person_name_tag in known_person_names_global:
            # Простая проверка на вхождение (можно улучшить для более точного матчинга)
            if person_name_tag.lower() in user_query.lower():
                current_person_tag_filter = person_name_tag
                logger.info(f"Обнаружен тег персоны '{current_person_tag_filter}' в запросе.")
                break 

    try:
        await update.message.chat.send_action(action='typing') # Показываем, что бот печатает
        
        llm_answer, user_msg_for_history = rag_agent.get_rag_response(
            user_query=user_query,
            dialog_history=current_dialog_history, # Передаем текущую историю этого пользователя
            qdrant_client=qdrant_client_global,
            sbert_model=sbert_model_global,
            gigachat_client_id=GIGACHAT_CLIENT_ID,
            gigachat_client_secret=GIGACHAT_CLIENT_SECRET,
            system_role=rag_agent.GIGACHAT_SYSTEM_ROLE, # Используем роль из rag_agent
            person_tag=current_person_tag_filter, # <--- ПЕРЕДАЕМ ТЕГ
            top_k_retriever=5, # Можно настроить
            max_history_turns_llm=MAX_HISTORY_TURNS_TELEGRAM # Используем настройку для Telegram
        )

        # Обновляем историю диалога для пользователя
        current_dialog_history.append(user_msg_for_history)
        if llm_answer and not llm_answer.startswith("К сожалению") and not llm_answer.startswith("Ошибка"):
            current_dialog_history.append({"role": "assistant", "content": llm_answer})
        
        # Ограничиваем длину истории
        if len(current_dialog_history) > MAX_HISTORY_TURNS_TELEGRAM * 2 + 2: 
            user_dialog_histories[user_id] = current_dialog_history[-(MAX_HISTORY_TURNS_TELEGRAM*2):]
        else:
            user_dialog_histories[user_id] = current_dialog_history

        await update.message.reply_text(llm_answer)

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения от user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("Извините, при обработке вашего запроса произошла ошибка. Попробуйте еще раз.")

def main_bot() -> None:
    """Запускает бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в .env. Телеграм-бот не будет запущен.")
        return
    if not GIGACHAT_CLIENT_ID or not GIGACHAT_CLIENT_SECRET:
         logger.error("GIGACHAT_CLIENT_ID и/или GIGACHAT_CLIENT_SECRET не установлены в .env. Телеграм-бот не будет запущен.")
         return

    if not initialize_all_components_for_bot(): # Инициализируем компоненты при старте
        logger.error("Не удалось инициализировать компоненты бота. Телеграм-бот не будет запущен.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Запуск Telegram бота...")
    application.run_polling()

if __name__ == "__main__":
    main_bot() 