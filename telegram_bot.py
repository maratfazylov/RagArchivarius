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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
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
known_person_names_global = [] # Глобальный список для имен персон

def load_known_person_names_from_rag_agent_qdrant(q_client: rag_agent.QdrantClient, collection: str) -> list[str]:
    """
    Извлекает уникальные 'person_tag' из Qdrant коллекции.
    """
    person_names = set()
    logger.info(f"Попытка загрузки уникальных имен персон из Qdrant коллекции '{collection}'...")
    try:
        # Используем scroll для получения всех точек с нужным payload
        # Обратите внимание, что для очень больших коллекций это может быть долго
        # и может потребоваться обработка next_page_offset
        offset = None
        while True:
            scroll_response, next_offset = q_client.scroll(
                collection_name=collection,
                scroll_filter=None, 
                limit=200, # Обрабатываем по 200 точек за раз
                offset=offset,
                with_payload=["person_tag"], 
                with_vectors=False
            )
            if not scroll_response:
                break
            
            for hit in scroll_response:
                if hit.payload and 'person_tag' in hit.payload and hit.payload['person_tag']:
                    person_names.add(hit.payload['person_tag'])
            
            if not next_offset:
                break
            offset = next_offset
        
        logger.info(f"Загружено {len(person_names)} уникальных имен персон из Qdrant: {list(person_names)[:10]}...") # Логируем первые 10
        return list(person_names)
    except Exception as e:
        logger.error(f"Ошибка при загрузке имен персон из Qdrant: {e}")
        # Возвращаем пустой список или ранее загруженные имена, если они были из другого источника
        return []


def initialize_all_components_for_bot():
    """Инициализирует все компоненты: RAG и список имен персон из Qdrant."""
    global qdrant_client_global, sbert_model_global, known_person_names_global
    try:
        logger.info("Инициализация RAG компонентов для Telegram бота...")
        qdrant_client_global = rag_agent.initialize_qdrant_client()
        sbert_model_global = rag_agent.initialize_sbert_model()
        logger.info("RAG компоненты успешно инициализированы.")
        
        # Загружаем имена персон напрямую из Qdrant
        if qdrant_client_global:
            known_person_names_global = load_known_person_names_from_rag_agent_qdrant(
                q_client=qdrant_client_global,
                collection=rag_agent.COLLECTION_NAME # Используем имя коллекции из rag_agent
            )
        else:
            logger.warning("Qdrant клиент не инициализирован, список имен персон не будет загружен из Qdrant.")
            known_person_names_global = [] # Запасной вариант, если Qdrant не доступен
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при полной инициализации компонентов бота: {e}", exc_info=True)
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение, когда пользователь вводит команду /start."""
    user = update.effective_user
    user_id = user.id
    user_dialog_histories[user_id] = [] # Очищаем историю для нового старта
    logger.info(f"Пользователь {user_id} ({user.username}) выполнил /start. История очищена.")
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Я Архивариус. Задавай мне вопросы по истории кафедры.",
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет сообщение с помощью, когда пользователь вводит команду /help."""
    user = update.effective_user
    user_id = user.id
    # Не сбрасываем историю по /help, чтобы пользователь мог продолжить диалог
    # user_dialog_histories[user_id] = [] 
    logger.info(f"Пользователь {user_id} ({user.username}) выполнил /help.")
    help_text = (
        "Просто напиши мне свой вопрос, и я постараюсь найти на него ответ в архивах.\n"
        "Я помню контекст нашего с тобой разговора (последние несколько сообщений).\n"
        "Чтобы начать разговор заново (сбросить контекст), можешь использовать команду /start."
    )
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовое сообщение от пользователя и отвечает с помощью RAG-системы."""
    user = update.effective_user
    user_id = user.id
    user_query = update.message.text
    logger.info(f"Получен вопрос от user {user_id} ({user.username}): '{user_query}'")

    if not qdrant_client_global or not sbert_model_global:
        logger.error("RAG компоненты не инициализированы. Невозможно обработать запрос.")
        await update.message.reply_text("Извините, произошла ошибка на сервере. Попробуйте позже.")
        return
    if not GIGACHAT_CLIENT_ID or not GIGACHAT_CLIENT_SECRET:
        logger.error("Учетные данные GigaChat не настроены.")
        await update.message.reply_text("Извините, сервис временно недоступен (ошибка конфигурации). Попробуйте позже.")
        return

    if user_id not in user_dialog_histories:
        logger.info(f"Для пользователя {user_id} не найдена история, инициализируем пустую.")
        user_dialog_histories[user_id] = []
    
    current_dialog_history = user_dialog_histories[user_id]

    # Логика определения current_person_tag_filter здесь больше не нужна,
    # так как get_query_details_from_local_llm в rag_agent теперь сама это делает
    # на основе known_person_names_list.

    try:
        await update.message.chat.send_action(action='typing') 
        
        llm_answer, user_msg_for_history = rag_agent.get_rag_response(
            user_query_original=user_query, # <--- ИСПРАВЛЕНО: правильное имя аргумента
            dialog_history=current_dialog_history, 
            qdrant_client=qdrant_client_global,
            sbert_model=sbert_model_global,
            gigachat_client_id=GIGACHAT_CLIENT_ID,
            gigachat_client_secret=GIGACHAT_CLIENT_SECRET,
            known_person_names_list=known_person_names_global, # <--- ПЕРЕДАЕМ СПИСОК ИЗВЕСТНЫХ ИМЕН
            # system_role_gigachat используется по умолчанию из rag_agent (GIGACHAT_SYSTEM_ROLE)
            # collection_name используется по умолчанию из rag_agent (COLLECTION_NAME)
            # person_tag больше не передается явно, LLM определяет его сама
            top_k_retriever=5, 
            max_history_turns_llm=MAX_HISTORY_TURNS_TELEGRAM 
        )

        current_dialog_history.append(user_msg_for_history)
        
        if llm_answer and not llm_answer.startswith("[Ошибка локальной LLM:") and \
           not llm_answer.startswith("К сожалению, не удалось получить токен для GigaChat") and \
           not llm_answer.startswith("Не удалось извлечь токен доступа GigaChat") and \
           not llm_answer.startswith("Ошибка при обращении к GigaChat") and \
           not llm_answer.startswith("Получен неожиданный формат ответа от GigaChat"):
            current_dialog_history.append({"role": "assistant", "content": llm_answer})
        
        # Ограничиваем длину истории
        if len(current_dialog_history) > MAX_HISTORY_TURNS_TELEGRAM * 2 + 4: # + запас на системные сообщения GigaChat
            user_dialog_histories[user_id] = current_dialog_history[-(MAX_HISTORY_TURNS_TELEGRAM*2):]
        else:
            user_dialog_histories[user_id] = current_dialog_history

        await update.message.reply_text(llm_answer)

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения от user {user_id} ('{user_query}'): {e}", exc_info=True)
        await update.message.reply_text("Извините, при обработке вашего запроса произошла внутренняя ошибка. Попробуйте еще раз.")

def main_bot() -> None:
    """Запускает бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в .env. Телеграм-бот не будет запущен.")
        return
    if not GIGACHAT_CLIENT_ID or not GIGACHAT_CLIENT_SECRET:
         logger.error("GIGACHAT_CLIENT_ID и/или GIGACHAT_CLIENT_SECRET не установлены в .env. Телеграм-бот не будет запущен.")
         return

    if not initialize_all_components_for_bot():
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