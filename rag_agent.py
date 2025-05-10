from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import base64
import uuid
import requests
import os
from dotenv import load_dotenv

# --- Константы ---
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "history_archive"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- Настройки для GigaChat ---
GIGACHAT_SYSTEM_ROLE = """Ты — Архивариус, дружелюбный и очень осведомленный ассистент. 
Твоя задача — отвечать на вопросы пользователя, основываясь на предоставленных текстовых фрагментах из исторических документов кафедры. 
Пожалуйста, давай точные ответы. **Всегда указывай источник информации (например, 'Источник: [имя_файла, фрагмент N]'), из которого взяты ключевые сведения для твоего ответа. Если ответ собирается из нескольких источников, укажи их все.** 
Если информации в предоставленных фрагментах недостаточно для ответа, честно сообщи об этом."""

# --- Функции инициализации ---
def initialize_qdrant_client():
    """Инициализирует и возвращает клиент Qdrant."""
    try:
        client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
        print(f"Успешное подключение к Qdrant по адресу {QDRANT_HOST}:{QDRANT_GRPC_PORT} (gRPC)")
        return client
    except Exception as e:
        print(f"Не удалось подключиться к Qdrant: {e}")
        raise  # Перевыбрасываем исключение, чтобы вызывающий код мог его обработать

def initialize_sbert_model():
    """Инициализирует и возвращает модель SentenceTransformer."""
    try:
        print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        print("Модель эмбеддингов успешно загружена.")
        return model
    except Exception as e:
        print(f"Не удалось загрузить модель эмбеддингов: {e}")
        raise # Перевыбрасываем исключение

# --- Основные функции RAG ---
def search_qdrant_for_context(
    query_text: str, 
    client: QdrantClient, 
    embedding_model: SentenceTransformer, 
    collection_name: str, 
    top_k: int = 3,
    person_tag: str | None = None
    ):
    """
    Ищет контекст в Qdrant. 
    Если person_tag указан, фильтрует по нему.
    """
    try:
        query_embedding = embedding_model.encode(query_text)
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        return []

    query_filter = None
    if person_tag:
        print(f"Применяется фильтр по person_tag: '{person_tag}'")
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="person_tag",
                    match=qdrant_models.MatchValue(value=person_tag)
                )
            ]
        )

    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )
        return search_results
    except Exception as e:
        print(f"Ошибка при поиске в Qdrant: {e}")
        return []

def format_context_for_llm(retrieved_chunks):
    """
    Форматирует найденные чанки в строку контекста для LLM.
    """
    if not retrieved_chunks:
        print("\n--- Контекст для LLM ---")
        print("Контекст не найден.")
        print("-----------------------")
        return "Контекст не найден."
    
    context_str = """Вот информация, найденная в архивах:
"""
    for i, hit in enumerate(retrieved_chunks):
        source_file = hit.payload.get('source_file', 'N/A')
        chunk_index = hit.payload.get('chunk_index', 'N/A')
        text_chunk = hit.payload.get('text', 'N/A')
        person_payload_tag = hit.payload.get('person_tag', 'N/A') 
        score = hit.score 
        context_str += f"""---
Источник {i+1}: {source_file} (Персона: {person_payload_tag}), Фрагмент: {chunk_index} (Схожесть: {score:.4f})
Текст: {text_chunk}
"""
    context_str += """---
"""
    print("\n--- Контекст для LLM ---")
    print(context_str)
    print("-----------------------")
    return context_str

def ask_gigachat_with_context(user_question: str, context_str: str, system_role: str, client_id: str, client_secret: str, dialog_history: list):
    """
    Формирует промпт для GigaChat, включая историю диалога, и вызывает API.
    Возвращает ответ LLM и сообщение пользователя для истории.
    """
    current_user_message_content = f"{context_str}\n\nПожалуйста, ответь на следующий вопрос: {user_question}"
    current_user_message = {"role": "user", "content": current_user_message_content}

    few_shot_examples = [
        {
            "role": "user",
            "content": """Вот информация, найденная в архивах:
---
Источник 1: Аксентьев.txt (Персона: Аксентьев), Фрагмент: 0 (Схожесть: 0.8500)
Текст: Леонид Александрович Аксентьев родился 1 марта 1932 года в г. Баку...
---

Пожалуйста, ответь на следующий вопрос: Когда родился Аксентьев?"""
        },
        {
            "role": "assistant",
            "content": "Леонид Александрович Аксентьев родился 1 марта 1932 года (Источник: Аксентьев.txt (Персона: Аксентьев), Фрагмент: 0)."
        },
    ]

    api_messages = [{"role": "system", "content": system_role}]
    api_messages.extend(few_shot_examples)
    api_messages.extend(dialog_history) # Добавляем предыдущие user/assistant сообщения
    api_messages.append(current_user_message) # Добавляем текущий запрос пользователя
    
    # --- Логгирование полного промпта для GigaChat ---
    print("\n--- Сообщения для GigaChat API (структура) ---")
    for i, msg in enumerate(api_messages):
       print(f"Сообщение #{i} (Роль: {msg['role']}):")
       # Выводим только начало длинных сообщений, чтобы не засорять лог
       content_to_log = msg['content']
       if len(content_to_log) > 300: # Показывать первые 150 и последние 150 символов
           print(f"  Начало: {content_to_log[:150]}...")
           print(f"  Конец: ...{content_to_log[-150:]}")
       else:
           print(f"  Контент: {content_to_log}")
    print("----------------------------------------------\n")
    # --- Конец логгирования --- 

    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth" 
    token_payload = {'scope': 'GIGACHAT_API_PERS'} 
    auth_string = f"{client_id}:{client_secret}"
    base64_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    token_headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': str(uuid.uuid4()), 
      'Authorization': f'Basic {base64_auth_string}'
    }
    access_token = None
    try:
        response = requests.post(token_url, headers=token_headers, data=token_payload, verify=False) 
        response.raise_for_status() 
        token_data = response.json()
        access_token = token_data.get("access_token")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении токена GigaChat: {e}")
        return "К сожалению, не удалось получить токен для GigaChat.", current_user_message
    if not access_token:
        return "Не удалось извлечь токен доступа GigaChat.", current_user_message

    chat_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions" 
    chat_payload = {
        "model": "GigaChat:latest", 
        "messages": api_messages,
        "temperature": 0.7, 
    }
    chat_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    answer = "Ошибка при обращении к GigaChat."
    try:
        response = requests.post(chat_url, headers=chat_headers, json=chat_payload, verify=False) 
        response.raise_for_status()
        chat_data = response.json()
        if chat_data.get("choices") and chat_data["choices"][0].get("message"):
            answer = chat_data["choices"][0]["message"]["content"]
        else:
            print("Неожиданный формат ответа от GigaChat.")
            answer = "Получен неожиданный формат ответа от GigaChat."
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к GigaChat: {e}")
        answer = "К сожалению, произошла ошибка при обращении к GigaChat."
    
    return answer, current_user_message

def get_rag_response(
    user_query: str, 
    dialog_history: list,
    qdrant_client: QdrantClient, 
    sbert_model: SentenceTransformer,
    gigachat_client_id: str,
    gigachat_client_secret: str,
    system_role: str = GIGACHAT_SYSTEM_ROLE,
    collection_name: str = COLLECTION_NAME,
    top_k_retriever: int = 5,
    max_history_turns_llm: int = 3,
    person_tag: str | None = None
    ):
    """
    Основная RAG-функция: получает запрос, ищет контекст, вызывает LLM.
    """
    print(f"Получен запрос: '{user_query}', Фильтр по персоне: '{person_tag if person_tag else "Нет"}'")
    
    retrieved_chunks = search_qdrant_for_context(
        query_text=user_query, 
        client=qdrant_client, 
        embedding_model=sbert_model, 
        collection_name=collection_name, 
        top_k=top_k_retriever,
        person_tag=person_tag
    )

    context_for_llm = format_context_for_llm(retrieved_chunks)
    limited_dialog_history = dialog_history[-(max_history_turns_llm*2):]

    llm_answer, user_message_for_history = ask_gigachat_with_context(
        user_query, 
        context_for_llm, 
        system_role, 
        gigachat_client_id, 
        gigachat_client_secret,
        limited_dialog_history
    )
    
    print(f"Ответ LLM: {llm_answer}")
    return llm_answer, user_message_for_history

# --- Основная функция для консольного запуска ---
def main():
    load_dotenv() 
    gigachat_client_id = os.getenv("GIGACHAT_CLIENT_ID")
    gigachat_client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    if not gigachat_client_id or not gigachat_client_secret:
        print("Ошибка: Переменные окружения GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не установлены.")
        return

    try:
        qdrant_client = initialize_qdrant_client()
        sbert_model = initialize_sbert_model()
    except Exception as e:
        # Сообщение об ошибке уже выведено функциями инициализации
        return

    dialog_history = [] 
    MAX_HISTORY_TURNS_CONSOLE = 3 

    print("\nДобро пожаловать в Архивариус (консольная версия)! Задавайте ваши вопросы. Для выхода введите 'выход' или 'exit'.")
    print("Для поиска по конкретной персоне, попробуйте включить ее имя в запрос, например: 'Биография Аксентьев'")

    # Для простоты тестирования в консоли, определим небольшой список тегов персон вручную
    # В боте это будет делаться динамически
    known_person_tags_console_test = ["Аксентьев", "Шуликовский Валентин Иванович", "Чеботарев Николай Григорьевич"] # Добавьте несколько из ваших файлов

    while True:
        user_query_original = input("\nВы: ")
        if user_query_original.lower() in ['выход', 'exit']:
            print("До свидания!")
            break
        
        if not user_query_original.strip():
            continue
        
        # Простая логика для извлечения тега персоны для консольного теста
        current_person_tag_filter = None
        for tag_candidate in known_person_tags_console_test:
            # Ищем полное совпадение имени/тега в запросе (можно улучшить до поиска фамилии и т.д.)
            if tag_candidate.lower() in user_query_original.lower():
                current_person_tag_filter = tag_candidate
                print(f"(Консольный тест: Обнаружен тег персоны '{current_person_tag_filter}' в запросе)")
                break

        llm_answer, current_user_msg_for_history = get_rag_response(
            user_query=user_query_original,
            dialog_history=dialog_history,
            qdrant_client=qdrant_client,
            sbert_model=sbert_model,
            gigachat_client_id=gigachat_client_id,
            gigachat_client_secret=gigachat_client_secret,
            person_tag=current_person_tag_filter
        )
        
        dialog_history.append(current_user_msg_for_history) 
        if llm_answer and not llm_answer.startswith("К сожалению") and not llm_answer.startswith("Ошибка"):
             dialog_history.append({"role": "assistant", "content": llm_answer})
        
        # Ограничиваем историю для консольного режима
        if len(dialog_history) > MAX_HISTORY_TURNS_CONSOLE * 2 + 2:
            dialog_history = dialog_history[-(MAX_HISTORY_TURNS_CONSOLE * 2):]


if __name__ == "__main__":
    main() 