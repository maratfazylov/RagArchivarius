from qdrant_client import QdrantClient
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
def search_qdrant_for_context(query_text: str, client: QdrantClient, embedding_model: SentenceTransformer, collection_name: str, top_k: int = 3):
    """
    Ищет контекст в Qdrant.
    """
    try:
        query_embedding = embedding_model.encode(query_text)
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        return []

    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
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
        return "Контекст не найден."
    
    context_str = """Вот информация, найденная в архивах:
"""
    for i, hit in enumerate(retrieved_chunks):
        source_file = hit.payload.get('source_file', 'N/A')
        chunk_index = hit.payload.get('chunk_index', 'N/A')
        text_chunk = hit.payload.get('text', 'N/A')
        score = hit.score 
        context_str += f"""---
Источник {i+1}: {source_file}, Фрагмент: {chunk_index} (Схожесть: {score:.4f})
Текст: {text_chunk}
"""
    context_str += """---
"""
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
Источник 1: Аксентьев.txt, Фрагмент: 0 (Схожесть: 0.8500)
Текст: Леонид Александрович Аксентьев родился 1 марта 1932 года в г. Баку...
---

Пожалуйста, ответь на следующий вопрос: Когда родился Аксентьев?"""
        },
        {
            "role": "assistant",
            "content": "Леонид Александрович Аксентьев родился 1 марта 1932 года (Источник: Аксентьев.txt, Фрагмент: 0)."
        },
        # Можно добавить больше примеров, если нужно
    ]

    api_messages = [{"role": "system", "content": system_role}]
    api_messages.extend(few_shot_examples)
    api_messages.extend(dialog_history)
    api_messages.append(current_user_message)
    
    # Для отладки можно распечатать, что передается в GigaChat
    # print("\n--- Сообщения для GigaChat API ---")
    # for msg in api_messages:
    #    print(f"- Роль: {msg['role']}, Контент: {msg['content'][:200]}...") # Выводим только начало контента
    # print("-----------------------------------\n")

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
    
    # print("\n--- Ответ от GigaChat ---") # Для отладки
    # print(answer)
    # print("------------------------------------")
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
    max_history_turns_llm: int = 3
    ):
    """
    Основная RAG-функция: получает запрос, ищет контекст, вызывает LLM.
    """
    print(f"Получен запрос: '{user_query}'")
    
    # 1. Поиск контекста в Qdrant
    # В этой версии мы не используем переписывание запроса, передаем user_query напрямую
    retrieved_chunks = search_qdrant_for_context(
        query_text=user_query, 
        client=qdrant_client, 
        embedding_model=sbert_model, 
        collection_name=collection_name, 
        top_k=top_k_retriever
    )

    # 2. Форматирование контекста
    context_for_llm = format_context_for_llm(retrieved_chunks)
    
    # 3. Ограничение истории диалога для LLM
    limited_dialog_history = dialog_history[-(max_history_turns_llm*2):]

    # 4. Запрос к GigaChat
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

    while True:
        user_query_original = input("\nВы: ")
        if user_query_original.lower() in ['выход', 'exit']:
            print("До свидания!")
            break
        
        if not user_query_original.strip():
            continue

        llm_answer, current_user_msg_for_history = get_rag_response(
            user_query=user_query_original,
            dialog_history=dialog_history,
            qdrant_client=qdrant_client,
            sbert_model=sbert_model,
            gigachat_client_id=gigachat_client_id,
            gigachat_client_secret=gigachat_client_secret,
            # system_role, collection_name, top_k_retriever, max_history_turns_llm - используются значения по умолчанию
        )
        
        dialog_history.append(current_user_msg_for_history) 
        if llm_answer and not llm_answer.startswith("К сожалению") and not llm_answer.startswith("Ошибка"):
             dialog_history.append({"role": "assistant", "content": llm_answer})
        
        # Ограничиваем историю для консольного режима
        if len(dialog_history) > MAX_HISTORY_TURNS_CONSOLE * 2 + 2:
            dialog_history = dialog_history[-(MAX_HISTORY_TURNS_CONSOLE * 2):]


if __name__ == "__main__":
    main() 