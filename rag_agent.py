from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import base64
import uuid
import requests
import os
from dotenv import load_dotenv # Импортируем load_dotenv

# --- Константы (копируем или импортируем из retriever_test) ---
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "history_archive"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Убедитесь, что модель та же, что и при загрузке

# --- Настройки для GigaChat ---
GIGACHAT_SYSTEM_ROLE = """Ты — Архивариус, дружелюбный и очень осведомленный ассистент. 
Твоя задача — отвечать на вопросы пользователя, основываясь на предоставленных текстовых фрагментах из исторических документов кафедры. 
Пожалуйста, давай точные ответы. **Всегда указывай источник информации (например, 'Источник: [имя_файла, фрагмент N]'), из которого взяты ключевые сведения для твоего ответа. Если ответ собирается из нескольких источников, укажи их все.** 
Если информации в предоставленных фрагментах недостаточно для ответа, честно сообщи об этом."""

def search_qdrant_for_context(query_text: str, client: QdrantClient, embedding_model: SentenceTransformer, collection_name: str, top_k: int = 3):
    """
    Ищет контекст в Qdrant. (Эта функция аналогична search_in_qdrant из retriever_test.py)
    """
    # print(f"\nПоисковый запрос для Qdrant: '{query_text}'")
    # print(f"Генерация эмбеддинга для запроса...")
    try:
        query_embedding = embedding_model.encode(query_text)
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        return []

    # print(f"Выполнение поиска в Qdrant (коллекция: '{collection_name}', top_k={top_k})...")
    try:
        search_results = client.search( # В будущем заменить на query_points
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        # print("Поиск в Qdrant завершен.")
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
        context_str += f""""---
Источник {i+1}: {source_file}, Фрагмент: {chunk_index} (Схожесть: {score:.4f})
Текст: {text_chunk}
"""
    context_str += """---
"""
    return context_str

def ask_gigachat_with_context(user_question: str, context_str: str, system_role: str, client_id: str, client_secret: str, dialog_history: list):
    """
    Формирует промпт для GigaChat, включая историю диалога, и вызывает API.
    """
    
    # Формируем текущее сообщение пользователя с найденным контекстом
    current_user_message_content = f"{context_str}\n\nПожалуйста, ответь на следующий вопрос: {user_question}"
    current_user_message = {"role": "user", "content": current_user_message_content}

    # --- Начало Few-Shot примеров ---
    # Эти примеры помогут модели лучше понять ожидаемый формат ответа, особенно цитирование.
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
        {
            "role": "user",
            "content": """Вот информация, найденная в архивах:
---
Источник 1: ОбщаяИсторияКафедры.txt, Фрагмент: 15 (Схожесть: 0.9200)
Текст: В 1953 году Сергей Николаевич становится заведующим кафедрой дифференциальных уравнений.
---
Источник 2: БиографияИванова.txt, Фрагмент: 3 (Схожесть: 0.8800)
Текст: Иванов П.П. начал работать на кафедре в 1950 году.
---

Пожалуйста, ответь на следующий вопрос: Кто стал заведующим кафедрой в 1953 году и когда Иванов начал там работать?"""
        },
        {
            "role": "assistant",
            "content": "В 1953 году Сергей Николаевич становится заведующим кафедрой дифференциальных уравнений (Источник: ОбщаяИсторияКафедры.txt, Фрагмент: 15). Иванов П.П. начал работать на кафедре в 1950 году (Источник: БиографияИванова.txt, Фрагмент: 3)."
        }
    ]
    # --- Конец Few-Shot примеров ---

    # Собираем полный список сообщений для API
    # Системная роль + Few-shot примеры + предыдущая история диалога + текущее сообщение пользователя
    api_messages = [
        {"role": "system", "content": system_role}
    ]
    api_messages.extend(few_shot_examples) # Добавляем few-shot примеры
    api_messages.extend(dialog_history) # Добавляем предыдущие user/assistant сообщения
    api_messages.append(current_user_message) # Добавляем текущий запрос пользователя
    
    # Собираем промпт для вывода в консоль (для наглядности)
    # Этот full_prompt не передается в API напрямую, но показывает, что видит LLM
    full_prompt_display = f"[СИСТЕМНАЯ РОЛЬ]\n{system_role}\n\n"
    
    full_prompt_display += "--- НАЧАЛО FEW-SHOT ПРИМЕРОВ ---\n"
    for i in range(0, len(few_shot_examples), 2):
        if few_shot_examples[i]["role"] == "user" and i + 1 < len(few_shot_examples) and few_shot_examples[i+1]["role"] == "assistant":
            full_prompt_display += f"[ПРИМЕР ВОПРОСА ПОЛЬЗОВАТЕЛЯ #{i//2 + 1}]\n{few_shot_examples[i]['content']}\n\n"
            full_prompt_display += f"[ПРИМЕР ОТВЕТА АССИСТЕНТА #{i//2 + 1}]\n{few_shot_examples[i+1]['content']}\n\n"
    full_prompt_display += "--- КОНЕЦ FEW-SHOT ПРИМЕРОВ ---\n\n"

    for msg in dialog_history:
        if msg["role"] == "user":
            full_prompt_display += f"[ПРЕДЫДУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ]\n{msg['content']}\n\n"
        elif msg["role"] == "assistant":
            full_prompt_display += f"[ПРЕДЫДУЩИЙ ОТВЕТ АРХИВАРИУСА]\n{msg['content']}\n\n"
    full_prompt_display += f"[КОНТЕКСТ ДЛЯ ТЕКУЩЕГО ВОПРОСА]\n{context_str}\n\n"
    full_prompt_display += f"[ТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ]\nПожалуйста, ответь на следующий вопрос: {user_question}"

    print("\n--- Промпт для GigaChat (с историей) ---")
    print(full_prompt_display)
    print("-----------------------------------------\n")
    
    # 1. Получение токена доступа
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
        # print("Запрос токена доступа GigaChat...") # Можно закомментировать для краткости вывода
        response = requests.post(token_url, headers=token_headers, data=token_payload, verify=False) 
        response.raise_for_status() 
        token_data = response.json()
        access_token = token_data.get("access_token")
        # expires_at = token_data.get("expires_at") 
        # print(f"Токен доступа получен. Действителен до: {expires_at}")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении токена GigaChat: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Ответ сервера: {e.response.text}")
        return "К сожалению, не удалось получить токен для GigaChat."
    except Exception as e:
        print(f"Неожиданная ошибка при получении токена: {e}")
        return "К сожалению, не удалось получить токен для GigaChat."
    if not access_token:
        return "Не удалось извлечь токен доступа GigaChat."

    # 2. Вызов модели GigaChat для генерации ответа
    chat_url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions" 
    
    chat_payload = {
        "model": "GigaChat:latest", 
        "messages": api_messages, # <--- ИСПОЛЬЗУЕМ api_messages С ПОЛНОЙ ИСТОРИЕЙ
        "temperature": 0.7, 
    }
    chat_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    answer = "Ошибка при обращении к GigaChat."
    try:
        # print("Запрос ответа от GigaChat...") # Можно закомментировать
        response = requests.post(chat_url, headers=chat_headers, json=chat_payload, verify=False) 
        response.raise_for_status()
        chat_data = response.json()
        if chat_data.get("choices") and chat_data["choices"][0].get("message"):
            answer = chat_data["choices"][0]["message"]["content"]
        else:
            print("Неожиданный формат ответа от GigaChat.")
            print(chat_data)
            answer = "Получен неожиданный формат ответа от GigaChat."
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к GigaChat: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Ответ сервера: {e.response.text}")
        answer = "К сожалению, произошла ошибка при обращении к GigaChat."
    except Exception as e:
        print(f"Неожиданная ошибка при обработке ответа GigaChat: {e}")
        answer = "Произошла неожиданная ошибка при получении ответа от GigaChat."
    
    print("\n--- Ответ от GigaChat ---")
    print(answer)
    print("------------------------------------")
    return answer, current_user_message # Возвращаем также сообщение пользователя для добавления в историю

def main():
    load_dotenv() 
    gigachat_client_id = os.getenv("GIGACHAT_CLIENT_ID")
    gigachat_client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
    if not gigachat_client_id or not gigachat_client_secret:
        print("Ошибка: Переменные окружения GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не установлены.")
        return

    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
        print(f"Успешное подключение к Qdrant по адресу {QDRANT_HOST}:{QDRANT_GRPC_PORT} (gRPC)")
    except Exception as e:
        print(f"Не удалось подключиться к Qdrant: {e}")
        return

    try:
        print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
        print("Модель эмбеддингов успешно загружена.")
    except Exception as e:
        print(f"Не удалось загрузить модель эмбеддингов: {e}")
        return

    dialog_history = [] # Инициализируем историю диалога
    # Ограничение на количество сообщений в истории (пар вопрос-ответ), чтобы не превысить лимит токенов LLM
    MAX_HISTORY_TURNS = 3 

    print("\nДобро пожаловать в Архивариус! Задавайте ваши вопросы. Для выхода введите 'выход' или 'exit'.")

    while True:
        user_query = input("\nВы: ")
        if user_query.lower() in ['выход', 'exit']:
            print("До свидания!")
            break
        
        if not user_query.strip():
            continue

        # print(f"\nПользовательский вопрос: {user_query}") # Уже выводится в input

        retrieved_chunks = search_qdrant_for_context(
            query_text=user_query, 
            client=qdrant_client, 
            embedding_model=sbert_model, 
            collection_name=COLLECTION_NAME, 
            top_k=3 
        )

        context_for_llm = format_context_for_llm(retrieved_chunks)
        
        # Передаем ограниченную историю диалога
        # Берем последние MAX_HISTORY_TURNS * 2 сообщений (вопрос + ответ)
        limited_dialog_history = dialog_history[-(MAX_HISTORY_TURNS*2):]

        llm_answer, current_user_msg_for_history = ask_gigachat_with_context(
            user_query, 
            context_for_llm, 
            GIGACHAT_SYSTEM_ROLE, 
            gigachat_client_id, 
            gigachat_client_secret,
            limited_dialog_history # Передаем ограниченную историю
        )
        
        # Добавляем текущий вопрос пользователя и ответ LLM в историю
        dialog_history.append(current_user_msg_for_history) # Добавляем отформатированное сообщение пользователя
        if llm_answer and not llm_answer.startswith("К сожалению") and not llm_answer.startswith("Ошибка") :
             dialog_history.append({"role": "assistant", "content": llm_answer})

        # Опционально: обрезать историю, если она слишком длинная (хотя мы уже передаем limited_dialog_history)
        # if len(dialog_history) > MAX_HISTORY_TURNS * 2 + 2: # +2 для запаса
        #     dialog_history = dialog_history[-(MAX_HISTORY_TURNS*2):]

if __name__ == "__main__":
    main() 