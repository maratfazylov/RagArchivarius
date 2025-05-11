import base64
import uuid
import requests
import os
from dotenv import load_dotenv
import json
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer


# --- Константы ---
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334
COLLECTION_NAME = "history_archive"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- Константы для локально запущенной LLM (через LM Studio или аналоги) ---
LOCAL_LLM_SERVER_IP = "192.168.1.7"  # IP-адрес вашего ПК, где запущена LLM
LOCAL_LLM_SERVER_PORT = 8000
LOCAL_LLM_CHAT_API_URL = f"http://{LOCAL_LLM_SERVER_IP}:{LOCAL_LLM_SERVER_PORT}/v1/chat/completions"

# Базовый промпт для анализа запроса. Список имен будет добавлен динамически.
BASE_LOCAL_LLM_QUERY_ANALYSIS_PROMPT = """Твоя задача — проанализировать запрос пользователя и выполнить две вещи:
1. Определить, является ли запрос осмысленным вопросом для поиска информации в базе знаний или это просто светская беседа/шутка/приветствие.
2. Если в запросе упоминается имя или фамилия личности, извлечь это имя.

{known_names_section}

Верни результат в формате JSON.
Если запрос — это осмысленный вопрос для поиска:
  {{"query_type": "question", "person_tag": "Имя Фамилия" или null, "refined_query": "исходный или немного уточненный запрос для поиска"}}
Если запрос — это светская беседа, приветствие, шутка и т.п. (не требует поиска в базе знаний):
  {{"query_type": "banter", "person_tag": null, "refined_query": "исходный запрос пользователя"}}

Постарайся нормализовать имя к именительному падежу, если это возможно.
Если ты идентифицировал личность из предоставленного списка известных имен, используй точное имя из этого списка в поле "person_tag".
Не добавляй ничего лишнего в refined_query, если запрос и так понятен. Если личность упоминается, refined_query должен содержать эту личность.

Примеры:
Запрос: "Расскажи про Аксентьева"
Результат: {{"query_type": "question", "person_tag": "Аксентьев", "refined_query": "Расскажи про Аксентьева"}}
Запрос: "Какая биография у Николая Лобачевского?"
Результат: {{"query_type": "question", "person_tag": "Николай Лобачевский", "refined_query": "Какая биография у Николая Лобачевского?"}}
Запрос: "Что известно о вкладе Шуликовского в науку?"
Результат: {{"query_type": "question", "person_tag": "Шуликовский", "refined_query": "Что известно о вкладе Шуликовского в науку?"}}
Запрос: "Когда была основана кафедра?"
Результат: {{"query_type": "question", "person_tag": null, "refined_query": "Когда была основана кафедра?"}}
Запрос: "Привет"
Результат: {{"query_type": "banter", "person_tag": null, "refined_query": "Привет"}}
Запрос: "Как дела?"
Результат: {{"query_type": "banter", "person_tag": null, "refined_query": "Как дела?"}}
"""

# --- Настройки для GigaChat ---
GIGACHAT_SYSTEM_ROLE = """Ты — Архивариус, дружелюбный и очень осведомленный ассистент.
Твоя задача — отвечать на вопросы пользователя, основываясь на предоставленных текстовых фрагментах из исторических документов кафедры.
Пожалуйста, давай точные ответы. **Всегда указывай источник информации (например, 'Источник: [имя_файла, фрагмент N]'), из которого взяты ключевые сведения для твоего ответа. Если ответ собирается из нескольких источников, укажи их все.**
Если информации в предоставленных фрагментах недостаточно для ответа, честно сообщи об этом."""

# --- Функция для вызова локальной LLM (например, через LM Studio) ---
def call_remote_local_llm_chat(user_query: str, dialog_history: list = None, system_prompt: str = "Ты — полезный ИИ-ассистент."):
    if dialog_history is None:
        dialog_history = []
    processed_messages = []
    for entry in dialog_history:
        if entry.get("role") != "system":
            processed_messages.append({"role": entry["role"], "content": entry["content"]})
    
    current_user_content = user_query
    if system_prompt and not processed_messages:
        current_user_content = f"{system_prompt}\n\nЗапрос пользователя:\n{user_query}"
        log_system_prompt = system_prompt.splitlines()[0] 
        if len(log_system_prompt) > 70: log_system_prompt = log_system_prompt[:67] + "..."
        print(f"(Системный промпт для локальной LLM '{log_system_prompt}' внедрен в первый запрос пользователя)")
            
    processed_messages.append({"role": "user", "content": current_user_content})
    
    payload = {
        "model": "local-model", 
        "messages": processed_messages,
        "temperature": 0.2, 
        "n_predict": 350, 
        "stream": False
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    print(f"\nОтправка запроса к локальной LLM ({LOCAL_LLM_CHAT_API_URL})...")
    try:
        response = requests.post(LOCAL_LLM_CHAT_API_URL, headers=headers, json=payload, timeout=30) 
        response.raise_for_status()
        llm_response_data = response.json()
        
        if llm_response_data.get("choices") and \
           len(llm_response_data["choices"]) > 0 and \
           llm_response_data["choices"][0].get("message") and \
           llm_response_data["choices"][0]["message"].get("content"):
            answer = llm_response_data["choices"][0]["message"]["content"].strip()
            print(f"\nОтвет от локальной LLM: {answer}")
            return answer
        else:
            error_detail = str(llm_response_data)[:500]
            print(f"Неожиданный формат ответа от локальной LLM: {error_detail}")
            return f"[Ошибка локальной LLM: Неожиданный формат ответа: {error_detail}]"
    except requests.exceptions.Timeout:
        print("Ошибка: Запрос к локальной LLM превысил время ожидания.")
        return "[Ошибка локальной LLM: Таймаут]"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к локальной LLM: {e}")
        if e.response is not None:
            print(f"Тело ответа сервера LLM (ошибка): {e.response.text}")
        return f"[Ошибка локальной LLM: Подключение: {e}]"
    except Exception as e:
        print(f"Неизвестная ошибка при работе с локальной LLM: {e}")
        return f"[Ошибка локальной LLM: Неизвестно: {e}]"

# --- Функция для извлечения деталей запроса с помощью локальной LLM ---
def get_query_details_from_local_llm(user_query: str, known_person_names: list = None):
    """
    Использует локальную LLM для извлечения типа запроса, тега персоны и, возможно, уточнения запроса.
    Принимает список известных имен для улучшения распознавания.
    Возвращает словарь.
    """
    
    known_names_section_str = ""
    if known_person_names and len(known_person_names) > 0:
        names_str = ", ".join(known_person_names)
        known_names_section_str = (
            f"Вот список известных личностей из базы знаний, на которые стоит обратить особое внимание: [{names_str}]. "
            "Если в запросе упоминается имя, похожее на одно из этих, и контекст соответствует, "
            "постарайся использовать точное имя из этого списка в поле 'person_tag'."
        )
        print(f"Добавляем в промпт LLM секцию с известными именами ({len(known_person_names)} имен). Первые несколько: {known_person_names[:3]}")

    current_system_prompt = BASE_LOCAL_LLM_QUERY_ANALYSIS_PROMPT.format(known_names_section=known_names_section_str)

    llm_response_str = call_remote_local_llm_chat(
        user_query,
        dialog_history=None, 
        system_prompt=current_system_prompt
    )
    
    default_response = {"query_type": "question", "person_tag": None, "refined_query": user_query}

    if llm_response_str and not llm_response_str.startswith("[Ошибка"):
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        if think_start_tag in llm_response_str:
            start_index = llm_response_str.find(think_start_tag)
            end_index_search_start = start_index + len(think_start_tag)
            end_index = llm_response_str.find(think_end_tag, end_index_search_start)
            if end_index != -1:
                json_candidate_after = llm_response_str[end_index + len(think_end_tag):].strip()
                if json_candidate_after.startswith("{") and json_candidate_after.endswith("}"):
                    llm_response_str = json_candidate_after
                    print(f"Извлечен JSON после блока <think>: {llm_response_str}")
                else:
                    json_candidate_before = llm_response_str[:start_index].strip()
                    if json_candidate_before.startswith("{") and json_candidate_before.endswith("}"):
                         llm_response_str = json_candidate_before
                         print(f"Извлечен JSON перед блоком <think>: {llm_response_str}")
                    else:
                        print(f"Найден блок <think>, но JSON не извлечен однозначно. Исходный ответ LLM: {llm_response_str[:200]}...")
            else: 
                print(f"Найден открывающий <think>, но нет закрывающего. Исходный ответ LLM: {llm_response_str[:200]}...")
        
        if llm_response_str.startswith("```json"):
            llm_response_str = llm_response_str.replace("```json", "").replace("```", "").strip()
        elif llm_response_str.startswith("```"):
             llm_response_str = llm_response_str.replace("```", "").strip()

        json_start = llm_response_str.find('{')
        json_end = llm_response_str.rfind('}')
        if json_start != -1 and json_end != -1 and json_start < json_end:
            llm_response_str = llm_response_str[json_start : json_end+1]
            print(f"Финальная изолированная JSON-строка для парсинга: {llm_response_str}")
        else:
            print(f"Не удалось найти четкий JSON-объект в строке '{llm_response_str[:200]}...' после очистки.")

        try:
            parsed_response = json.loads(llm_response_str)
            if isinstance(parsed_response, dict) and \
               all(key in parsed_response for key in ["query_type", "person_tag", "refined_query"]):
                print(f"Локальная LLM извлекла: query_type='{parsed_response['query_type']}', person_tag='{parsed_response.get('person_tag')}', refined_query='{parsed_response['refined_query']}'")
                if not parsed_response.get("refined_query", "").strip():
                    parsed_response["refined_query"] = user_query
                    print(f"Локальная LLM вернула пустой refined_query, используем исходный: '{user_query}'")
                return parsed_response
            else:
                print(f"Локальная LLM вернула валидный JSON, но без ожидаемых ключей: {llm_response_str}")
                return default_response
        except json.JSONDecodeError:
            print(f"Не удалось распарсить JSON из ответа локальной LLM ПОСЛЕ ОЧИСТКИ: '{llm_response_str}'")
            return default_response
    else:
        print(f"Ошибка при обращении к локальной LLM для анализа запроса или пустой ответ: {llm_response_str}")
        return default_response

# --- Функции инициализации ---
def initialize_qdrant_client():
    try:
        client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
        print(f"Успешное подключение к Qdrant по адресу {QDRANT_HOST}:{QDRANT_GRPC_PORT} (gRPC)")
        return client
    except Exception as e:
        print(f"Не удалось подключиться к Qdrant: {e}")
        raise

def initialize_sbert_model():
    try:
        print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        print("Модель эмбеддингов успешно загружена.")
        return model
    except Exception as e:
        print(f"Не удалось загрузить модель эмбеддингов: {e}")
        raise

# --- Основные функции RAG ---
def search_qdrant_for_context(
    query_text: str, client: QdrantClient, embedding_model: SentenceTransformer,
    collection_name: str, top_k: int = 3, person_tag: str | None = None ):
    try:
        query_embedding = embedding_model.encode(query_text)
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        return []
    
    query_filter = None
    if person_tag and person_tag.strip():
        print(f"Применяется фильтр по person_tag: '{person_tag}'")
        query_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="person_tag", match=qdrant_models.MatchValue(value=person_tag))]
        )
    else:
        print("Фильтр по person_tag не применяется (тег отсутствует или пустой).")
    
    try:
        search_results = client.search(
            collection_name=collection_name, query_vector=query_embedding.tolist(),
            query_filter=query_filter, limit=top_k, with_payload=True
        )
        return search_results
    except Exception as e:
        print(f"Ошибка при поиске в Qdrant: {e}")
        return []

def format_context_for_llm(retrieved_chunks):
    if not retrieved_chunks:
        print("\n--- Контекст для GigaChat ---")
        print("Контекст не найден.")
        print("-----------------------")
        return "Контекст не найден."
    
    context_str = "Вот информация, найденная в архивах:\n"
    for i, hit in enumerate(retrieved_chunks):
        source_file = hit.payload.get('source_file', 'N/A')
        chunk_index = hit.payload.get('chunk_index', 'N/A')
        text_chunk = hit.payload.get('text', 'N/A')
        person_payload_tag = hit.payload.get('person_tag', 'N/A') 
        score = hit.score
        context_str += f"""---
Источник {i+1}: {source_file} (Персона в документе: {person_payload_tag}), Фрагмент: {chunk_index} (Схожесть: {score:.4f})
Текст: {text_chunk}
"""
    context_str += "---\n"
    print("\n--- Контекст для GigaChat ---")
    print(context_str)
    print("-----------------------")
    return context_str

def ask_gigachat_with_context(user_question: str, context_str: str, system_role: str, client_id: str, client_secret: str, dialog_history: list):
    current_user_message_content = f"{context_str}\n\nПожалуйста, ответь на следующий вопрос: {user_question}"
    current_user_message = {"role": "user", "content": current_user_message_content}
    
    few_shot_examples = [
        {"role": "user", "content": "Вот информация, найденная в архивах:\n---\nИсточник 1: Аксентьев.txt (Персона в документе: Аксентьев), Фрагмент: 0 (Схожесть: 0.8500)\nТекст: Леонид Александрович Аксентьев родился 1 марта 1932 года в г. Баку...\n---\n\nПожалуйста, ответь на следующий вопрос: Когда родился Аксентьев?"},
        {"role": "assistant", "content": "Леонид Александрович Аксентьев родился 1 марта 1932 года (Источник: Аксентьев.txt (Персона в документе: Аксентьев), Фрагмент: 0)."}
    ]
    
    api_messages = [{"role": "system", "content": system_role}]
    api_messages.extend(few_shot_examples) 
    api_messages.extend(dialog_history)    
    api_messages.append(current_user_message) 
    
    print("\n--- Сообщения для GigaChat API (структура) ---")
    for i, msg in enumerate(api_messages):
       print(f"Сообщение #{i} (Роль: {msg['role']}):")
       content_to_log = msg['content']
       if len(content_to_log) > 300: 
           print(f"  Начало: {content_to_log[:150]}...")
           print(f"  Конец: ...{content_to_log[-150:]}")
       else:
           print(f"  Контент: {content_to_log}")
    print("----------------------------------------------\n")
    
    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    token_payload = {'scope': 'GIGACHAT_API_PERS'}
    auth_string = f"{client_id}:{client_secret}"
    base64_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json', 'RqUID': str(uuid.uuid4()), 'Authorization': f'Basic {base64_auth_string}'}
    
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
    chat_payload = {"model": "GigaChat:latest", "messages": api_messages, "temperature": 0.7} 
    chat_headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
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

# --- Обновленная функция get_rag_response ---
def get_rag_response(
    user_query_original: str, dialog_history: list, qdrant_client: QdrantClient,
    sbert_model: SentenceTransformer, gigachat_client_id: str, gigachat_client_secret: str,
    known_person_names_list: list = None, # <--- ИСПРАВЛЕНИЕ ЗДЕСЬ: параметр добавлен
    system_role_gigachat: str = GIGACHAT_SYSTEM_ROLE, collection_name: str = COLLECTION_NAME,
    top_k_retriever: int = 5, max_history_turns_llm: int = 3 ):
    
    print(f"\nПолучен оригинальный запрос от пользователя: '{user_query_original}'")

    # Шаг 1: Анализ запроса с помощью локальной LLM, передаем список известных имен
    llm_analysis = get_query_details_from_local_llm(user_query_original, known_person_names=known_person_names_list) 
    query_type = llm_analysis.get("query_type", "question") 
    person_tag_from_llm = llm_analysis.get("person_tag")
    query_for_qdrant_search = llm_analysis.get("refined_query", user_query_original).strip()
    
    if not query_for_qdrant_search: 
        query_for_qdrant_search = user_query_original
        print("Уточненный запрос от локальной LLM был пустым, используем оригинальный для поиска в Qdrant.")

    print(f"Тип запроса от локальной LLM: '{query_type}'")
    print(f"Тег персоны от локальной LLM: '{person_tag_from_llm if person_tag_from_llm else "Нет"}'")
    print(f"Запрос для поиска в Qdrant (от LLM или исходный): '{query_for_qdrant_search}'")

    if query_type == "banter":
        print("Локальная LLM определила запрос как 'banter'. Пропускаем поиск в Qdrant и обращение к GigaChat.")
        banter_user_message_for_history = {
            "role": "user",
            "content": f"Контекст не искался (запрос типа 'banter').\n\nЗапрос пользователя: {user_query_original}"
        }
        lower_query = user_query_original.lower()
        if "привет" in lower_query:
            friendly_response = "Привет! Я Архивариус, готов помочь с вопросами по истории кафедры."
        elif "как дела" in lower_query or "как ты" in lower_query:
            friendly_response = "Все отлично, спасибо! Готов к работе. У вас есть вопросы по архивам?"
        elif "спасибо" in lower_query or "благодарю" in lower_query:
            friendly_response = "Пожалуйста! Рад был помочь."
        else:
            friendly_response = "Хорошо! Если у вас есть вопросы по архивам кафедры, спрашивайте."
        
        print(f"Итоговый ответ (banter): {friendly_response}")
        return friendly_response, banter_user_message_for_history

    retrieved_chunks = search_qdrant_for_context(
        query_text=query_for_qdrant_search, client=qdrant_client, embedding_model=sbert_model,
        collection_name=collection_name, top_k=top_k_retriever, person_tag=person_tag_from_llm
    )

    context_for_llm = format_context_for_llm(retrieved_chunks)
    limited_dialog_history = dialog_history[-(max_history_turns_llm*2):] 
    
    llm_answer, user_message_for_history = ask_gigachat_with_context(
        user_query_original, context_for_llm, system_role_gigachat,
        gigachat_client_id, gigachat_client_secret, limited_dialog_history
    )
    
    print(f"Итоговый ответ от GigaChat: {llm_answer}")
    return llm_answer, user_message_for_history

# --- Основная функция для консольного запуска ---
def main():
    load_dotenv()
    gigachat_client_id = os.getenv("GIGACHAT_CLIENT_ID")
    gigachat_client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")

    if not gigachat_client_id or not gigachat_client_secret:
        print("Ошибка: Переменные окружения GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не установлены.")
        return

    # Демонстрационный список имен. В реальном приложении лучше загружать из Qdrant.
    known_person_names_list_demo = [
        "Аксентьев Леонид Александрович", "Нужин Михаил Тихонович", "Шуликовский Владимир Игнатьевич", 
        "Чеботарев Николай Григорьевич", "Широков Александр Петрович", "Габдулхаев Билсур Габдулхаевич",
        "Лаптев Борис Лукич", "Чибрикова Любовь Ивановна", "Молокович Юрий Матвеевич", "Иванов Николай Николаевич"
    ]
    
    try:
        qdrant_client = initialize_qdrant_client()
        sbert_model = initialize_sbert_model()
        
        # Попытка загрузить имена из Qdrant, если клиент доступен
        final_known_names_list = []
        if qdrant_client:
            # Для консольной версии мы не можем напрямую импортировать из telegram_bot,
            # поэтому, если нужна эта функция, ее нужно либо скопировать сюда,
            # либо сделать частью rag_agent.py (что логичнее).
            # Пока используем демонстрационный список.
            # Если бы load_known_person_names_from_rag_agent_qdrant была здесь:
            # final_known_names_list = load_known_person_names_from_rag_agent_qdrant(qdrant_client, COLLECTION_NAME)
            pass # Заглушка, чтобы показать, где была бы загрузка

        if not final_known_names_list: # Если из Qdrant не загрузилось или функция не вызывалась
            print(f"Используется демонстрационный список из {len(known_person_names_list_demo)} известных имен.")
            final_known_names_list = known_person_names_list_demo
        else:
            print(f"Загружено {len(final_known_names_list)} уникальных имен из Qdrant.")


    except Exception:
        print("Не удалось инициализировать Qdrant или SBERT модель. Завершение работы.")
        return 
    
    dialog_history = []
    MAX_HISTORY_TURNS_CONSOLE = 3 
    
    print("\nДобро пожаловать в Архивариус (консольная версия)! Задавайте ваши вопросы. Для выхода введите 'выход' или 'exit'.")
    print(f"Используется локальная LLM для анализа запросов (API: {LOCAL_LLM_CHAT_API_URL}).")
    print("GigaChat используется для генерации ответов на основе найденного контекста.")

    while True:
        user_query_original = input("\nВы: ")
        if user_query_original.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break
        if not user_query_original.strip(): 
            continue
            
        llm_answer, current_user_msg_for_history = get_rag_response(
            user_query_original=user_query_original, 
            dialog_history=dialog_history,
            qdrant_client=qdrant_client,
            sbert_model=sbert_model,
            gigachat_client_id=gigachat_client_id, 
            gigachat_client_secret=gigachat_client_secret,
            known_person_names_list=final_known_names_list # Передаем актуальный список
        )
        
        dialog_history.append(current_user_msg_for_history) 
        
        if llm_answer and not llm_answer.startswith("[Ошибка локальной LLM:") and \
           not llm_answer.startswith("К сожалению, не удалось получить токен для GigaChat") and \
           not llm_answer.startswith("Не удалось извлечь токен доступа GigaChat") and \
           not llm_answer.startswith("Ошибка при обращении к GigaChat") and \
           not llm_answer.startswith("Получен неожиданный формат ответа от GigaChat"):
            dialog_history.append({"role": "assistant", "content": llm_answer})

        if len(dialog_history) > (MAX_HISTORY_TURNS_CONSOLE * 2 + 4) : 
            dialog_history = dialog_history[-(MAX_HISTORY_TURNS_CONSOLE * 2):]


if __name__ == "__main__":
    print("--- Запуск RAG агента с интеграцией локальной LLM и GigaChat ---")
    main()