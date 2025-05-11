import requests
import json

# --- НАСТРОЙКА ---
GEMMA_SERVER_IP = "192.168.1.7"
GEMMA_SERVER_PORT = 8000
GEMMA_COMPLETIONS_API_URL = f"http://{GEMMA_SERVER_IP}:{GEMMA_SERVER_PORT}/v1/completions" 
GEMMA_CHAT_API_URL = f"http://{GEMMA_SERVER_IP}:{GEMMA_SERVER_PORT}/v1/chat/completions"
# --- КОНЕЦ НАСТРОЙКИ ---

def call_remote_gemma_completion(prompt_text: str, stop_sequences: list = None):
    """
    Отправляет запрос на удаленный сервер Gemma (эндпоинт /v1/completions).
    """
    if stop_sequences is None:
        stop_sequences = ["\n", "###"] 

    payload = {
        "prompt": prompt_text,
        "stop": stop_sequences,
        "temperature": 0.2, 
        "n_predict": 150, # Немного увеличим для возможного русского ответа
        "stream": False
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    print(f"\nОтправка запроса к Gemma API ({GEMMA_COMPLETIONS_API_URL})...")
    print(f"Промпт: {prompt_text}")
    try:
        response = requests.post(GEMMA_COMPLETIONS_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        gemma_response_data = response.json()
        if gemma_response_data.get("choices") and \
           len(gemma_response_data["choices"]) > 0 and \
           gemma_response_data["choices"][0].get("text"):
            answer = gemma_response_data["choices"][0]["text"].strip() 
            print(f"\nОтвет от Gemma (/v1/completions):\n{answer}")
            return answer
        else:
            error_detail = str(gemma_response_data)[:500]
            print(f"Неожиданный формат ответа от Gemma API (/v1/completions): {error_detail}")
            return f"[Ошибка /v1/completions: Неожиданный формат ответа: {error_detail}]"
    except requests.exceptions.Timeout:
        print("Ошибка: Запрос к Gemma API (/v1/completions) превысил время ожидания.")
        return "[Ошибка /v1/completions: Таймаут]"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к Gemma API (/v1/completions): {e}")
        if e.response is not None: print(f"Тело ответа сервера (ошибка): {e.response.text}")
        return f"[Ошибка /v1/completions: Подключение: {e}]"
    except Exception as e:
        print(f"Неизвестная ошибка при работе с Gemma API (/v1/completions): {e}")
        return f"[Ошибка /v1/completions: Неизвестно: {e}]"

def call_remote_gemma_chat(user_query: str, dialog_history: list = None, system_prompt: str = "Ты — полезный ИИ-ассистент."):
    """
    Отправляет запрос на удаленный сервер Gemma (эндпоинт /v1/chat/completions).
    Системный промпт "внедряется" в первое сообщение пользователя, если история пуста.
    """
    if dialog_history is None:
        dialog_history = []
    
    processed_messages = []

    for entry in dialog_history:
        if entry.get("role") != "system":
            processed_messages.append({"role": entry["role"], "content": entry["content"]})
    
    current_user_content = user_query
    if system_prompt and not processed_messages: 
        current_user_content = f"{system_prompt}\n\nЗапрос пользователя:\n{user_query}" # Изменил "Инструкции пользователю" на "Запрос пользователя" для ясности
        print(f"(Системный промпт '{system_prompt[:50]}...' внедрен в первый запрос пользователя)")
    
    processed_messages.append({"role": "user", "content": current_user_content})
    
    payload = {
        "model": "gemma-model", 
        "messages": processed_messages,
        "temperature": 0.7, 
        "n_predict": 300,   # Увеличим для русских ответов, они могут быть длиннее в токенах
        "stream": False
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    print(f"\nОтправка запроса к Gemma Chat API ({GEMMA_CHAT_API_URL})...")
    # print("Payload для /v1/chat/completions:", json.dumps(payload, indent=2, ensure_ascii=False)) 
    try:
        response = requests.post(GEMMA_CHAT_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        gemma_response_data = response.json()
        if gemma_response_data.get("choices") and \
           len(gemma_response_data["choices"]) > 0 and \
           gemma_response_data["choices"][0].get("message") and \
           gemma_response_data["choices"][0]["message"].get("content"):
            answer = gemma_response_data["choices"][0]["message"]["content"].strip() 
            print(f"\nОтвет от Gemma Chat (/v1/chat/completions):\n{answer}")
            return answer
        else:
            error_detail = str(gemma_response_data)[:500]
            print(f"Неожиданный формат ответа от Gemma Chat API: {error_detail}")
            return f"[Ошибка Chat: Неожиданный формат ответа: {error_detail}]"
    except requests.exceptions.Timeout:
        print("Ошибка: Запрос к Gemma Chat API превысил время ожидания.")
        return "[Ошибка Chat: Таймаут]"
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к Gemma Chat API: {e}")
        if e.response is not None:
            print(f"Тело ответа сервера (ошибка): {e.response.text}")
        return f"[Ошибка Chat: Подключение: {e}]"
    except Exception as e:
        print(f"Неизвестная ошибка при работе с Gemma Chat API: {e}")
        return f"[Ошибка Chat: Неизвестно: {e}]"

# --- Тестовые вызовы ---
if __name__ == '__main__':
    print("--- Тест 1 (RU): Запрос к /v1/completions ---")
    curl_like_prompt_ru = "\n\n### Инструкция:\nКакая столица Франции?\n\n### Ответ:\n"
    call_remote_gemma_completion(curl_like_prompt_ru)

    print("\n\n--- Тест 2 (RU): /v1/chat/completions - Системный промпт внедряется (нет истории) ---")
    chat_question_1_ru = "Какая столица Германии?"
    system_prompt_ru_generic = "Ты — очень знающий и полезный ИИ-ассистент. Твои ответы должны быть краткими."
    call_remote_gemma_chat(chat_question_1_ru, system_prompt=system_prompt_ru_generic)

    print("\n\n--- Тест 3 (RU): /v1/chat/completions - С историей ---")
    history_for_test3_ru = [
        {"role": "user", "content": f"{system_prompt_ru_generic}\n\nЗапрос пользователя:\nКакая столица Германии?"},
        {"role": "assistant", "content": "Столица Германии — Берлин."}
    ]
    chat_question_2_ru = "Каково его примерное население?"
    call_remote_gemma_chat(chat_question_2_ru, dialog_history=history_for_test3_ru, system_prompt=system_prompt_ru_generic)

    print("\n\n--- Тест 4 (RU): /v1/chat/completions - Переформулировка запроса ---")
    query_to_rewrite_ru = "А какие там основные отрасли промышленности?"
    rewriting_system_prompt_ru = """Ты — эксперт по переформулировке запросов на русском языке. Твоя задача — взять исходный запрос пользователя и историю диалога, и переформулировать запрос так, чтобы он был максимально понятен для системы поиска информации, сохраняя при этом все ключевые детали и контекст из истории.
Пример:
ИСХОДНЫЙ ЗАПРОС: Кто он такой?
ИСТОРИЯ: [{"role": "user", "content": "Я читал про Аксентьева."}, {"role": "assistant", "content": "Леонид Аксентьев был выдающимся математиком."}]
РЕЗУЛЬТАТ: Расскажи подробнее о Леониде Аксентьеве."""
    
    history_for_rewrite_task_ru = [
         {"role": "user", "content": f"{rewriting_system_prompt_ru}\n\nЗапрос пользователя:\nРасскажи о городе Гамбург."},
         {"role": "assistant", "content": "Гамбург — крупный портовый город на севере Германии."}
    ]
    call_remote_gemma_chat(query_to_rewrite_ru, dialog_history=history_for_rewrite_task_ru, system_prompt=rewriting_system_prompt_ru)
    
    print("\n\n--- Тест 5 (RU): /v1/chat/completions - Простое приветствие ---")
    greeting_system_prompt_ru = "Ты — дружелюбный ассистент по имени Архивариус. Вежливо и кратко отвечай на приветствия на русском языке."
    call_remote_gemma_chat("Привет!", system_prompt=greeting_system_prompt_ru)
    call_remote_gemma_chat("Как твои дела?", system_prompt=greeting_system_prompt_ru)
