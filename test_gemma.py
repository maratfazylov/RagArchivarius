import requests
import json

# --- НАСТРОЙКА ---
# IP-адрес и порт вашего сервера Gemma
GEMMA_SERVER_IP = "192.168.1.7"  # IP-адрес вашего мощного ПК
GEMMA_SERVER_PORT = 8000
# Используем эндпоинт /v1/completions для этого теста
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
        # "temperature": 0.7, # Можно добавить другие параметры, если нужно
        # "max_tokens": 50,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print(f"\nОтправка запроса к Gemma API ({GEMMA_COMPLETIONS_API_URL})...")
    print(f"Промпт: {prompt_text}")
    # print("Payload для /v1/completions:", json.dumps(payload, indent=2, ensure_ascii=False))


    try:
        response = requests.post(GEMMA_COMPLETIONS_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        
        gemma_response_data = response.json()
        
        # print("Полный ответ от Gemma API (/v1/completions):", json.dumps(gemma_response_data, indent=2, ensure_ascii=False)) 
        
        if gemma_response_data.get("choices") and \
           len(gemma_response_data["choices"]) > 0 and \
           gemma_response_data["choices"][0].get("text"):
            
            answer = gemma_response_data["choices"][0]["text"]
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
        return f"[Ошибка /v1/completions: Подключение: {e}]"
    except Exception as e:
        print(f"Неизвестная ошибка при работе с Gemma API (/v1/completions): {e}")
        return f"[Ошибка /v1/completions: Неизвестно: {e}]"

def call_remote_gemma_chat(user_query: str, dialog_history: list = None, system_prompt: str = "Ты — полезный ИИ-ассистент.", include_system_prompt_in_messages: bool = True):
    if dialog_history is None:
        dialog_history = []
    
    messages = []

    if include_system_prompt_in_messages and system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    for entry in dialog_history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    
    messages.append({"role": "user", "content": user_query})
    
    if not include_system_prompt_in_messages and system_prompt and not dialog_history:
         messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_query}"}]

    payload = {
        "model": "gemma-model", 
        "messages": messages,
        "temperature": 0.7,
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
            answer = gemma_response_data["choices"][0]["message"]["content"]
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
    print("--- Тест 1: Запрос к /v1/completions (аналогично curl) ---")
    curl_like_prompt = "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n"
    call_remote_gemma_completion(curl_like_prompt)

    print("\n\n--- Тест 2А: Запрос к /v1/chat/completions БЕЗ system prompt в messages ---")
    chat_question = "What is the capital of Germany?"
    call_remote_gemma_chat(chat_question, system_prompt="You are a helpful assistant.", include_system_prompt_in_messages=False)
    
    print("\n\n--- Тест 2Б: Запрос к /v1/chat/completions С system prompt (как было, ожидаем ошибку) ---")
    call_remote_gemma_chat(chat_question, system_prompt="You are a helpful assistant.", include_system_prompt_in_messages=True)
