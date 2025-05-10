import os
import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import nltk

# Попытка загрузить данные для токенизатора предложений nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer data not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' downloaded.")

# --- Константы ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6334  # gRPC port, according to Qdrant logs
# Для REST API, если нужно будет обращаться через HTTP (например, для дашборда)
# QDRANT_HTTP_PORT = 6333 # HTTP port, according to Qdrant logs
COLLECTION_NAME = "history_archive"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Размерность вектора для paraphrase-multilingual-mpnet-base-v2
VECTOR_SIZE = 768  
TEXT_SOURCE_DIR = "raw_text" # Директория с вашими текстовыми файлами

# Примерная максимальная длина чанка в символах. 
# Если абзац длиннее, он будет разбит на предложения.
MAX_CHUNK_LENGTH_CHARS = 1000 # Можно настроить
# Минимальное количество предложений в чанке, если разбиваем по предложениям
MIN_SENTENCES_PER_CHUNK = 2 # Можно настроить
MIN_CHUNK_LENGTH_FOR_STORAGE_CHARS = 75 # Новая константа: минимальная длина чанка для сохранения

def get_text_chunks(filepath):
    """
    Читает текстовый файл и разбивает его на чанки.
    Сначала пытается разделить по абзацам (два переноса строки).
    Если абзац слишком длинный или получается всего один чанк на файл,
    пытается разбить этот абзац (или весь текст) на группы предложений.
    Добавлена фильтрация по минимальной длине чанка.
    """
    chunks = []
    full_text = "" # Инициализируем на случай, если файл не откроется
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 1. Сначала пытаемся разделить по двойным переводам строки (абзацы)
        initial_paragraphs = full_text.split('\n\n')
        
        processed_paragraphs = []
        for p_text in initial_paragraphs:
            p_text = p_text.strip()
            if not p_text:
                continue

            # 2. Если абзац слишком длинный, или если у нас всего один "абзац" на весь файл
            #    и он не короткий, то разбиваем его на предложения.
            if len(p_text) > MAX_CHUNK_LENGTH_CHARS or (len(initial_paragraphs) == 1 and len(p_text) > 200): # 200 - эвристика для коротких файлов
                sentences = nltk.sent_tokenize(p_text)
                current_chunk_sentences = []
                current_chunk_len = 0
                for sent_idx, sent in enumerate(sentences):
                    current_chunk_sentences.append(sent)
                    current_chunk_len += len(sent)
                    
                    # Формируем чанк, если:
                    # 1. Набрали достаточно предложений И (длина чанка превышает половину максимума ИЛИ это последнее предложение абзаца)
                    # 2. ИЛИ длина текущего чанка уже превысила максимум (даже если предложений мало, но это одно длинное предложение)
                    if (len(current_chunk_sentences) >= MIN_SENTENCES_PER_CHUNK and \
                        (current_chunk_len > MAX_CHUNK_LENGTH_CHARS / 1.5 or sent_idx == len(sentences) - 1)) \
                        or current_chunk_len > MAX_CHUNK_LENGTH_CHARS:
                        
                        # Пытаемся не создавать слишком длинные чанки, если это возможно
                        if current_chunk_len > MAX_CHUNK_LENGTH_CHARS and len(current_chunk_sentences) > 1:
                            # Если без последнего предложения чанк все еще валиден по количеству
                            if len(current_chunk_sentences[:-1]) >= MIN_SENTENCES_PER_CHUNK or len(current_chunk_sentences[:-1]) == 0: # len == 0, если первое предложение уже слишком длинное
                                processed_paragraphs.append(" ".join(current_chunk_sentences[:-1]))
                                current_chunk_sentences = [sent] # Начинаем новый чанк с текущего предложения
                                current_chunk_len = len(sent)
                            else: # Оставляем как есть, если убрав последнее предложение, станет слишком мало
                                processed_paragraphs.append(" ".join(current_chunk_sentences))
                                current_chunk_sentences = []
                                current_chunk_len = 0
                        else: # Обычное добавление
                            processed_paragraphs.append(" ".join(current_chunk_sentences))
                            current_chunk_sentences = []
                            current_chunk_len = 0
                
                # Добавляем оставшиеся предложения, если они есть
                if current_chunk_sentences:
                    processed_paragraphs.append(" ".join(current_chunk_sentences))
            else:
                # Если абзац не слишком длинный, оставляем его как есть
                processed_paragraphs.append(p_text)
        
        # Финальная очистка, фильтрация по длине и добавление в chunks
        for chunk_candidate in processed_paragraphs:
            cleaned_chunk = chunk_candidate.strip()
            # Фильтруем по минимальной длине И по непустоте
            if cleaned_chunk and len(cleaned_chunk) >= MIN_CHUNK_LENGTH_FOR_STORAGE_CHARS:
                chunks.append(cleaned_chunk)
            elif cleaned_chunk: # Если чанк непустой, но слишком короткий
                print(f"Предупреждение: Пропускается слишком короткий чанк (длина {len(cleaned_chunk)}) из файла {filepath}: '{cleaned_chunk[:50]}...'")

    except FileNotFoundError:
        print(f"Файл не найден: {filepath}")
    except Exception as e:
        print(f"Ошибка при чтении или обработке файла {filepath}: {e}")
    
    if not chunks and full_text.strip() and len(full_text.strip()) >= MIN_CHUNK_LENGTH_FOR_STORAGE_CHARS:
        print(f"Предупреждение: не удалось разбить на чанки файл {filepath} удовлетворяющие критериям. Загружается как один чанк, если он достаточно длинный.")
        chunks.append(full_text.strip())
    elif not chunks and full_text.strip():
         print(f"Предупреждение: не удалось разбить на чанки файл {filepath}. Текст слишком короткий для загрузки как единый чанк (длина {len(full_text.strip())}).")
        
    return chunks

def main():
    # 1. Инициализация клиента Qdrant
    try:
        # client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_PORT, prefer_grpc=True)
        print(f"Успешное подключение к Qdrant по адресу {QDRANT_HOST}:{QDRANT_PORT} (gRPC)")
    except Exception as e:
        print(f"Не удалось подключиться к Qdrant: {e}")
        return

    # 2. Инициализация модели эмбеддингов
    try:
        print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        # Можно указать device='cuda' если доступен GPU и установлен PyTorch с поддержкой CUDA
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') 
        print("Модель эмбеддингов успешно загружена.")
    except Exception as e:
        print(f"Не удалось загрузить модель эмбеддингов: {e}")
        return

    # 3. Удаление и пересоздание коллекции в Qdrant
    try:
        print(f"Попытка удаления существующей коллекции '{COLLECTION_NAME}' (если существует)...")
        client.delete_collection(collection_name=COLLECTION_NAME) # Удаляем, если существует
        print(f"Коллекция '{COLLECTION_NAME}' успешно удалена или не существовала.")
        
        print(f"Создание новой коллекции '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        print(f"Коллекция '{COLLECTION_NAME}' успешно создана/пересоздана.")

    except Exception as e:
        print(f"Ошибка при пересоздании коллекции Qdrant: {e}")
        # Если ошибка связана с тем, что коллекция не найдена при удалении, это нормально,
        # но recreate_collection должна справиться. Если другая ошибка - выходим.
        # Проверяем, это ли ошибка "Not found"
        if "Not found" in str(e) or "Collection not found" in str(e): # Примерная проверка текста ошибки
            print("Продолжаем, так как коллекция просто не была найдена для удаления.")
            # Попробуем создать, если удаление не удалось из-за отсутствия
            try:
                print(f"Повторная попытка создания коллекции '{COLLECTION_NAME}'...")
                client.create_collection( # Используем create_collection, если recreate вызвал ошибку из-за несуществующей коллекции
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
                )
                print(f"Коллекция '{COLLECTION_NAME}' успешно создана.")
            except Exception as e2:
                print(f"Ошибка при создании коллекции '{COLLECTION_NAME}' после неудачного удаления: {e2}")
                return
        else:
            return # Если была другая ошибка при recreate_collection

    # 4. Обработка и загрузка текстовых файлов
    processed_files = 0
    total_chunks_loaded = 0
    
    if not os.path.exists(TEXT_SOURCE_DIR):
        print(f"Директория с исходными текстами '{TEXT_SOURCE_DIR}' не найдена.")
        return

    print(f"Начало обработки файлов из директории: {TEXT_SOURCE_DIR}")
    for filename_with_ext in os.listdir(TEXT_SOURCE_DIR):
        if filename_with_ext.endswith(".txt"):
            filepath = os.path.join(TEXT_SOURCE_DIR, filename_with_ext)
            person_tag = os.path.splitext(filename_with_ext)[0] # Извлекаем тег из имени файла
            print(f"Обработка файла: {filepath} (Тег персоны: '{person_tag}')...")
            
            chunks = get_text_chunks(filepath)
            if not chunks:
                print(f"В файле {filename_with_ext} не найдено текстовых фрагментов для загрузки.")
                continue

            print(f"Найдено {len(chunks)} чанков. Генерация эмбеддингов...")
            
            try:
                chunk_embeddings = model.encode(chunks, show_progress_bar=True)
            except Exception as e:
                print(f"Ошибка при генерации эмбеддингов для файла {filename_with_ext}: {e}")
                continue
            
            points_to_upsert = []
            for i, chunk_text in enumerate(chunks):
                points_to_upsert.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=chunk_embeddings[i].tolist(),
                        payload={
                            "text": chunk_text,
                            "source_file": filename_with_ext,
                            "chunk_index": i,
                            "person_tag": person_tag # <--- ДОБАВЛЕН ТЕГ ПЕРСОНЫ
                        }
                    )
                )
            
            if points_to_upsert:
                try:
                    print(f"Загрузка {len(points_to_upsert)} точек в Qdrant...")
                    client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
                    print(f"Успешно загружено {len(points_to_upsert)} точек из файла {filename_with_ext}.")
                    total_chunks_loaded += len(points_to_upsert)
                except Exception as e:
                    print(f"Ошибка при загрузке точек в Qdrant для файла {filename_with_ext}: {e}")
            
            processed_files += 1
            
    print(f"\n--- Завершение загрузки ---")
    print(f"Всего обработано файлов: {processed_files}")
    print(f"Всего загружено чанков (точек) в Qdrant: {total_chunks_loaded}")

if __name__ == "__main__":
    main() 