import os
import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

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

def get_text_chunks(filepath):
    """
    Читает текстовый файл и разбивает его на абзацы (чанки).
    Каждый непустой абзац считается отдельным чанком.
    """
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        # Разделяем по двум или более переводам строки, чтобы корректно обрабатывать абзацы
        paragraphs = text.split('\n\n') 
        for p in paragraphs:
            cleaned_paragraph = p.strip()
            if cleaned_paragraph: # Добавляем только непустые абзацы
                chunks.append(cleaned_paragraph)
    except FileNotFoundError:
        print(f"Файл не найден: {filepath}")
    except Exception as e:
        print(f"Ошибка при чтении файла {filepath}: {e}")
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

    # 3. Создание или проверка коллекции в Qdrant
    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Коллекция '{COLLECTION_NAME}' не найдена. Создание новой коллекции...")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            )
            print(f"Коллекция '{COLLECTION_NAME}' успешно создана.")
        else:
            print(f"Коллекция '{COLLECTION_NAME}' уже существует.")
            # Можно добавить проверку, соответствует ли конфигурация существующей коллекции ожидаемой
            # collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            # print(f"Информация о коллекции: {collection_info}")

    except Exception as e:
        print(f"Ошибка при работе с коллекциями Qdrant: {e}")
        return

    # 4. Обработка и загрузка текстовых файлов
    processed_files = 0
    total_chunks_loaded = 0
    
    if not os.path.exists(TEXT_SOURCE_DIR):
        print(f"Директория с исходными текстами '{TEXT_SOURCE_DIR}' не найдена.")
        return

    print(f"Начало обработки файлов из директории: {TEXT_SOURCE_DIR}")
    for filename in os.listdir(TEXT_SOURCE_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(TEXT_SOURCE_DIR, filename)
            print(f"Обработка файла: {filepath}...")
            
            chunks = get_text_chunks(filepath)
            if not chunks:
                print(f"В файле {filename} не найдено текстовых фрагментов для загрузки.")
                continue

            print(f"Найдено {len(chunks)} чанков в файле {filename}. Генерация эмбеддингов...")
            
            # Генерируем эмбеддинги для всех чанков файла сразу (батчинг)
            try:
                chunk_embeddings = model.encode(chunks, show_progress_bar=True)
            except Exception as e:
                print(f"Ошибка при генерации эмбеддингов для файла {filename}: {e}")
                continue
            
            points_to_upsert = []
            for i, chunk_text in enumerate(chunks):
                points_to_upsert.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()), # Уникальный ID для каждого чанка
                        vector=chunk_embeddings[i].tolist(), # Преобразуем numpy array в list
                        payload={
                            "text": chunk_text,
                            "source_file": filename,
                            "chunk_index": i 
                        }
                    )
                )
            
            if points_to_upsert:
                try:
                    print(f"Загрузка {len(points_to_upsert)} точек в Qdrant для файла {filename}...")
                    client.upsert(collection_name=COLLECTION_NAME, points=points_to_upsert)
                    print(f"Успешно загружено {len(points_to_upsert)} точек из файла {filename}.")
                    total_chunks_loaded += len(points_to_upsert)
                except Exception as e:
                    print(f"Ошибка при загрузке точек в Qdrant для файла {filename}: {e}")
            
            processed_files += 1
            
    print(f"\n--- Завершение загрузки ---")
    print(f"Всего обработано файлов: {processed_files}")
    print(f"Всего загружено чанков (точек) в Qdrant: {total_chunks_loaded}")

if __name__ == "__main__":
    main() 