import uuid
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- Константы (должны совпадать с load_data_to_qdrant.py) ---
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334  # gRPC port, как мы выяснили
COLLECTION_NAME = "history_archive"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# VECTOR_SIZE = 768 # Необязательно здесь, но полезно для справки

def search_in_qdrant(query_text: str, client: QdrantClient, embedding_model: SentenceTransformer, collection_name: str, top_k: int = 3):
    """
    Генерирует эмбеддинг для запроса и ищет наиболее похожие векторы в Qdrant.
    
    Args:
        query_text: Текст запроса.
        client: Клиент Qdrant.
        embedding_model: Модель для генерации эмбеддингов.
        collection_name: Имя коллекции в Qdrant.
        top_k: Количество возвращаемых наиболее похожих результатов.
        
    Returns:
        Список найденных точек (ScoredPoint).
    """
    print(f"\nПоисковый запрос: '{query_text}'")
    print(f"Генерация эмбеддинга для запроса...")
    try:
        query_embedding = embedding_model.encode(query_text)
    except Exception as e:
        print(f"Ошибка при генерации эмбеддинга для запроса: {e}")
        return []

    print(f"Выполнение поиска в Qdrant (коллекция: '{collection_name}', top_k={top_k})...")
    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(), # Преобразуем numpy array в list
            limit=top_k,
            with_payload=True # Запрашиваем payload вместе с результатами
        )
        print("Поиск завершен.")
        return search_results
    except Exception as e:
        print(f"Ошибка при поиске в Qdrant: {e}")
        return []

def main():
    # 1. Инициализация клиента Qdrant
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True)
        print(f"Успешное подключение к Qdrant по адресу {QDRANT_HOST}:{QDRANT_GRPC_PORT} (gRPC)")
    except Exception as e:
        print(f"Не удалось подключиться к Qdrant: {e}")
        return

    # 2. Инициализация модели эмбеддингов
    try:
        print(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
        sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        print("Модель эмбеддингов успешно загружена.")
    except Exception as e:
        print(f"Не удалось загрузить модель эмбеддингов: {e}")
        return

    # 3. Пример поискового запроса
    test_query = "Когда родился Леонид Аксентьев??"
    
    results = search_in_qdrant(
        query_text=test_query, 
        client=qdrant_client, 
        embedding_model=sbert_model, 
        collection_name=COLLECTION_NAME, 
        top_k=5 # Увеличим до 5
    )
    
    if results:
        print(f"\n--- Результаты поиска (top {len(results)}) ---")
        for i, hit in enumerate(results):
            print(f"Результат #{i+1}")
            print(f"  ID: {hit.id}")
            print(f"  Score: {hit.score:.4f}") # Оценка схожести (для Cosine чем ближе к 1, тем лучше)
            # Чтобы увидеть сам текст, нам нужно его либо хранить в payload, либо получить его позже
            # В load_data_to_qdrant.py мы сохраняли текст в payload['text']
            # Если мы не указали with_payload=True при поиске, payload будет None
            # Давайте модифицируем search, чтобы он всегда запрашивал payload
            if hit.payload:
                print(f"  Текст: {hit.payload.get('text', 'N/A')}")
                print(f"  Источник: {hit.payload.get('source_file', 'N/A')}")
                print(f"  Индекс чанка: {hit.payload.get('chunk_index', 'N/A')}") # Добавим вывод индекса чанка
            else:
                print("  Payload не был загружен (это не должно произойти, если with_payload=True).")
            print("---")
    else:
        print("Результатов не найдено или произошла ошибка.")

if __name__ == "__main__":
    main() 