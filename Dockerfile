# Использовать официальный образ Python в качестве базового
FROM python:3.10-slim

# Установить рабочую директорию внутри контейнера
WORKDIR /app

# Скопировать файл с зависимостями в рабочую директорию
COPY requirements.txt .

# Установить CPU-версию PyTorch, она значительно меньше и быстрее.
# Это ключевая оптимизация для ускорения сборки и уменьшения размера образа.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Установить остальные зависимости, указанные в requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать все остальные файлы проекта (код, тексты и т.д.) в рабочую директорию
COPY . .

# Указать команду, которая будет выполняться при запуске контейнера
CMD ["python", "telegram_bot.py"] 