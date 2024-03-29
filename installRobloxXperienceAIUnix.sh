#!/bin/bash

# Проверка наличия Git
if ! command -v git &> /dev/null; then
    echo "Ошибка: Git не найден. Установите Git и повторите попытку."
    exit 1
fi


# Проверка наличия Python
if ! command -v python &> /dev/null; then
    echo "Ошибка: Python не найден. Установите Python и повторите попытку."
    exit 1
fi


# Установка зависимостей
pip install -r r_requirements.txt

echo "Установка завершена. Теперь вы можете запустить проект, следуя инструкциям в README.md."
