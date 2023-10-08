@echo off

rem Проверка наличия Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ошибка: Git не найден. Установите Git и повторите попытку.
    exit /b 1
)

rem Клонирование репозитория
git clone https://github.com/ror0ror02/RobloxXperienceAI.git
cd RobloxXperienceAI

rem Проверка наличия Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ошибка: Python не найден. Установите Python и повторите попытку.
    exit /b 1
)

rem Создание виртуального окружения
python -m venv venv

rem Активация виртуального окружения
.\venv\Scripts\activate

rem Установка зависимостей
pip install -r r_requirements.txt

echo Установка завершена. Теперь вы можете запустить проект, следуя инструкциям в README.md.
