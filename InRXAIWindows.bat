
@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Проверка наличия Git и Python
echo Checking for Git and Python...
where git >nul 2>nul
IF !ERRORLEVEL! NEQ 0 (
    echo Git is not installed. Please install Git and try again.
    exit /b
)
where python >nul 2>nul
IF !ERRORLEVEL! NEQ 0 (
    echo Python is not installed. Please install Python and try again.
    exit /b
)
echo Git and Python found.

:: Клонирование или обновление репозитория
echo Checking RobloxXperienceAI repository...
if not exist "RobloxXperienceAI" (
    echo Cloning RobloxXperienceAI repository...
    git clone https://github.com/ror0ror02/RobloxXperienceAI.git
) else (
    echo RobloxXperienceAI repository already exists. Pulling latest updates...
    cd RobloxXperienceAI
    git pull
    cd..
)

:: Установка зависимостей
cd RobloxXperienceAI
echo Installing dependencies...
pip install -r r_requirements.txt
IF !ERRORLEVEL! NEQ 0 (
    echo Error occurred during installation of dependencies.
    exit /b
)
echo Dependencies installed successfully.

:: Запуск RobloxXperienceAI
echo Launching RobloxXperienceAI...
cd Networks RXAI folder
python RXAI-New.py
IF !ERRORLEVEL! NEQ 0 (
    echo Error occurred during launching RobloxXperienceAI.
    exit /b
)

echo RobloxXperienceAI launched successfully.
ENDLOCAL
