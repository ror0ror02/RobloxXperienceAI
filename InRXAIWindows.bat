@echo off

rem Проверка наличия Git
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ошибка: Git не найден. Установите Git и повторите попытку.
    exit /b 1
)


rem Проверка наличия Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Ошибка: Python не найден. Установите Python и повторите попытку.
    exit /b 1
)


rem Установка зависимостей
pip install -r r_requirements.txt

echo Установка завершена. Теперь вы можете запустить проект, следуя инструкциям в README.md.
