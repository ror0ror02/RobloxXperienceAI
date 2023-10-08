# RobloxXperienceAI

RobloxXperienceAI - это проект, который использует машинное обучение для определенных действий в игре Roblox. Этот README файл предоставляет описание кода, который используется для управления действиями персонажа в игре с использованием нейронных сетей и клавишей.

## Описание

Код проекта использует следующие библиотеки и инструменты:

- `pyautogui`: Для управления мышью и клавиатурой.
- `random`: Генерация случайных чисел для обучения.
- `time`: Для управления задержкой между действиями.
- `torch` и `torch.nn`: Для создания и обучения нейронных сетей.
- `keyboard`: Для управления клавишами клавиатуры.
- `tkinter`: Для создания окна настроек программы.

## Описание нейронных сетей

Проект использует нейронные сети для действий в игре. Есть нейронные сети для каждого из следующих действий:

- Движение вперед (W)
- Движение влево (A)
- Движение назад (S)
- Движение вправо (D)
- Действие 1 (1)
- Действие 2 (2)
- Действие E (E)
- Нажатие клавиши Enter (ENTER)
- Нажатие клавиши Space (SPACE)
- Клик правой кнопкой мыши (RIGHT_CLICK)
- Клик левой кнопкой мыши (LEFT_CLICK)
- Ничего (NONE)

## Описание окна настроек

Проект включает в себя окно настроек, где можно настроить следующие параметры:

- Задержка между действиями: Этот параметр определяет, как долго должно проходить между каждым действием. Это позволяет установить скорость выполнения действий в игре.
- Параметр L2-регуляризации: Этот параметр определяет силу регуляризации для нейронных сетей. Он может влиять на обучение моделей.

## Запуск программы

Убедитесь, что все необходимые библиотеки установлены. Запустите программу, указав необходимые настройки в окне настроек. Программа будет предсказывать и выполнять действия в игре на основе обученных нейронных сетей.

## Важно!

Проект находится в стадии разработки и может требовать дополнительной настройки и оптимизации. Используйте программу в соответствии с правилами и условиями использования игры Roblox. Будьте осторожны и не используйте программу в недобросовестных целях.

## Установка

Для установки и запуска этого проекта, выполните следующие шаги:

1. Клонируйте репозиторий: Сначала склонируйте этот репозиторий на свой компьютер с помощью Git:

```
git clone https://github.com/ror0ror02/RobloxXperienceAI.git
cd RobloxXperienceAI
```

2. Настройте виртуальное окружение: Рекомендуется создать виртуальное окружение для изоляции зависимостей проекта. Вы можете использовать venv в Python:
   
```
python -m venv venv
```
3. Активируйте виртуальное окружение: Затем активируйте созданное виртуальное окружение:

- На Windows:

  ```
  venv\Scripts\activate
  ```

- На macOS и Linux:

  ```
  source venv/bin/activate
  ```

4. Установите зависимости: Установите необходимые зависимости из файла `r_requirements.txt` с помощью pip:
```
pip install -r r_requirements.txt
```
5. Настройте параметры: Отредактируйте параметры в файле `RXAI-New.py` по вашему усмотрению, включая задержку между действиями и другие настройки.

6. Запустите код: Запустите код с помощью Python:
```
python RXAI-New.py
```
(`RXAI-New.py` должен работать в игре Roblox)

*После запуска кода, проект будет использовать нейронные сети для имитации игрока в Roblox*


==================================================***English***=====================================


**RobloxXperienceAI**


*RobloxXperienceAI* is a project that uses machine learning to perform certain actions in the Roblox game. This README file provides a description of the code that is used to control the character's actions in the game using neural networks and keystrokes.


## Description


The project code uses the following libraries and tools:

- `pyautogui`: For mouse and keyboard control.
- `random`: Generate random numbers for training.
- `time`: To control the delay between actions.
- `torch` and `torch.nn`: For creating and training neural networks.
- `keyboard`: To control the keyboard keys.
- `tkinter`: To create a program settings window.

## Description of neural networks

The project uses neural networks for actions in the game. There are neural networks for each of the following actions:

- Move forward (W)
- Move left (A)
- Move backward (S)
- Move right (D)
- Action 1 (1)
- Action 2 (2)
- Action E (E)
- Pressing the Enter key (ENTER)
- Pressing the Space (SPACE) key
- Right-click (RIGHT_CLICK)
- Click with the left mouse button (LEFT_CLICK)
- Nothing (NONE)

## Description of the settings window

The project includes a settings window where you can configure the following parameters:

- Delay between actions: This parameter determines how long should pass between each action. This allows you to set the speed of actions in the game.
- L2 regularization parameter: This parameter determines the strength of regularization for neural networks. It can influence model training.

## Running the program

Make sure all required libraries are installed. Launch the program, specifying the necessary settings in the settings window. The program will predict and perform actions in the game based on trained neural networks.

## Important!

The project is under development and may require additional configuration and optimization. Use the program in accordance with the rules and terms of use of the Roblox game. Be careful and do not use the program for dishonest purposes.

## Installation

To install and run this project, follow these steps:

1. Clone the repository: First clone this repository to your computer using Git:

git clone https://github.com/ror0ror02/RobloxXperienceAI.git
cd RobloxXperienceAI


2. Set up a virtual environment: It is recommended to create a virtual environment to isolate project dependencies. You can use venv in Python:

python -m venv venv

3. Activate the virtual environment: Then activate the created virtual environment:

- On Windows:

   ```
   venv\Scripts\activate
   ```

- On macOS and Linux:

   ```
   source venv/bin/activate
   ```

4. Install dependencies: Install the required dependencies from the `r_requirements.txt` file using pip:
```
pip install -r r_requirements.txt
```
5. Configure parameters: Edit the parameters in the `RXAI-New.py` file as you wish, including the delay between actions and other settings.

6. Run the code: Run the code using Python:
```
python RXAI-New.py
```
(`RXAI-New.py` should work in the Roblox game)

*After running the code, the project will use neural networks to simulate a player in Roblox*
