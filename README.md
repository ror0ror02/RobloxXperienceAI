# RobloxXperienceAI

**ССЫЛКА НА ПОЛНЫЙ GITHUB** - https://github.com/ror0ror02/RobloxXperienceAI


https://github.com/ror0ror02/RobloxXperienceAI/assets/142999522/d1860581-f1a3-4b33-bb2f-88ba7fdb9e02




## На этом видео нейросеть учится играть в the Rake




https://github.com/ror0ror02/RobloxXperienceAI/assets/142999522/f357865f-b476-4d8a-b536-61e3d6f12dcf




## А на этом видео - Пишет в чат и играет в Doors

**RobloxXperienceAI**- это проект, который использует машинное обучение для определенных действий в игре Roblox. Этот README файл предоставляет описание кода, который используется для управления действиями персонажа в игре с использованием нейронных сетей и клавишей.

В этом README представлен обзор проекта и его социальных экспериментов.

## Социальные эксперименты

### Сотрудничество с другими игроками ***(СКОРО!)***

- **Описание:** RobloxXperienceAI сотрудничает с другими игроками в Roblox для достижения общих целей.
- **Цель:** Исследовать, как RobloxXperienceAI может сотрудничать и взаимодействовать с другими игроками.
- **Шаги:** Запустить RobloxXperienceAI на сервере с другими игроками и наблюдать за совместной работой.

### Эксперимент в игре "Doors" **(Проводился)**

- **Описание:** RobloxXperienceAI взаимодействует с другими игроками через чат и играет в игру "Doors".
- **Цель:** Исследовать, как RobloxXperienceAI может решать головоломки и взаимодействовать с другими игроками.
- **Шаги:** Запустить RobloxXperienceAI в игре ***"Doors"*** и наблюдать за его взаимодействием.

### Эксперимент в игре "The Rake" **(Проводился)**

- **Описание:** RobloxXperienceAI играет в игру "The Rake", сосредотачиваясь на выживании.
- **Цель:** Проверить способность RobloxXperienceAI выживать и навигироваться в игре.
- **Шаги:** Запустить RobloxXperienceAI в игре ***"The Rake"*** и наблюдать за его производительностью.

### Имитация новичка **(Проводился)**

- **Описание:** RobloxXperienceAI выступает в роли новичка в игре ***"Jailbreak"***, допуская ошибки и запрашивая помощь.
- **Цель:** Исследовать реакцию игроков на новичков и их готовность помогать.
- **Шаги:** Запустить RobloxXperienceAI как новичка в игре "Jailbreak" и наблюдать за взаимодействием игроков.

## Результаты

- В эксперименте в игре "Doors" RobloxXperienceAI взаимодействовал с другими игроками через чат и следовал за игроками.
- В эксперименте в игре "The Rake" RobloxXperienceAI продемонстрировал Хорошие навыки выживания (После обучения).
- Эксперимент в игре "Jailbreak" "Имитация новичка" .


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


### Windows - Авто установка

Для пользователей Windows, установите проект с помощью следующих шагов:

1. Откройте командную строку (cmd) с правами администратора.

2. Запустите скрипт `InRXAIWindows.bat`, выполнив следующую команду:
   
```
InRXAIWindows.bat
```

3. Следуйте инструкциям, которые появятся в командной строке. Скрипт выполнит все необходимые действия для установки.

### Unix-подобные системы (Linux и macOS)

Для пользователей Unix-подобных систем, установите проект с помощью следующих шагов:

1. Откройте терминал.

2. Запустите скрипт `installRobloxXperienceAIUnix.sh`, выполнив следующую команду:
```
   chmod +x installRobloxXperienceAIUnix.sh
./installRobloxXperienceAIUnix.sh
```

3. Следуйте инструкциям, которые появятся в терминале. Скрипт выполнит все необходимые действия для установки.

После выполнения скрипта установки, вы можете запустить проект, следуя инструкциям в README.md.

**Примечание**: Если у вас возникают проблемы с установкой, убедитесь, что у вас установлены Git и Python. В противном случае, установите их перед выполнением скрипта установки.


# English


**RobloxXperienceAI**


*RobloxXperienceAI* is a project that uses machine learning to perform certain actions in the Roblox game. This README file provides a description of the code that is used to control the character's actions in the game using neural networks and keystrokes.


## Social Experiments

### Cooperation with Other Players **(Not Conducted)**

- **Description:** RobloxXperienceAI collaborates with other players in Roblox to achieve common goals.
- **Objective:** Explore how RobloxXperienceAI can cooperate and interact with other players.
- **Steps:** Launch RobloxXperienceAI on a server with other players and observe its collaboration.

### Experiment in "Doors" Game **(Conducted)**

- **Description:** RobloxXperienceAI interacts with other players through chat and plays the "Doors" game.
- **Objective:** Investigate how RobloxXperienceAI can solve puzzles and interact with other players.
- **Steps:** Launch RobloxXperienceAI in the "Doors" game and observe its interactions.

### Experiment in "The Rake" Game **(Conducted)**

- **Description:** RobloxXperienceAI plays "The Rake" game, focusing on survival.
- **Objective:** Test RobloxXperienceAI's ability to survive and navigate the game.
- **Steps:** Launch RobloxXperienceAI in "The Rake" game and observe its performance.

### Novice Imitation **( End)**

- **Description:** RobloxXperienceAI acts as a newcomer in the "Jailbreak" game, making mistakes and seeking help.
- **Objective:** Investigate player reactions to newcomers and their willingness to help.
- **Steps:** Launch RobloxXperienceAI as a newcomer in the "Jailbreak" game and observe player interactions.

## Results

- In the "Doors" game experiment, RobloxXperienceAI interacted with other players through chat and solved puzzles.
- In the "The Rake" game experiment, RobloxXperienceAI demonstrated survival skills.
- The "Jailbreak" game experiment "Novice Imitation" .


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
python RXAI-ENG-New.py
```
(`RXAI-ENG-New.py` should work in the Roblox game)

*After running the code, the project will use neural networks to simulate a player in Roblox*
