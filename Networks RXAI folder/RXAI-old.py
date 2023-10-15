import pyautogui
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import tkinter as tk
from tkinter import ttk

# Отключаем fail-safe для PyAutoGUI
pyautogui.FAILSAFE = False

# Определение критерия для обучения
criterion = nn.CrossEntropyLoss()

# Определение нейронной сети с L2-регуляризацией
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, l2_penalty=0.001):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.l2_penalty = l2_penalty  # Параметр для L2-регуляризации

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def l2_regularization(self):
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param)
        return self.l2_penalty * l2_reg

# Создание и обучение нейронных сетей для каждого действия
input_size = 4  # Для WASD и перемещения мыши (x, y)
hidden_size = 256  # Увеличиваем размер скрытого слоя
output_size = 17  # 17 классов: W, A, S, D, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK, NONE, 1, 2, E, SHIFT
l2_penalty = 0.001  # Параметр L2-регуляризации

# Создаем словарь нейронных сетей для каждого действия
action_networks = {
    'W': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'A': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'S': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'D': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'MOVE_LEFT': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'MOVE_RIGHT': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'MOVE_UP': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'MOVE_DOWN': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'ENTER': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'SPACE': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'RIGHT_CLICK': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'LEFT_CLICK': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'NONE': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    '1': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    '2': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'E': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'SHIFT': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty)
}

# Определение оптимизаторов для каждой нейронной сети
optimizers = {
    action: optim.Adam(model.parameters(), lr=0.0001)  # Уменьшаем шаг обучения
    for action, model in action_networks.items()
}

# Процедура обучения нейронной сети с L2-регуляризацией
def train_neural_network(inputs, labels, action):
    optimizer = optimizers[action]
    optimizer.zero_grad()
    outputs = action_networks[action](inputs)
    loss = criterion(outputs, labels)
    loss += action_networks[action].l2_regularization()  # Добавляем L2-регуляризацию к функции потерь
    loss.backward()
    optimizer.step()

# Преобразование символов действий в числовой формат
action_to_index = {
    'W': 0, 'A': 1, 'S': 2, 'D': 3,
    'MOVE_LEFT': 4, 'MOVE_RIGHT': 5,
    'MOVE_UP': 6, 'MOVE_DOWN': 7,
    'ENTER': 8, 'SPACE': 9,
    'RIGHT_CLICK': 10, 'LEFT_CLICK': 11,
    'NONE': 12, '1': 13, '2': 14, 'E': 15,
    'SHIFT': 16
}

# Окно Tkinter для настроек
def start_program():
    global action_delay, l2_penalty, use_mouse_movement
    action_delay = float(action_delay_var.get())
    l2_penalty = float(l2_penalty_var.get())
    use_mouse_movement = use_mouse_movement_var.get()
    root.quit()

root = tk.Tk()
root.title("Настройки программы")

# Добавляем поле для ввода задержки между действиями
action_delay_label = tk.Label(root, text="Задержка между действиями (сек):")
action_delay_label.pack()
action_delay_var = tk.StringVar()
action_delay_entry = ttk.Entry(root, textvariable=action_delay_var)
action_delay_entry.pack()

# Добавляем поле для ввода параметра L2-регуляризации
l2_penalty_label = tk.Label(root, text="Параметр L2-регуляризации:")
l2_penalty_label.pack()
l2_penalty_var = tk.StringVar()
l2_penalty_entry = ttk.Entry(root, textvariable=l2_penalty_var)
l2_penalty_entry.pack()

# Добавляем чекбокс для выбора использования перемещения мыши
use_mouse_movement_var = tk.IntVar()
use_mouse_movement_checkbox = tk.Checkbutton(root, text="Использовать перемещение мыши", variable=use_mouse_movement_var)
use_mouse_movement_checkbox.pack()

# Добавляем кнопку для запуска программы
start_button = tk.Button(root, text="Запустить программу", command=start_program)
start_button.pack()

root.mainloop()

# Выводим выбранные настройки
print(f"Задержка между действиями: {action_delay} сек.")
print(f"Параметр L2-регуляризации: {l2_penalty}")
print(f"Использовать перемещение мыши: {'Да' if use_mouse_movement else 'Нет'}")

# Устанавливаем минимальное и максимальное время ожидания между действиями (в секундах)
min_pause_between_actions = action_delay
max_pause_between_actions = action_delay * 2

# Установите максимальное количество действий, которые программа выполнит
max_actions = 100

# Основной цикл программы
try:
    for _ in range(max_actions):
        # Генерируем случайное действие (W, A, S, D, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK, NONE, 1, 2, E, SHIFT)
        action = random.choice(list(action_networks.keys()))

        # Генерируем случайные входные данные (x, y) и (x_offset, y_offset) для перемещения мыши, если используется
        if use_mouse_movement:
            x, y = random.random(), random.random()
            x_offset, y_offset = random.random() * 2 - 1, random.random() * 2 - 1
        else:
            x, y, x_offset, y_offset = 0, 0, 0, 0

        input_tensor = torch.tensor([x, y, x_offset, y_offset], dtype=torch.float32)
        label_tensor = torch.tensor([action_to_index[action]], dtype=torch.long)

        # Выводим данные для обучения в консоль
        print(f"Данные для обучения: Вход - {input_tensor}, Метка - {label_tensor}, Действие - {action}")

        # Обучаем нейронную сеть для текущего действия
        label_index = action_to_index[action]
        label_tensor = torch.tensor([label_index], dtype=torch.long)

        train_neural_network(input_tensor.unsqueeze(0), label_tensor, action)

        # Выполняем действие на основе предсказания нейронной сети
        _, predicted = torch.max(action_networks[action](input_tensor.unsqueeze(0)), 1)
        predicted_action = [k for k, v in action_to_index.items() if v == predicted.item()][0]

        # Записываем информацию о текущем выполняемом действии в лог-файл
        action_description = {
            'W': 'движение вперед',
            'A': 'движение влево',
            'S': 'движение назад',
            'D': 'движение вправо',
            'MOVE_LEFT': 'перемещение мыши влево',
            'MOVE_RIGHT': 'перемещение мыши вправо',
            'MOVE_UP': 'перемещение мыши вверх',
            'MOVE_DOWN': 'перемещение мыши вниз',
            'ENTER': 'нажатие клавиши Enter',
            'SPACE': 'нажатие клавиши SPACE',
            'RIGHT_CLICK': 'клик правой кнопкой мыши',
            'LEFT_CLICK': 'клик левой кнопкой мыши',
            'NONE': 'ничего',
            '1': 'нажатие клавиши 1',
            '2': 'нажатие клавиши 2',
            'E': 'нажатие клавиши E',
            'SHIFT': 'удержание клавиши Shift'
        }
        action_description_text = action_description.get(predicted_action, 'неизвестное действие')

        print(f"Выполняется действие: {action_description_text}")

        if predicted_action == 'W':
            keyboard.press('w')
        elif predicted_action == 'A':
            keyboard.press('a')
        elif predicted_action == 'S':
            keyboard.press('s')
        elif predicted_action == 'D':
            keyboard.press('d')
        elif predicted_action == 'ENTER':
            keyboard.press('enter')
            time.sleep(0.1)  # Добавляем небольшую задержку перед отпусканием клавиши Enter
            keyboard.release('enter')
        elif predicted_action == 'SPACE':
            keyboard.press('space')
        elif predicted_action == 'RIGHT_CLICK':
            pyautogui.rightClick()
        elif predicted_action == 'LEFT_CLICK':
            pyautogui.leftClick()
        elif predicted_action == '1':
            keyboard.press('1')
            time.sleep(0.1)
            keyboard.release('1')
        elif predicted_action == '2':
            keyboard.press('2')
            time.sleep(0.1)
            keyboard.release('2')
        elif predicted_action == 'E':
            keyboard.press('e')
            time.sleep(0.1)
            keyboard.release('e')
        elif predicted_action == 'SHIFT':
            keyboard.press_and_release('shift')
            time.sleep(10)  # Удерживаем клавишу Shift в течение 10 секунд

        # Добавляем задержку между действиями
        pause_duration = random.uniform(min_pause_between_actions, max_pause_between_actions)
        time.sleep(pause_duration)

        # Отпускаем все клавиши после действия
        keyboard.release('w')
        keyboard.release('a')
        keyboard.release('s')
        keyboard.release('d')
        keyboard.release('enter')
        keyboard.release('space')
        keyboard.release('1')
        keyboard.release('2')
        keyboard.release('e')
        keyboard.release('shift')

except KeyboardInterrupt:
    print("Программа завершена по запросу пользователя.")
except Exception as e:
    print(f"Произошла ошибка: {e}")
