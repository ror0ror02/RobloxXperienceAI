import pyautogui
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard

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
input_size = 2  # Для WASD
hidden_size = 128
output_size = 16  # 16 классов: W, A, S, D, 1, 2, E, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK, NONE, F, Z
l2_penalty = 0.001  # Параметр L2-регуляризации

# Создаем словарь нейронных сетей для каждого действия
action_networks = {
    'W': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'A': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'S': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'D': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    '1': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    '2': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'E': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'ENTER': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'SPACE': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'RIGHT_CLICK': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'LEFT_CLICK': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'NONE': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'F': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty),
    'Z': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty)
}

# Определение оптимизаторов для каждой нейронной сети
optimizers = {
    action: optim.Adam(model.parameters(), lr=0.001)
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
    '1': 4, '2': 5, 'E': 6,
    'ENTER': 7, 'SPACE': 8,
    'RIGHT_CLICK': 9, 'LEFT_CLICK': 10,
    'NONE': 11, 'F': 12, 'Z': 13
}

# Основной цикл программы
try:
    while True:
        # Генерируем случайное действие (W, A, S, D, 1, 2, E, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK, NONE, F, Z)
        action = random.choice(list(action_networks.keys()))

        # Генерируем случайные входные данные (x, y) для WASD
        x, y = random.random(), random.random()

        input_tensor = torch.tensor([x, y], dtype=torch.float32)
        label_tensor = torch.tensor([action_to_index[action]], dtype=torch.long)

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
            '1': 'действие 1',
            '2': 'действие 2',
            'E': 'действие E',
            'ENTER': 'нажатие клавиши Enter',
            'SPACE': 'нажатие клавиши SPACE',
            'RIGHT_CLICK': 'правый клик мыши',
            'LEFT_CLICK': 'левый клик мыши',
            'NONE': 'ничего',
            'F': 'действие F',
            'Z': 'действие Z'
        }
        action_description_text = action_description.get(predicted_action, 'неизвестное действие')

        print(f"Выполняется действие: {action_description_text}")

        if predicted_action == 'W':
            keyboard.press_and_release('w')
        elif predicted_action == 'A':
            keyboard.press_and_release('a')
        elif predicted_action == 'S':
            keyboard.press_and_release('s')
        elif predicted_action == 'D':
            keyboard.press_and_release('d')
        elif predicted_action == '1':
            keyboard.press_and_release('1')
        elif predicted_action == '2':
            keyboard.press_and_release('2')
        elif predicted_action == 'E':
            keyboard.press_and_release('e')
        elif predicted_action == 'ENTER':
            keyboard.press_and_release('enter')
        elif predicted_action == 'SPACE':
            keyboard.press_and_release('space')
        elif predicted_action == 'RIGHT_CLICK':
            pyautogui.rightClick()
        elif predicted_action == 'LEFT_CLICK':
            pyautogui.leftClick()
        elif predicted_action == 'F':
            keyboard.press_and_release('f')
        elif predicted_action == 'Z':
            keyboard.press_and_release('z')

except KeyboardInterrupt:
    print("Программа завершена по запросу пользователя.")
except Exception as e:
    print(f"Произошла ошибка: {e}")
