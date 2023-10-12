import random

# Список доступных юнитов
units = ["Tank", "Airplane", "Infantry"]

# Список доступных действий
actions = ["Move", "Attack", "Defend"]

# Выбор случайного юнита и действия
random_unit = random.choice(units)
random_action = random.choice(actions)

# Выполнение действия
if random_action == "Move":
    print(f"{random_unit} двигается к цели.")
elif random_action == "Attack":
    print(f"{random_unit} атакует врага.")
elif random_action == "Defend":
    print(f"{random_unit} защищает позиции.")
else:
    print("Неверное действие.")

