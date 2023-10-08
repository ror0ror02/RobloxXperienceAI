import pyautogui
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import tkinter as tk
from tkinter import ttk

# Definition of criterion for training
criterion = nn.CrossEntropyLoss()

# Definition of a neural network with L2 regularization
class NeuralNetwork(nn.Module):
     def __init__(self, input_size, hidden_size, output_size, l2_penalty=0.001):
         super(NeuralNetwork, self).__init__()
         self.fc1 = nn.Linear(input_size, hidden_size)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(hidden_size, output_size)
         self.l2_penalty = l2_penalty # Parameter for L2 regularization

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

# Create and train neural networks for each action
input_size = 4 # For WASD
hidden_size = 128
output_size = 11 # 16 classes: W, A, S, D, 1, 2, E, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK, NONE
l2_penalty = 0.001 # L2 regularization parameter

# Create a dictionary of neural networks for each action
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
     'NONE': NeuralNetwork(input_size, hidden_size, output_size, l2_penalty)
}

# Define optimizers for each neural network
optimizers = {
     action: optim.Adam(model.parameters(), lr=0.001)
     for action, model in action_networks.items()
}

# Procedure for training a neural network with L2 regularization
def train_neural_network(inputs, labels, action):
     optimizer = optimizers[action]
     optimizer.zero_grad()
     outputs = action_networks[action](inputs)
     loss = criterion(outputs, labels)
     loss += action_networks[action].l2_regularization() # Add L2 regularization to the loss function
     loss.backward()
     optimizer.step()

# Convert action characters to numeric format
action_to_index = {
     'W': 0, 'A': 1, 'S': 2, 'D': 3,
     '1': 4, '2': 5, 'E': 6,
     'ENTER': 7, 'SPACE': 8,
     'RIGHT_CLICK': 9, 'LEFT_CLICK': 10,
     'NONE': 11
}

# Tkinter settings window
def start_program():
     global action_delay, l2_penalty
     action_delay = float(action_delay_var.get())
     l2_penalty = float(l2_penalty_var.get())
     print(f"Delay between actions: {action_delay} seconds")
     print(f"L2 regularization parameter: {l2_penalty}")
     root.quit()

root = tk.Tk()
root.title("Program settings")

# Add a field to enter a delay between actions
action_delay_label = tk.Label(root, text="Delay between actions (sec):")
action_delay_label.pack()
action_delay_var = tk.StringVar()
action_delay_entry = ttk.Entry(root, textvariable=action_delay_var)
action_delay_entry.pack()

# Add a field for entering the L2-regularization parameter
l2_penalty_label = tk.Label(root, text="L2 regularization parameter:")
l2_penalty_label.pack()
l2_penalty_var = tk.StringVar()
l2_penalty_entry = ttk.Entry(root, textvariable=l2_penalty_var)
l2_penalty_entry.pack()

# Add a button to start the program with the selected settings
start_button = tk.Button(root, text="Start", command=start_program)
start_button.pack()

root.mainloop()

# Tkinter window
root = tk.Tk()
root.title("Action Prediction")

# Label for neural network prediction output
prediction_label = tk.Label(root, text="Action prediction: waiting...")
prediction_label.pack()

# Main program loop
try:
     while True:
         # Generate a random action (W, A, S, D, 1, 2, E, ENTER, SPACE, RIGHT_CLICK, LEFT_CLICK)
         action = random.choice(list(action_networks.keys()))

         input_tensor = torch.tensor([random.random(), random.random(), random.random(), random.random()], dtype=torch.float32)
         label_tensor = torch.tensor([action_to_index[action]], dtype=torch.long)

         # Train the neural network for the current action
         label_index = action_to_index[action]
         label_tensor = torch.tensor([label_index], dtype=torch.long)

         train_neural_network(input_tensor.unsqueeze(0), label_tensor, action)
       _, predicted = torch.max(action_networks[action](input_tensor.unsqueeze(0)), 1)
         predicted_action = [k for k, v in action_to_index.items() if v == predicted.item()][0]

         # Write information about the currently performed action to a log file
         action_description = {
             'W': 'moving forward',
             'A': 'movement left',
             'S': 'backward movement',
             'D': 'movement to the right',
             '1': 'action 1',
             '2': 'action 2',
             'E': 'action E',
             'ENTER': 'pressing the Enter key',
             'SPACE': 'pressing the SPACE key',
             'RIGHT_CLICK': 'right mouse click',
             'LEFT_CLICK': 'left mouse click',
             'NONE': 'nothing'
         }
         action_description_text = action_description.get(predicted_action, 'unknown action')

         print(f"Action in progress: {action_description_text}")

         if predicted_action == 'W':
             keyboard.press('w')
         elif predicted_action == 'A':
             keyboard.press('a')
         elif predicted_action == 'S':
             keyboard.press('s')
         elif predicted_action == 'D':
             keyboard.press('d')
         elif predicted_action == '1':
             keyboard.press('1')
         elif predicted_action == '2':
             keyboard.press('2')
         elif predicted_action == 'E':
             keyboard.press('e')
         elif predicted_action == 'ENTER':
             keyboard.press('enter')
             time.sleep(0.1) # Add a short delay before releasing the Enter key
             keyboard.release('enter')
         elif predicted_action == 'SPACE':
             keyboard.press('space')
         elif predicted_action == 'RIGHT_CLICK':
             pyautogui.rightClick()
         elif predicted_action == 'LEFT_CLICK':
             pyautogui.leftClick()

         # Add a delay between actions
         time.sleep(action_delay)

         # Release all keys after the action
         keyboard.release('w')
         keyboard.release('a')
         keyboard.release('s')
         keyboard.release('d')
         keyboard.release('1')
         keyboard.release('2')
         keyboard.release('e')
         keyboard.release('enter')
         keyboard.release('space')

except KeyboardInterrupt:
     print("The program ended at the user's request.")
except Exception as e:
     print(f"An error occurred: {e}")

# Close the Tkinter window
root.destroy()
