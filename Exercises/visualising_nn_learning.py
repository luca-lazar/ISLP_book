import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# # Step 1: Define the Dataset

# # Generate data points from a sine function.

# class SineWaveDataset(Dataset):
#     def __init__(self, num_samples=1000):
#         self.x = np.linspace(-6 * np.pi, 6 * np.pi, num_samples)
#         self.y = np.sin(self.x)
#         self.y = np.where(self.y>0, 1, self.y)
#         self.y = np.where(self.y<0, -1, self.y)
#         self.x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(1)
#         self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)
    
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

# dataset = SineWaveDataset()
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# # Step 2: Build the Neural Network

# # Define a simple feedforward neural network.

# import torch.nn as nn
# import torch.optim as optim

# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(1, 10)
#         self.fc2 = nn.Linear(10, 30)
#         self.fc3 = nn.Linear(30, 50)
#         self.fc4 = nn.Linear(50, 30)
#         self.fc5 = nn.Linear(30, 10)
#         self.fc6 = nn.Linear(10, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = torch.relu(self.fc5(x))
#         x = self.fc6(x)
#         return x

# model = SimpleNN()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)


# # Step 3: Train the Model and Save Parameters

# # Train the model and save the parameters at each epoch.


# epochs = 600
# predictions = []

# # Generate test data for visualization
# test_x = torch.linspace(-6 * np.pi, 6 * np.pi, 1000).unsqueeze(1)

# for epoch in range(epochs):
#     for x_batch, y_batch in dataloader:
#         optimizer.zero_grad()
#         outputs = model(x_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
    
#     # Save predictions at each epoch
#     with torch.no_grad():
#         predicted_y = model(test_x).numpy()
#     predictions.append(predicted_y)

#     print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# # Step 4: Visualize the Learning Process

# # Use matplotlib to animate the training process.


# # True function for comparison
# test_y = torch.sin(test_x).numpy()
# test_y = np.where(test_y>0, 1, test_y)
# test_y = np.where(test_y<0, -1, test_y)


# fig, ax = plt.subplots()
# line, = ax.plot(test_x.numpy(), test_y, label='True Function')
# predicted_line, = ax.plot(test_x.numpy(), predictions[0], label='Model Prediction')
# ax.legend()

# def update(frame):
#     predicted_line.set_ydata(predictions[frame])
#     return predicted_line,

# # Adjust the interval parameter to slow down the animation
# ani = animation.FuncAnimation(fig, update, frames=len(predictions), blit=True, interval=50)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Neural Network Learning to Fit a Sine Function')

# # Save the animation as a GIF
# ani.save('nn_learning_sine_function.gif', writer='imagemagick', fps=5)

# plt.show()

# POLYNOMIAL FIT
# Generate data
x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
y = np.sin(x)
y = np.where(y>0, 1, y)
y = np.where(y<0, -1, y)

degrees = range(1, 61)
polynomial_predictions = []

for degree in degrees:
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    y_pred = polynomial(x)
    polynomial_predictions.append(y_pred)

fig, ax = plt.subplots()
line, = ax.plot(x, y, label='True Function')
predicted_line, = ax.plot(x, polynomial_predictions[0], label='Polynomial Fit')
ax.legend()

def update(frame):
    predicted_line.set_ydata(polynomial_predictions[frame])
    ax.set_title(f'Polynomial Degree: {degrees[frame]}')
    return predicted_line,

# Adjust the interval parameter to slow down the animation
ani = animation.FuncAnimation(fig, update, frames=len(polynomial_predictions), blit=True, interval=200)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Fit to Sine Function')

# Save the animation as a GIF
ani.save('polynomial_fitting_sine_function.gif', writer='imagemagick', fps=5)

plt.show()
