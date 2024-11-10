import numpy as np

# Initialize parameters
gamma = 0.9  # Discount factor
theta = 0.0001  # Small threshold for value iteration convergence
reward_matrix = np.array([
    [0, -1, -1, -1, 0],
    [-1, 0, -1, 0, -1],
    [-1, -1, 0, -1, -1],
    [0, -1, -1, 0, -1],
    [-1, 0, -1, -1, 10]  # Goal state with a high reward
])

# Initialize the value matrix with zeros
value_matrix = np.zeros(reward_matrix.shape)

# Function to apply the Bellman equation and compute the value matrix
def bellman_update(reward_matrix, value_matrix, gamma, theta):
    num_rows, num_cols = reward_matrix.shape
    while True:
        delta = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if (i, j) == (num_rows - 1, num_cols - 1):
                    continue  # Skip the goal state
                v = value_matrix[i, j]
                # Bellman equation: Update the value for state (i, j)
                new_value = reward_matrix[i, j] + gamma * max([
                    value_matrix[i-1, j] if i > 0 else 0,  # Up
                    value_matrix[i+1, j] if i < num_rows-1 else 0,  # Down
                    value_matrix[i, j-1] if j > 0 else 0,  # Left
                    value_matrix[i, j+1] if j < num_cols-1 else 0  # Right
                ])
                value_matrix[i, j] = new_value
                delta = max(delta, abs(v - new_value))
        if delta < theta:
            break
    return value_matrix

# Run the Bellman equation to compute the value matrix
updated_value_matrix = bellman_update(reward_matrix, value_matrix, gamma, theta)

# Output the value matrix
print("Value Matrix after Bellman Equation:")
print(updated_value_matrix)
