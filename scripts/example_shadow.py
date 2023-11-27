import numpy as np
import matplotlib.pyplot as plt

# Define the x values
x1 = np.linspace(0, 2*np.pi, 100)
x2 = np.linspace(0, 10, 100)

# Define two functions to compare
y1 = np.sin(x1)
y2 = np.cos(x1)

# Define the function for the second subplot
y3 = np.sin(x2)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# First subplot: sin(x) and cos(x) with a shadow
axs[0].plot(x1, y1, label='sin(x)')
axs[0].plot(x1, y2, label='cos(x)')
axs[0].fill_between(x1, y1, y2, where=(y1 > y2), color='blue', alpha=0.3, interpolate=True, label='Difference')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

# Second subplot: sin(x) with a shadow representing a domain of values
axs[1].plot(x2, y3, label='Function')
lower_bound = y3 - 0.2  # Define lower boundary
upper_bound = y3 + 0.2  # Define upper boundary
axs[1].fill_between(x2, lower_bound, upper_bound, color='gray', alpha=0.5, label='Domain of Values')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()

# Adjust the layout
plt.tight_layout()

# Show the subplots
plt.show()
