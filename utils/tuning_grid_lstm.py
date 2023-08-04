import json
import itertools

# Define the possible values for each argument
input_sizes = [64, 128, 256]
hidden_sizes = [128, 256, 512]
num_layers = [3, 4, 5]
bidirectional = [True, False]

# Generate the grid of argument combinations
grid = list(itertools.product(input_sizes, hidden_sizes, num_layers, bidirectional))

# Save the grid to a file
with open('/home/j/data/tuning_grid.json', 'w') as f:
    json.dump(grid, f)