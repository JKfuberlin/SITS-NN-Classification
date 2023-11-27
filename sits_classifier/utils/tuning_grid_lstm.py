import json
import itertools

# Define the possible values for each argument
input_sizes = [64, 128, 256]
hidden_sizes = [128, 256, 512]
num_layers = [3, 4, 5]
batch_size = [128, 256, 512]
bidirectional = [False]

# Generate the grid of argument combinations
grid = list(itertools.product(input_sizes, hidden_sizes, num_layers, bidirectional))

# Save the grid to a file
with open('/home/jonathak90/tuning_grid.json', 'w') as f:
    json.dump(grid, f)