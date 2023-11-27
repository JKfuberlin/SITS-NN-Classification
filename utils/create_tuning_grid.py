import json
import itertools

LSTM = 0

if LSTM == 1:
    # Define the possible values for each argument
    input_sizes = [64, 128, 256]
    hidden_sizes = [128, 256, 512]
    num_layers = [3, 4, 5]
    batch_size = [128, 256, 512]
    bidirectional = [False]

    # Generate the grid of argument combinations
    grid = list(itertools.product(input_sizes, hidden_sizes, num_layers, bidirectional))
    # Save the grid to a file
    with open('/home/jonathak90/tuning_grid_LSTM_pixelbased.json', 'w') as f:
        json.dump(grid, f)

else: # Transformer
    # Define the possible values for each argument
    d_model = [64, 128, 256]
    nhead = [128, 256, 512]
    num_layers = [3, 4, 5]
    dim_feedforward = [128, 256, 512]
    batch_size = [128, 256, 512]


    # Generate the grid of argument combinations
    grid = list(itertools.product(d_model, nhead, num_layers, dim_feedforward, batch_size))
    # Save the grid to a file
    with open('/home/jonathak90/tuning_grid_transformer_pixelbased.json', 'w') as f:
        json.dump(grid, f)

