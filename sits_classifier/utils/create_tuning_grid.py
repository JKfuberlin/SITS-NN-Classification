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
    with open('/home/jonathan/data/tuning_grid_LSTM_pixelbased.json', 'w') as f:
        json.dump(grid, f)

else: # Transformer
    # Define the possible values for each argument
    d_model = [512, 1028] # 128 - 1028 This parameter represents the dimensionality of the model. Higher values provide the model with more capacity to learn complex patterns but also increase computational requirements.
    nhead = [12, 16] # 2 - 16 The number of attention heads. A higher number allows the model to focus on different parts of the input sequence simultaneously.
    num_layers = [6, 12] # 2 - 12 This parameter controls the depth of the model. Deeper models can capture more intricate patterns but might require more data and computation.
    dim_feedforward = [256, 1028, 4096] # 256 - 4096 This is the dimensionality of the feedforward networks within the transformer. A larger dimension allows the model to learn more complex representations.
    batch_size = [16, 128] # 16 - 128 Reasoning: Batch size affects the number of samples used in each iteration. A larger batch size can lead to more stable training but requires more memory.

    # Generate the grid of argument combinations
    grid = list(itertools.product(d_model, nhead, num_layers, dim_feedforward, batch_size))
    # Save the grid to a file
    with open('/home/jonathan/data/tuning_grid_transformer_pixelbased.json', 'w') as f:
        json.dump(grid, f)

'''can't get dump(indent = 2) method to work to add newlines after each combination
use :%s/],/],\r/g in vim instead'''
