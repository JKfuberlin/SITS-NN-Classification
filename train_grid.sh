#!/bin/bash

# Read the grid file using Python and convert the JSON array to individual lines
# grid=$(python -c "import json; print('\n'.join(json.load(open('/home/j/data/tuning_grid.json'))))")
grid=$(python -c "import json; print('\n'.join(map(json.dumps, json.load(open('/home/j/data/tuning_grid.json')))))")

# Set the initial UID value
uid_counter=1

echo "starting"
# Iterate over the grid
while IFS= read -r combo; do
    # Parse the arguments from the grid using Python
    input_size=$(echo "$combo" | python -c "import json, sys; print(json.loads(sys.stdin.read())[0])")
    hidden_size=$(echo "$combo" | python -c "import json, sys; print(json.loads(sys.stdin.read())[1])")
    num_layers=$(echo "$combo" | python -c "import json, sys; print(json.loads(sys.stdin.read())[2])")
    batch_size=$(echo "$combo" | python -c "import json, sys; print(json.loads(sys.stdin.read())[3])")
    bidirectional=$(echo "$combo" | python -c "import json, sys; print(json.loads(sys.stdin.read())[4])")

     # Display the current combination of arguments
    echo "Running with arguments:"
    echo "UID: $uid_counter"
    echo "input_size: $input_size"
    echo "hidden_size: $hidden_size"
    echo "num_layers: $num_layers"
    echo "batch_size: $batch_size"
    echo "bidirectional: $bidirectional"


    # Call the Python script with the argument values
    python /home/j/PycharmProjects/TSC_CNN/lstm_classification_tuning.py "$uid_counter" "$input_size" "$hidden_size" "$num_layers"  "$batch_size" "$bidirectional"

    uid_counter=$((uid_counter + 1))

    # Display a separator between combinations for clarity
    echo "=========================================="

done <<< "$grid"
