#!/bin/bash

# List of step values to test
STEP_VALUES=(10 50 100)

# Path to your configuration file
CONFIG_PATH="cfgs/tent_proxy.yaml"

# Directory to store results
RESULTS_DIR="test_results"
mkdir -p $RESULTS_DIR

# Run the script for each step value
for steps in "${STEP_VALUES[@]}"; do
    echo "Running with OPTIM.STEPS=$steps..."
    
    # Run the Python script with overridden OPTIM.STEPS
    python cifar10c.py --cfg $CONFIG_PATH OPTIM.STEPS $steps | tee "$RESULTS_DIR/results_steps_$steps.txt"
done

echo "All tests completed. Results saved in $RESULTS_DIR/"

