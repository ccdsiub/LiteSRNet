#!/bin/bash
for config in configs/*.yaml; do
    config_name=$(basename $config .yaml)
    config_name=${config_name%_config}
    echo "Training model with config: $config_name"
    python main.py --config $config_name
done