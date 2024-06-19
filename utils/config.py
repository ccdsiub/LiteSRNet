# -*- coding: utf-8 -*-

"""
Configuration file for the USTGCN model.

This file contains configuration of Recurrent CNN model.
"""

__author__ = "Mir Sazzat Hossain"

import argparse

import torch
import yaml

recursive_cnn_config = {
    "exp_params": {
        "scale_factor": 4,
        "patch_size": 48,
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-4,
        "num_workers": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "add_discriminator": False,
    },
    "model_params": {
        "width": 64,
        "depth": 13,
    },
    "data_params": {
        "data_dir": "data",
        "train_dataset_name": "DIV2K",
        "test_dataset_names": ["Set5", "Set14", "BSD100", "Urban100"],
        "overlap_height_ratio": 0.,
        "overlap_width_ratio": 0.,
    },
    "log_params": {
        "log_dir": "logs",
        "result_dir": "results",
        "save_interval": 5,
        "test_model_path": None,
    },
}


def write_config(config: dict, config_name: str) -> None:
    """
    Write the configuration dictionary to a yaml file.

    :param config: The configuration dictionary.
    :type config: dict
    :param config_name: The name of the configuration file.
    :type config_name: str
    """
    with open(
        f'configs/{config_name}_config.yaml',
        'w', encoding='utf8'
    ) as config_file:
        yaml.dump(config, config_file, default_flow_style=False)


def load_config(config_name: str) -> dict:
    """
    Load the configuration dictionary from a yaml file.

    :param config_name: The name of the configuration file.
    :type config_name: str
    :return: The configuration dictionary.
    :rtype: dict
    """
    with open(
        f'configs/{config_name}_config.yaml',
        'r', encoding='utf8'
    ) as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Configuration file for Recursive CNN model.'
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default='recursive_cnn',
        help='The name of the configuration file.'
    )
    args = parser.parse_args()
    write_config(recursive_cnn_config, args.config_name)
