# -*- coding: utf-8 -*-

"""
Controller for the Recurrent CNN model for image super-resolution.

This file contains the main function to run the model.
"""

__author__ = "Mir Sazzat Hossain"

import argparse
import random

import numpy as np
import torch

from train.recur_cnn_trainer import RecurrentCNNTrainer
from utils.config import load_config


def clear_cache() -> None:
    """Clear the cache of PyTorch."""
    torch.cuda.empty_cache()


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    :param seed: The seed value.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed value for reproducibility.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="recursive_cnn",
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
    )
    args = parser.parse_args()

    clear_cache()
    set_seed(args.seed)

    config = load_config(args.config)

    if args.test_only and config["log_params"]["test_model_path"] is None:
        raise ValueError(
            "Please provide the path to the model to be tested."
        )

    trainer = RecurrentCNNTrainer(
        data_dir=config["data_params"]["data_dir"],
        train_dataset_name=config["data_params"]["train_dataset_name"],
        test_dataset_names=config["data_params"]["test_dataset_names"],
        scale_factor=config["exp_params"]["scale_factor"],
        patch_size=config["exp_params"]["patch_size"],
        batch_size=config["exp_params"]["batch_size"],
        overlap_height_ratio=config["data_params"]["overlap_height_ratio"],
        overlap_width_ratio=config["data_params"]["overlap_width_ratio"],
        width=config["model_params"]["width"],
        depth=config["model_params"]["depth"],
        num_epochs=config["exp_params"]["epochs"],
        lr=config["exp_params"]["lr"],
        num_workers=config["exp_params"]["num_workers"],
        device=torch.device(config["exp_params"]["device"]),
        log_dir=config["log_params"]["log_dir"],
        result_dir=config["log_params"]["result_dir"],
        save_interval=config["log_params"]["save_interval"],
        add_discriminator=config["exp_params"]["add_discriminator"],
    )

    if not args.test_only:
        trainer.train()

    test_model_path = config["log_params"]["test_model_path"] \
        if args.test_only else None
    
    print("Testing model: ", test_model_path)

    for dataset_name in config["data_params"]["test_dataset_names"]:
        trainer.test(
            dataset_name=dataset_name,
            model_path=test_model_path,
        )
