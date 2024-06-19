# -*- coding: utf-8 -*-

"""This module contains tests for the recursive CNN trainer."""

__author__ = "Mir Sazzat Hossain"

import unittest

import torch

from train.recur_cnn_trainer import RecurrentCNNTrainer


class TestRecurrentCNNTrainer(unittest.TestCase):
    """Test case for the recursive CNN trainer."""

    def setUp(self):
        """Set up function."""
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                      else "cpu")
        self.recur_cnn_trainer = RecurrentCNNTrainer(
            data_dir="data",
            scale_factor=4,
            patch_size=48,
            batch_size=16,
            overlap_height_ratio=0.,
            overlap_width_ratio=0.,
            width=64,
            depth=13,
            num_epochs=2,
            lr=1e-4,
            num_workers=4,
            device=self.device,
            log_dir="logs",
            result_dir="results",
            save_interval=1,
            add_discriminator=False,
        )

    def test_recur_cnn_trainer(self):
        """Test the recursive CNN trainer."""
        self.recur_cnn_trainer.train()
        self.recur_cnn_trainer.test()

        self.recur_cnn_trainer.test(
            model_path = "logs/run_0/best_model.pth"
        )

    def test_recur_cnn_trainer_with_discriminator(self):
        """Test the recursive CNN trainer with discriminator."""
        self.recur_cnn_trainer.add_discriminator = True
        self.recur_cnn_trainer.train()
        self.recur_cnn_trainer.test()

        self.recur_cnn_trainer.test(
            model_path = "logs/run_1/best_model.pth"
        )

if __name__ == "__main__":
    unittest.main()