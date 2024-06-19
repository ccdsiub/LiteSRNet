# -*- coding: utf-8 -*-

"""Test script for DIV2K dataset."""

__author__ = "Mir Sazzat Hossain"


import os
import unittest

from torch.utils.data import DataLoader

from utils.div2k_dataset import DIV2KDataset


class TestDIV2KDataset(unittest.TestCase):
    """Test class for DIV2K dataset."""

    def setUp(self) -> None:
        """Set up function."""
        self.data_dir = os.path.join("data")
        self.scale_factor = 4
        self.patch_size = 48
        self.overlap_height_ratio = 0.
        self.overlap_width_ratio = 0.
        
        self.train_dataset = DIV2KDataset(
            data_dir=self.data_dir,
            scale_factor=self.scale_factor,
            patch_size=self.patch_size * self.scale_factor,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            dataset_type="train",
        )
        self.test_dataset = DIV2KDataset(
            data_dir=self.data_dir,
            scale_factor=self.scale_factor,
            patch_size=self.patch_size * self.scale_factor,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            dataset_type="test",
        )

    def test_train_dataset(self) -> None:
        """Test train dataset."""
        self.assertEqual(len(self.train_dataset), 68920)
        self.assertEqual(self.train_dataset[0][0].shape, (3, 48, 48))
        self.assertEqual(self.train_dataset[0][1].shape, (3, 192, 192))

    def test_test_dataset(self) -> None:
        """Test test dataset."""
        self.assertEqual(len(self.test_dataset), 8723)
        self.assertEqual(self.test_dataset[0][0].shape, (3, 48, 48))
        self.assertEqual(self.test_dataset[0][1].shape, (3, 192, 192))

    def test_train_dataloader(self) -> None:
        """Test train dataloader."""
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.assertEqual(len(train_dataloader), 4308)
        for lr_images, hr_images in train_dataloader:
            self.assertEqual(lr_images.shape, (4, 3, 48, 48))
            self.assertEqual(hr_images.shape, (4, 3, 192, 192))
            break

    def test_test_dataloader(self) -> None:
        """Test test dataloader."""
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.assertEqual(len(test_dataloader), 546)
        for lr_images, hr_images, path_names in test_dataloader:
            self.assertEqual(lr_images.shape, (16, 3, 48, 48))
            self.assertEqual(hr_images.shape, (16, 3, 192, 192))
            self.assertEqual(len(path_names), 16)
            break

if __name__ == "__main__":
    unittest.main()