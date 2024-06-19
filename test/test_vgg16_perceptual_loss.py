# -*- coding: utf-8 -*-

"""This module contains tests for VGG16 perceptual loss."""

__author__ = "Mir Sazzat Hossain"


import unittest

import torch

from models.perceptual_loss import VGG16PerceptualLoss


class TestVGG16PerceptualLoss(unittest.TestCase):
    """Test case for the VGG16 perceptual loss."""

    def setUp(self):
        """Set up function."""
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                      else "cpu")
        self.vgg16_perceptual_loss = VGG16PerceptualLoss(
            device=self.device
        ).to(self.device)


    def test_vgg16_perceptual_loss(self):
        """Test the VGG16 perceptual loss."""
        # Create fake input.
        x = torch.randn(16, 3, 256, 256).to(self.device)
        y = torch.randn(16, 3, 256, 256).to(self.device)

        # Get output.
        out = self.vgg16_perceptual_loss(x, y)

        # Check output shape.
        self.assertEqual(out.shape, torch.Size([]))

if __name__ == "__main__":
    unittest.main()