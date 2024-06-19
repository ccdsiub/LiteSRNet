# -*- coding: utf-8 -*-

"""This module contains tests for the discriminator."""

__author__ = "Mir Sazzat Hossain"


import unittest

import torch
from torchsummary import summary

from models.discriminator import Discriminator


class TestDiscriminator(unittest.TestCase):
    """Test case for the discriminator."""

    def setUp(self):
        """Set up function."""
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                      else "cpu")
        self.discriminator = Discriminator().to(self.device)

    def test_discriminator(self):
        """Test the discriminator."""
        # Create fake input.
        x = torch.randn(1, 3, 48, 48).to(self.device)

        # Get output.
        out = self.discriminator(x)

        # Check output shape.
        self.assertEqual(out.shape, torch.Size([1, 1]))

        # Check output values.
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

        # Print model summary.
        print()
        summary(self.discriminator, (3, 48, 48))

if __name__ == "__main__":
    unittest.main()