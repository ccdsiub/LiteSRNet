# -*- coding: utf-8 -*-

"""This module contains tests for the discriminator."""

__author__ = "Mir Sazzat Hossain"


import unittest

import torch
from torchsummary import summary

from models.recur_cnn import RecurCNN


class TestRecurrentCNN(unittest.TestCase):
    """Test case for the recurrent CNN."""

    def setUp(self):
        """Set up function."""
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                      else "cpu")
        self.recur_cnn = RecurCNN(
            width=64,
            depth=13
        ).to(self.device)


    def test_recur_cnn(self):
        """Test the recurrent CNN."""
        # print model summary
        print()
        summary(self.recur_cnn, (3, 48, 48))

        # Create fake input.
        x = torch.randn(16, 3, 48, 48).to(self.device)

        # Get output.
        out = self.recur_cnn(x)

        # Check output shape.
        self.assertEqual(out.shape, torch.Size([16, 3, 4*48, 4*48]))

if __name__ == "__main__":
    unittest.main()