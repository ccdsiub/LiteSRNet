# -*- coding: utf-8 -*-

"""Discriminator model for image super-resolution.

This script contains the following classes:
    * Discriminator: Discriminator model class.
"""

__author__ = "Mir Sazzat Hossain"


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator model class."""

    def __init__(self, in_channels: int = 3) -> None:
        """
        Init function.

        :param in_channels: int, number of input channels.
        :type in_channels: int
        """
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: torch.Tensor, input tensor.
        :type x: torch.Tensor

        :return: torch.Tensor, output tensor.
        :rtype: torch.Tensor
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)
        out = nn.Flatten()(out)
        out = self.fc(out)
        pred = nn.Sigmoid()(out)
        return pred
