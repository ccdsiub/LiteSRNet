# -*- coding: utf-8 -*-

"""Recurrent CNN model for image super-resolution.

This script contains the following classes:
    * RecurrentCNN: Recurrent CNN model class.
    * UpscaleDecoder: Upscale decoder class.
"""

___author___ = "Mir Sazzat Hossain"


import torch
import torch.nn as nn


class UpscaleDecoder(nn.Module):
    """Upscale decoder class."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        scale_factor: int = 4
    ) -> None:
        """
        Init function.

        :param in_channels: int, number of input channels.
        :type in_channels: int
        :param out_channels: int, number of output channels.
        :type out_channels: int
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(UpscaleDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.trans_conv = nn.Identity()

        if self.scale_factor == 2:
            self.trans_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    self.in_channels, self.out_channels,
                    kernel_size=2, stride=2),
                nn.ReLU(inplace=True))
        elif self.scale_factor == 3:
            self.trans_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    self.in_channels, self.out_channels,
                    kernel_size=3, stride=3),
                nn.ReLU(inplace=True))
        elif self.scale_factor == 4:
            self.trans_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    self.in_channels, self.in_channels//2,
                    kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    self.in_channels//2, self.out_channels,
                    kernel_size=2, stride=2),
                nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: torch.Tensor, input tensor.
        :type x: torch.Tensor

        :return: torch.Tensor, output tensor.
        :rtype: torch.Tensor
        """
        out = self.trans_conv(x)
        return out


class RecurCNN(nn.Module):
    """Recurrent CNN model."""

    def __init__(
        self,
        width,
        depth,
        in_channels: int = 3,
        out_channels: int = 3,
        scale_factor: int = 4,
    ) -> None:
        """
        Init function.

        :param width: int, width of the model.
        :type width: int
        :param depth: int, depth of the model.
        :type depth: int
        :param in_channels: int, number of input channels.
        :type in_channels: int
        :param out_channels: int, number of output channels.
        :type out_channels: int
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(RecurCNN, self).__init__()
        self.width = width
        if scale_factor == 4:
            self.iters = depth - 5
        else:
            self.iters = depth - 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, int(self.width / 2),
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(self.width / 2), self.width,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.recur_layers = nn.Sequential(
            nn.Conv2d(self.width, self.width,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.upscale_layer = UpscaleDecoder(
            self.width, self.out_channels, self.scale_factor)

        self.final_conv = nn.Conv2d(
            self.out_channels, self.out_channels,
            kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: torch.Tensor, input tensor.
        :type x: torch.Tensor

        :return: torch.Tensor, output tensor.
        :rtype: torch.Tensor
        """
        self.thoughts = torch.zeros(
            (self.iters, x.shape[0], self.out_channels,
             x.shape[2] * self.scale_factor,
             x.shape[3] * self.scale_factor)).to(x.device)

        out = self.first_layer(x)
        for i in range(self.iters):
            out = self.recur_layers(out)
            thought = self.upscale_layer(out)
            thought = self.final_conv(thought)
            self.thoughts[i] = thought

        return self.thoughts[-1]
