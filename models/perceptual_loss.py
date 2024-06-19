# -*- coding: utf-8 -*-

"""Perceptual Loss for Super-Resolution.

This code is adapted from https://github.com/gfxdisp/mdf/

This script contains the following classes:
    * VGG16PerceptualLoss: VGG16 perceptual loss class.
"""

__author__ = "Mir Sazzat Hossain"


import torch
import torch.nn as nn
import torchvision.models as models


class VGG16PerceptualLoss(nn.Module):
    """VGG16 Perceptual Loss."""

    def __init__(self, device) -> None:
        """
        Init function.

        :param device: torch.device, device.
        """
        super(VGG16PerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss()
        blocks = []
        blocks.append(models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.blocks = nn.ModuleList(blocks).to(device)

    def forward(self, x, y) -> torch.Tensor:
        """
        Forward function.

        :param x: torch.Tensor, input tensor.
        :param y: torch.Tensor, target tensor.
        :return: torch.Tensor, loss.
        """
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += self.criterion(x, y)
        return loss/len(self.blocks)
