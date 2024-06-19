"""Recurrence residual NN for SR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

bn_tracker = 0


class BasicBlock(nn.Module):
    """Basic residual block."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        iters: int = 1,
    ):
        """
        Init function.

        :param in_planes: int, input channel.
        :type in_planes: int
        :param planes: int, output channel.
        :type planes: int
        :param stride: int, stride.
        :type stride: int
        :param iters: int, number of iterations.
        :type iters: int
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
            )

        self.bn_layer_list1 = nn.Sequential(
            *[nn.BatchNorm2d(planes) for _ in range(iters)])
        self.bn_layer_list2 = nn.Sequential(
            *[nn.BatchNorm2d(planes) for _ in range(iters)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: tensor, input tensor.
        :type x: tensor

        :return: tensor, output tensor.
        :rtype: tensor
        """
        global bn_tracker
        out = F.relu(self.bn_layer_list1[bn_tracker](self.conv1(x)))
        out = self.bn_layer_list2[bn_tracker](self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpsampleBlock(nn.Module):
    """Upsample block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ):
        """
        Init function.

        :param in_channels: int, input channel.
        :type in_channels: int
        :param out_channels: int, output channel.
        :type out_channels: int
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv2 = nn.Conv2d(out_channels, 3, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: tensor, input tensor.
        :type x: tensor

        :return: tensor, output tensor.
        :rtype: tensor
        """
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        return out


class RecurResNetSR(nn.Module):
    """Recurrence residual NN for SR."""

    def __init__(
        self,
        block: nn.Module,
        num_blocks: list,
        out_channels: int = 3,
        depth: int = 3,
        width: int = 1,
        scale_factor: int = 4,
    ):
        """
        Init function.

        :param block: nn.Module, residual block.
        :type block: nn.Module
        :param num_blocks: list, number of blocks.
        :type num_blocks: list
        :param out_channels: int, output channel.
        :type out_channels: int
        :param depth: int, depth.
        :type depth: int
        :param width: int, width.
        :type width: int
        :param scale_factor: int, scale factor.
        :type scale_factor: int
        """
        super(RecurResNetSR, self).__init__()
        assert (depth - 3) % 4 == 0, \
            "Depth not compatible with recurrent architectue."
        self.iters = (depth - 3) // 4
        self.in_planes = int(64 * width)
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(3, int(64 * width), kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * width))

        layers = []
        for i in range(len(num_blocks)):
            layers.append(self._make_layer(block, int(64 * width),
                          num_blocks[i], stride=1, iters=self.iters))

        self.recur_block = nn.Sequential(*layers)

        self.upsample_block = UpsampleBlock(in_channels=int(64 * width),
                                            out_channels=int(32 * width),
                                            scale_factor=scale_factor)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int,
                    stride: int, iters: int) -> nn.Module:
        """
        Make layer.

        :param block: nn.Module, residual block.
        :type block: nn.Module
        :param planes: int, output channel.
        :type planes: int
        :param num_blocks: int, number of blocks.
        :type num_blocks: int
        :param stride: int, stride.
        :type stride: int
        :param iters: int, number of iterations.
        :type iters: int

        :return: nn.Module, layer.
        :rtype: nn.Module
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd, iters))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        :param x: tensor, input tensor.
        :type x: tensor

        :return: tensor, output tensor.
        :rtype: tensor
        """
        global bn_tracker
        bn_tracker = 0
        out = F.relu(self.bn1(self.conv1(x)))

        for i in range(self.iters):
            out = self.recur_block(out)
            bn_tracker += 1

        out = self.upsample_block(out)
        return out
