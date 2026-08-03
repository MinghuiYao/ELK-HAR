"""Installed compatibility exports for ``ELK_block``."""

from torch import nn

from elk_har.layers import (
    ELKBlock,
    ReparamLargeKernelConv,
    conv_bn,
    conv_bn_relu,
    fuse_bn,
    fuse_conv_bn,
    get_conv2d,
)
from elk_har.legacy import ELK


def get_bn(channels):
    return nn.BatchNorm2d(channels)


__all__ = [
    "ELK",
    "ELKBlock",
    "ReparamLargeKernelConv",
    "conv_bn",
    "conv_bn_relu",
    "fuse_bn",
    "fuse_conv_bn",
    "get_bn",
    "get_conv2d",
]
