"""Large-kernel blocks with train-time multi-branch reparameterization."""

from typing import Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Pair = tuple[int, int]


def _pair(value: Union[int, Pair]) -> Pair:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("Expected an integer or a pair.")
    return int(value[0]), int(value[1])


def _padding(kernel_size: Union[int, Pair]) -> Pair:
    kernel = _pair(kernel_size)
    return kernel[0] // 2, kernel[1] // 2


def get_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Pair],
    stride: Union[int, Pair],
    padding: Union[int, Pair],
    dilation: Union[int, Pair],
    groups: int,
    bias: bool,
) -> nn.Conv2d:
    """Construct the portable convolution used by the reference artifact."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Pair],
    stride: Union[int, Pair] = 1,
    padding: Union[int, Pair, None] = None,
    groups: int = 1,
    dilation: Union[int, Pair] = 1,
) -> nn.Sequential:
    if padding is None:
        padding = _padding(kernel_size)
    return nn.Sequential(
        get_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        ),
        nn.BatchNorm2d(out_channels),
    )


def conv_bn_relu(*args, **kwargs) -> nn.Sequential:
    result = conv_bn(*args, **kwargs)
    result.add_module("activation", nn.ReLU(inplace=True))
    return result


def fuse_conv_bn(convolution: nn.Conv2d, batch_norm: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
    """Return the equivalent convolution kernel and bias in evaluation mode."""
    kernel = convolution.weight
    bias = (
        convolution.bias
        if convolution.bias is not None
        else torch.zeros(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    )
    scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
    fused_bias = batch_norm.bias + (bias - batch_norm.running_mean) * scale
    return fused_kernel, fused_bias


class ReparamLargeKernelConv(nn.Module):
    """Large and small depthwise branches that can be fused for inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Pair],
        stride: Union[int, Pair],
        groups: int,
        small_kernel: Union[int, Pair, None],
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.small_kernel = _pair(small_kernel) if small_kernel is not None else None
        self.stride = _pair(stride)
        self.groups = groups
        padding = _padding(self.kernel_size)
        if self.small_kernel is not None and any(
            small > large for small, large in zip(self.small_kernel, self.kernel_size)
        ):
            raise ValueError("small_kernel cannot be larger than kernel_size.")

        if small_kernel_merged:
            self.reparam = get_conv2d(
                in_channels,
                out_channels,
                self.kernel_size,
                self.stride,
                padding,
                1,
                groups,
                True,
            )
        else:
            self.large_branch = conv_bn(
                in_channels,
                out_channels,
                self.kernel_size,
                stride=self.stride,
                padding=padding,
                groups=groups,
            )
            if self.small_kernel is not None:
                self.small_branch = conv_bn(
                    in_channels,
                    out_channels,
                    self.small_kernel,
                    stride=self.stride,
                    padding=_padding(self.small_kernel),
                    groups=groups,
                )

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, "reparam"):
            return self.reparam(inputs)
        output = self.large_branch(inputs)
        if hasattr(self, "small_branch"):
            output = output + self.small_branch(inputs)
        return output

    def equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        if hasattr(self, "reparam"):
            return self.reparam.weight, self.reparam.bias
        kernel, bias = fuse_conv_bn(self.large_branch[0], self.large_branch[1])
        if hasattr(self, "small_branch"):
            small_kernel, small_bias = fuse_conv_bn(self.small_branch[0], self.small_branch[1])
            delta_h = self.kernel_size[0] - self.small_kernel[0]
            delta_w = self.kernel_size[1] - self.small_kernel[1]
            padding = (
                delta_w // 2,
                delta_w - delta_w // 2,
                delta_h // 2,
                delta_h - delta_h // 2,
            )
            kernel = kernel + F.pad(small_kernel, padding)
            bias = bias + small_bias
        return kernel, bias

    def merge_kernel(self) -> None:
        """Fuse training branches into one convolution without changing outputs."""
        if hasattr(self, "reparam"):
            return
        kernel, bias = self.equivalent_kernel_bias()
        large = self.large_branch[0]
        reparam = get_conv2d(
            large.in_channels,
            large.out_channels,
            self.kernel_size,
            self.stride,
            _padding(self.kernel_size),
            large.dilation,
            self.groups,
            True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.large_branch
        if hasattr(self, "small_branch"):
            del self.small_branch

    # Original method name.
    get_equivalent_kernel_bias = equivalent_kernel_bias


class ELKBlock(nn.Module):
    """Pointwise expansion, reparameterizable large kernel, and residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        large_kernel: Union[int, Pair],
        small_kernel: Union[int, Pair] = 3,
        small_kernel_merged: bool = False,
    ) -> None:
        super().__init__()
        self.expand = conv_bn_relu(in_channels, out_channels, 1, padding=0)
        self.large_kernel = ReparamLargeKernelConv(
            out_channels,
            out_channels,
            large_kernel,
            stride=1,
            groups=out_channels,
            small_kernel=small_kernel,
            small_kernel_merged=small_kernel_merged,
        )
        self.activation = nn.ReLU(inplace=True)
        self.project = conv_bn(out_channels, out_channels, 1, padding=0)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.project(self.activation(self.large_kernel(self.expand(inputs))))
        return self.shortcut(inputs) + output


# Original public names.
ELK = ELKBlock
fuse_bn = fuse_conv_bn

__all__ = [
    "ELK",
    "ELKBlock",
    "ReparamLargeKernelConv",
    "conv_bn",
    "conv_bn_relu",
    "fuse_bn",
    "fuse_conv_bn",
    "get_conv2d",
]
