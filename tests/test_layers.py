import torch

from ELK_block import ELK
from elk_har.layers import ELKBlock, ReparamLargeKernelConv


def test_tuple_kernel_reparameterization_preserves_output() -> None:
    layer = ReparamLargeKernelConv(
        4,
        4,
        kernel_size=(11, 1),
        stride=1,
        groups=4,
        small_kernel=(3, 1),
        small_kernel_merged=False,
    ).eval()
    inputs = torch.randn(2, 4, 32, 7)
    with torch.no_grad():
        expected = layer(inputs)
        layer.merge_kernel()
        actual = layer(inputs)

    assert hasattr(layer, "reparam")
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_elk_block_trains_with_both_kernel_branches() -> None:
    block = ELKBlock(4, 8, (11, 1), (3, 1))
    assert hasattr(block.large_kernel, "large_branch")
    assert hasattr(block.large_kernel, "small_branch")

    inputs = torch.randn(2, 4, 32, 7, requires_grad=True)
    outputs = block(inputs)
    assert outputs.shape == (2, 8, 32, 7)
    outputs.mean().backward()
    assert inputs.grad is not None


def test_original_elk_keyword_signature() -> None:
    block = ELK(
        in_channels=4,
        dw_channels=8,
        block_lk_size=(11, 1),
        small_kernel=(3, 1),
        small_kernel_merged=False,
    )
    assert block(torch.randn(2, 4, 32, 7)).shape == (2, 8, 32, 7)
