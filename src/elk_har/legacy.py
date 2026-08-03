"""Adapters for the public names in the original source-only release."""

from .layers import ELKBlock
from .models import ELKCNN, BaseCNN, LargeKernelCNN


class ELK(ELKBlock):
    def __init__(
        self,
        in_channels,
        dw_channels,
        block_lk_size,
        small_kernel,
        small_kernel_merged=False,
    ):
        super().__init__(
            in_channels,
            dw_channels,
            block_lk_size,
            small_kernel,
            small_kernel_merged=small_kernel_merged,
        )


class _LegacyModelProperties:
    @property
    def layer(self):
        return self.features

    @property
    def ada_pool(self):
        return self.pool

    @property
    def fc(self):
        return self.classifier


class Base_CNN(_LegacyModelProperties, BaseCNN):
    def __init__(self, train_shape, category):
        super().__init__(train_shape, category)


class LK_CNN(_LegacyModelProperties, LargeKernelCNN):
    def __init__(self, train_shape, category):
        super().__init__(train_shape, category)


class ELK_CNN(_LegacyModelProperties, ELKCNN):
    def __init__(self, train_shape, category):
        super().__init__(train_shape, category)


__all__ = ["Base_CNN", "ELK", "ELK_CNN", "LK_CNN"]
