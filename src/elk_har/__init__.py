"""Effective large-kernel models for wearable human activity recognition."""

from .layers import ELKBlock, ReparamLargeKernelConv
from .models import ELKCNN, BaseCNN, LargeKernelCNN

__all__ = ["BaseCNN", "ELKBlock", "ELKCNN", "LargeKernelCNN", "ReparamLargeKernelConv"]
__version__ = "0.2.0"
