"""Installed compatibility exports for the original ``Model`` module."""

from elk_har.legacy import ELK_CNN, LK_CNN, Base_CNN
from elk_har.models import ELKCNN, BaseCNN, LargeKernelCNN

__all__ = ["BaseCNN", "Base_CNN", "ELKCNN", "ELK_CNN", "LK_CNN", "LargeKernelCNN"]
