"""Reference ELK-HAR classifiers."""

from collections.abc import Sequence

from torch import Tensor, nn

from .layers import ELKBlock


class _Classifier(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int, features: nn.Module) -> None:
        super().__init__()
        if len(input_shape) < 2 or num_classes <= 0:
            raise ValueError("input_shape and num_classes are invalid.")
        self.num_modalities = int(input_shape[-1])
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, self.num_modalities))
        self.classifier = nn.Linear(256 * self.num_modalities, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError("Expected [batch, 1, time, modalities].")
        if inputs.shape[-1] != self.num_modalities:
            raise ValueError(
                f"Model configured for {self.num_modalities} modalities, got {inputs.shape[-1]}."
            )
        return self.classifier(self.pool(self.features(inputs)).flatten(start_dim=1))


def _conv_stage(in_channels, out_channels, kernel, stride, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class BaseCNN(_Classifier):
    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        features = nn.Sequential(
            _conv_stage(1, 64, (6, 1), (2, 1), (1, 0)),
            _conv_stage(64, 128, (6, 1), (2, 1), (1, 0)),
            _conv_stage(128, 256, (6, 1), (2, 1), (1, 0)),
        )
        super().__init__(input_shape, num_classes, features)


class LargeKernelCNN(_Classifier):
    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        features = nn.Sequential(
            _conv_stage(1, 64, (6, 1), (2, 1), (1, 0)),
            _conv_stage(64, 128, (31, 1), (10, 1), (5, 0)),
            _conv_stage(128, 256, (6, 1), (2, 1), (1, 0)),
        )
        super().__init__(input_shape, num_classes, features)


class ELKCNN(_Classifier):
    def __init__(self, input_shape: Sequence[int], num_classes: int) -> None:
        features = nn.Sequential(
            _conv_stage(1, 64, (6, 1), (2, 1), (1, 0)),
            ELKBlock(64, 128, (31, 1), (3, 1), small_kernel_merged=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _conv_stage(128, 256, (6, 1), (2, 1), (1, 0)),
        )
        super().__init__(input_shape, num_classes, features)

    def structural_reparameterize(self) -> None:
        for module in list(self.modules()):
            if hasattr(module, "merge_kernel"):
                module.merge_kernel()

    structural_reparam = structural_reparameterize


__all__ = ["BaseCNN", "ELKCNN", "LargeKernelCNN"]
