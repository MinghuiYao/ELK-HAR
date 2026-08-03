import pytest
import torch

from elk_har.models import ELKCNN, BaseCNN, LargeKernelCNN
from Model import ELK_CNN


@pytest.mark.parametrize("model_type", [BaseCNN, LargeKernelCNN, ELKCNN])
def test_reference_models_return_logits(model_type) -> None:
    model = model_type((1, 128, 9), 6)
    outputs = model(torch.randn(2, 1, 128, 9))
    assert outputs.shape == (2, 6)


def test_full_model_reparameterization_preserves_output() -> None:
    model = ELKCNN((1, 128, 9), 6).eval()
    inputs = torch.randn(2, 1, 128, 9)
    with torch.no_grad():
        expected = model(inputs)
        model.structural_reparameterize()
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_legacy_model_name() -> None:
    model = ELK_CNN((1, 128, 9), 6)
    assert model(torch.randn(1, 1, 128, 9)).shape == (1, 6)
    assert model.layer is model.features
    assert model.ada_pool is model.pool
    assert model.fc is model.classifier
