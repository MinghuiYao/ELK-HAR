"""Dataset-free ELK-HAR forward and reparameterization example."""

import torch

from elk_har import ELKCNN


def main() -> None:
    torch.manual_seed(7)
    model = ELKCNN((1, 128, 9), 6).eval()
    inputs = torch.randn(4, 1, 128, 9)
    with torch.no_grad():
        before = model(inputs)
        model.structural_reparameterize()
        after = model(inputs)
    print(f"logits: {tuple(after.shape)}")
    print(f"maximum reparameterization error: {(before - after).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
