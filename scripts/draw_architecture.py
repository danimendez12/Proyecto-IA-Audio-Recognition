from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_b import MobileNetStyleCNN


def main() -> None:
    """Print each leaf layer output shape and parameter count."""
    parser = argparse.ArgumentParser(description="Print Model B architecture summary")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    model = MobileNetStyleCNN(width_mult=args.width_mult, dropout=args.dropout).eval()

    def hook(name: str):
        def print_layer(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
            del inputs
            shape = tuple(output.shape) if isinstance(output, Tensor) else str(type(output))
            parameters = sum(parameter.numel() for parameter in module.parameters())
            print(f"{name:30} output={shape!s:18} parameters={parameters}")

        return print_layer

    handles = [
        module.register_forward_hook(hook(name))
        for name, module in model.named_modules()
        if name and not list(module.children())
    ]
    with torch.no_grad():
        model(torch.zeros(1, 1, 64, 101))
    for handle in handles:
        handle.remove()
    print(f"total_trainable_parameters={model.count_parameters()}")
    print(f"estimated_macs={model.estimate_macs()}")


if __name__ == "__main__":
    main()