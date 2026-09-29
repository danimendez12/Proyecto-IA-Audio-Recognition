from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, random_split

from models.model_b import MobileNetStyleCNN
from preprocessing.dataset import LABELS, MelSpectrogramDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX and verify output parity")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/best_model.pt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data/spectrogram_cache"))
    parser.add_argument("--onnx-path", type=Path, default=Path("./checkpoints/model.onnx"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=5)
    return parser.parse_args()


def build_test_loader(cache_dir: Path, seed: int, num_samples: int) -> DataLoader:
    dataset = MelSpectrogramDataset(cache_dir)
    total = len(dataset)
    train_len = int(total * 0.70)
    val_len = int(total * 0.15)
    test_len = total - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    _, _, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    sample_count = min(num_samples, len(test_set))
    subset, _ = random_split(
        test_set,
        [sample_count, len(test_set) - sample_count],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return DataLoader(subset, batch_size=1, shuffle=False)


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    model = MobileNetStyleCNN(
        num_classes=int(config.get("num_classes", len(LABELS))),
        width_mult=float(config.get("width_mult", 1.0)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = build_test_loader(args.cache_dir, args.seed, args.num_samples)
    first_input, _ = next(iter(test_loader))

    args.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        first_input,
        args.onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])

    for idx, (input_tensor, _) in enumerate(test_loader, start=1):
        with torch.no_grad():
            torch_output = model(input_tensor).cpu().numpy()

        ort_output = session.run(["logits"], {"input": input_tensor.cpu().numpy()})[0]

        np.testing.assert_allclose(torch_output, ort_output, rtol=1e-03, atol=1e-04)
        print(f"Sample {idx}: PyTorch and ONNX outputs match.")

    print(f"ONNX model exported and verified at: {args.onnx_path.resolve()}")


if __name__ == "__main__":
    main()
