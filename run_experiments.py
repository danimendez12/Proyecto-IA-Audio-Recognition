from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import wandb

from configs.experiments import EXPERIMENTS
from train import train


def _args_for_run(base: argparse.Namespace, mode: str, experiment: dict[str, Any]) -> SimpleNamespace:
    """Build a complete train namespace for one reproducible experiment."""
    run_name = f"model-b-{mode}-{experiment['name']}"
    values = vars(base).copy()
    values.update(experiment)
    values.update(
        {
            "augment_mode": "full" if mode == "augmented" else "none",
            "run_name": run_name,
            "wandb_group": mode,
            "wandb_tags": ["model-b", mode, experiment["name"]],
            "checkpoint_path": base.checkpoint_root / run_name / "best_model.pt",
        }
    )
    return SimpleNamespace(**values)


def main() -> None:
    """Run the selected three-experiment group and update the shared summary."""
    parser = argparse.ArgumentParser(description="Run six Model B experiments")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/speech_commands_10"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data/spectrogram_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--wandb-project", type=str, default="speech-commands-10")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--noise-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["base", "augmented", "all"],
        default="all",
        help="Experiment group to run; use base and augmented in separate invocations to split the workload",
    )
    args = parser.parse_args()

    common = {
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
        "build_cache": False,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_mels": args.n_mels,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "noise_dir": args.noise_dir,
        "scheduler": "cosine",
        "early_stopping_patience": 5,
        "augment_train": False,
        "snr_min": 5.0,
        "snr_max": 20.0,
        "max_shift_ms": 100.0,
        "p_augment": 1.0,
        "freq_mask_param": 8,
        "time_mask_param": 16,
        "num_masks": 2,
        "use_pitch_shift": False,
        "use_time_stretch": False,
    }
    results: list[dict[str, Any]] = []
    modes = ("base", "augmented") if args.mode == "all" else (args.mode,)
    for mode in modes:
        for experiment in EXPERIMENTS:
            run_args = _args_for_run(SimpleNamespace(**{**common, "checkpoint_root": args.checkpoint_root}), mode, experiment)
            results.append(train(run_args))

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.csv"
    fields = ["mode", "experiment", "learning_rate", "width_mult", "dropout", "batch_size", "weight_decay", "best_val_f1", "test_accuracy", "test_f1", "gap", "parameters"]
    existing_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                existing_rows[(row["mode"], row["experiment"])] = row

    for result in results:
        config = result["config"]
        mode = "augmented" if result["mode"] == "full" else "base"
        existing_rows[(mode, config["name"])] = {
            "mode": mode,
            "experiment": config["name"],
            "learning_rate": config["learning_rate"],
            "width_mult": config["width_mult"],
            "dropout": config["dropout"],
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "best_val_f1": result["best_val_f1"],
            "test_accuracy": result["test_accuracy"],
            "test_f1": result["test_f1"],
            "gap": result["gap"],
            "parameters": result["parameters"],
        }

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in existing_rows.values():
            writer.writerow(row)

    summary_run = wandb.init(project=args.wandb_project, job_type="summary", mode=args.wandb_mode, tags=["comparison"])
    with summary_path.open(newline="", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    summary_run.log({"experiments/summary": wandb.Table(data=[list(row[field] for field in fields) for row in summary_rows], columns=fields)})
    summary_run.finish()
    print(f"Summary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()