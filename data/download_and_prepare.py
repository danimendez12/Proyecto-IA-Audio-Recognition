from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchaudio.datasets.utils import download_url, extract_archive

DATASET_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
TARGET_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def download_speech_commands(raw_root: Path) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / ARCHIVE_NAME

    if not archive_path.exists():
        download_url(DATASET_URL, str(raw_root), ARCHIVE_NAME)

    extracted_dir = raw_root / "speech_commands_v0.02"
    if not extracted_dir.exists():
        extract_archive(str(archive_path), str(raw_root))

    return extracted_dir


def copy_target_classes(source_dir: Path, output_dir: Path, commands: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for command in commands:
        src = source_dir / command
        if not src.exists():
            raise FileNotFoundError(f"Missing class directory: {src}")

        dst = output_dir / command
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Speech Commands v0.02 and keep only target classes.")
    parser.add_argument("--root", type=Path, default=Path("./data"), help="Data root directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/speech_commands_10"),
        help="Destination directory for filtered classes",
    )
    args = parser.parse_args()

    source_dir = download_speech_commands(args.root)
    copy_target_classes(source_dir, args.output_dir, TARGET_COMMANDS)

    print(f"Prepared 10-class dataset at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
