from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
TARGET_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SPLIT_FILES = ("validation_list.txt", "testing_list.txt")


def download_speech_commands(raw_root: Path) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / ARCHIVE_NAME

    if not archive_path.exists():
        urllib.request.urlretrieve(DATASET_URL, archive_path)

    if not all((raw_root / command).is_dir() for command in TARGET_COMMANDS):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(raw_root)

    return raw_root


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

    for split_file in SPLIT_FILES:
        source_list = source_dir / split_file
        if not source_list.exists():
            raise FileNotFoundError(f"Missing official split file: {source_list}")

        selected = []
        for line in source_list.read_text(encoding="utf-8").splitlines():
            relative_path = Path(line.strip())
            if relative_path.parts and relative_path.parts[0] in commands:
                selected.append(line)
        (output_dir / split_file).write_text("\n".join(selected) + "\n", encoding="utf-8")

    noise_source = source_dir / "_background_noise_"
    if noise_source.exists():
        noise_destination = output_dir / "_background_noise_"
        if noise_destination.exists():
            shutil.rmtree(noise_destination)
        shutil.copytree(noise_source, noise_destination)


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
