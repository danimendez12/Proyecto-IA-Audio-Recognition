from __future__ import annotations

import argparse
import contextlib
import shutil
import ssl
import tarfile
import urllib.request
from urllib.error import URLError
from pathlib import Path

DATASET_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
TARGET_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SPLIT_FILES = ("validation_list.txt", "testing_list.txt")


def download_speech_commands(raw_root: Path, insecure: bool = False) -> Path:
    """Download and extract Speech Commands, optionally bypassing TLS checks."""
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / ARCHIVE_NAME

    if not archive_path.exists():
        context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            with contextlib.closing(urllib.request.urlopen(DATASET_URL, context=context)) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            temporary_path.replace(archive_path)
        except URLError as error:
            temporary_path.unlink(missing_ok=True)
            if not insecure:
                raise RuntimeError(
                    "Could not verify the dataset server certificate. "
                    "Fix the local proxy/CA configuration or retry with --insecure-download "
                    "on a trusted network."
                ) from error
            raise

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
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="Disable TLS certificate verification for the download only; use only on a trusted network",
    )
    args = parser.parse_args()

    source_dir = download_speech_commands(args.root, insecure=args.insecure_download)
    copy_target_classes(source_dir, args.output_dir, TARGET_COMMANDS)

    print(f"Prepared 10-class dataset at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
