"""Write a GNU sha256sum-compatible manifest for immutable release artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(artifacts: list[Path], output: Path) -> None:
    resolved_output = output.resolve()
    unique: dict[str, Path] = {}
    for artifact in artifacts:
        if not artifact.is_file():
            raise ValueError(f"release artifact is not a file: {artifact}")
        if artifact.resolve() == resolved_output:
            raise ValueError("checksum manifest cannot checksum itself")
        if artifact.name in unique:
            raise ValueError(f"duplicate release artifact name: {artifact.name}")
        unique[artifact.name] = artifact

    if not unique:
        raise ValueError("at least one release artifact is required")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256(path)}  {name}\n" for name, path in sorted(unique.items())]
    output.write_text("".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    write_manifest(args.artifacts, args.output)


if __name__ == "__main__":
    main()
