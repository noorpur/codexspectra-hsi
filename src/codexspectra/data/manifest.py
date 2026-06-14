from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CubeRecord:
    cube_id: str
    path: str
    extension: str
    group_id: str
    label: str
    checksum: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return "unavailable"


def infer_label(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    known = ["ink", "substrate", "paper", "parchment", "pigment", "background", "aged"]
    for token in known:
        if any(token in p for p in parts):
            return token
    return "unlabeled"


def infer_group(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return path.stem.split("_")[0]


def build_manifest(raw_root: str | Path, extensions: list[str]) -> pd.DataFrame:
    root = Path(raw_root)
    extensions = [e.lower() for e in extensions]
    records: list[CubeRecord] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            records.append(
                CubeRecord(
                    cube_id=path.stem,
                    path=str(path),
                    extension=path.suffix.lower(),
                    group_id=infer_group(path, root),
                    label=infer_label(path),
                    checksum=sha256_file(path),
                )
            )
    return pd.DataFrame([r.__dict__ for r in records])


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
