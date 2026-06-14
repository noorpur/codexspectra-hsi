#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from codexspectra.data.manifest import build_manifest, save_manifest
from codexspectra.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["data"]
    df = build_manifest(data["raw_root"], data["cube_extensions"])
    save_manifest(df, data["manifest_path"])
    print(f"Wrote {len(df)} manifest rows to {data['manifest_path']}")
    if len(df) == 0:
        print("No cubes found. Add raw HYPERDOC files or run scripts/run_smoke_test.py.")


if __name__ == "__main__":
    main()
