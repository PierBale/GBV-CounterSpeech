#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import requests
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from card_routed_rag.io_utils import read_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Download source documents listed in configs/sources.yaml.")
    ap.add_argument("--config", default="configs/sources.yaml")
    ap.add_argument("--output-dir", default="data/sources")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in cfg.get("sources", []):
        url = src["url"]
        file_name = src["file_name"]
        dest = out_dir / file_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {file_name} already exists")
            continue
        print(f"[download] {url} -> {dest}")
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "card-routed-rag/0.2"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"[ok] {dest} ({dest.stat().st_size} bytes)")
        except Exception as exc:
            print(f"[warn] failed to download {url}: {exc}")


if __name__ == "__main__":
    main()
