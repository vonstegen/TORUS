#!/usr/bin/env python3
"""EXP-RPM-SITE-DISCOVERY sweep summarizer CLI.

Usage:
  .venv/bin/python site-discovery-summarize.py \
      --run-dir runs/r/EXP-RPM-SITE-DISCOVERY/<timestamp>

Applies the frozen rules from the manifest via
examples/site_discovery_summary.py; writes per-site
site_cal_summary.json and the sweep-level sweep_summary.json.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "site_discovery_summary",
        Path(__file__).resolve().parent
        / "examples" / "site_discovery_summary.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["site_discovery_summary"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"error: not a directory: {args.run_dir}", file=sys.stderr)
        return 2
    m = _load_module()
    summary = m.summarize_sweep(args.run_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["decision"] != "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
