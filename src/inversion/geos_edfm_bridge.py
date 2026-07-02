from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def wslpath(path: Path) -> str:
    completed = subprocess.run(
        ["wsl", "wslpath", "-a", path.as_posix()],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows-to-WSL bridge for GEOS/EDFM forward simulation.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    script = Path(__file__).with_name("geos_edfm_forward_wsl.py").resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not script.exists():
        raise FileNotFoundError(f"WSL forward script not found: {script}")

    command = ["wsl", "python3", wslpath(script), "--run-dir", wslpath(run_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
