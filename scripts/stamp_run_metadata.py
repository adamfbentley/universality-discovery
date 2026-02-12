"""
stamp_run_metadata.py

Lightweight provenance stamping for long experimental projects.

Usage examples (Windows):
  C:/Users/adamf/anaconda3/python.exe scripts/stamp_run_metadata.py results_exp52d_full --note "Exp 52d full run"
  C:/Users/adamf/anaconda3/python.exe scripts/stamp_run_metadata.py results_exp50r --command "python experiments/50r_kpz_alpha_only.py"

This script writes (into the given output directory):
  - run_metadata.json
  - git_status_porcelain.txt
  - git_untracked_files.txt
  - git_diff.patch               (optional; can be large)
  - pip_freeze.txt               (optional; can be large)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    branch: str | None
    dirty: bool | None


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )


def _safe_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(text, encoding="utf-8")


def _safe_write_bytes(path: Path, data: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_bytes(data)


def _git_info(repo_root: Path) -> tuple[GitInfo, dict[str, str]]:
    # Best-effort: if git isn't available or we're not in a repo, return Nones.
    cp_commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    cp_branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    cp_status = _run(["git", "status", "--porcelain"], cwd=repo_root)

    if cp_commit.returncode != 0:
        return GitInfo(commit=None, branch=None, dirty=None), {
            "git_error": (cp_commit.stderr or cp_commit.stdout or "").strip()
        }

    commit = cp_commit.stdout.strip()
    branch = cp_branch.stdout.strip() if cp_branch.returncode == 0 else None
    dirty = bool(cp_status.stdout.strip()) if cp_status.returncode == 0 else None
    return GitInfo(commit=commit, branch=branch, dirty=dirty), {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stamp run metadata into a results directory.")
    parser.add_argument("out_dir", type=str, help="Directory to write metadata files into.")
    parser.add_argument("--note", type=str, default="", help="Freeform note to attach to this run.")
    parser.add_argument(
        "--command",
        type=str,
        default="",
        help="Command that produced these results (for provenance).",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=".",
        help="Path to repo root (used for git status/diff). Default: current directory.",
    )
    parser.add_argument(
        "--no_git_diff",
        action="store_true",
        help="Do not write git_diff.patch (can be large).",
    )
    parser.add_argument(
        "--no_pip_freeze",
        action="store_true",
        help="Do not write pip_freeze.txt (can be large).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metadata files (default is to keep first write).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root).resolve()

    now = datetime.now().astimezone().isoformat()

    git_info, git_extra = _git_info(repo_root)

    metadata: dict[str, Any] = {
        "timestamp": now,
        "note": args.note,
        "command": args.command,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "git": asdict(git_info),
        **git_extra,
    }

    _safe_write_text(
        out_dir / "run_metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )

    # Git status / diff (best effort)
    cp_status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    if cp_status.returncode == 0:
        _safe_write_text(
            out_dir / "git_status_porcelain.txt",
            cp_status.stdout,
            overwrite=args.overwrite,
        )

    cp_untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    if cp_untracked.returncode == 0:
        _safe_write_text(
            out_dir / "git_untracked_files.txt",
            cp_untracked.stdout,
            overwrite=args.overwrite,
        )

    if not args.no_git_diff:
        cp_diff = _run(["git", "diff"], cwd=repo_root)
        if cp_diff.returncode == 0:
            _safe_write_text(
                out_dir / "git_diff.patch",
                cp_diff.stdout,
                overwrite=args.overwrite,
            )

    if not args.no_pip_freeze:
        # Use the current interpreter to run pip. This ensures we freeze the same environment.
        cp_freeze = _run([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
        if cp_freeze.returncode == 0:
            _safe_write_text(
                out_dir / "pip_freeze.txt",
                cp_freeze.stdout,
                overwrite=args.overwrite,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

