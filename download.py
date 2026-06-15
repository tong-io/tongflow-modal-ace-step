"""Modal download entry for ace-step.

Run:
  modal run download.py::download

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal



_cfg: dict[str, Any] = {}


def _repo_dirs() -> list[tuple[str, str]]:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and repos:
        out: list[tuple[str, str]] = []
        for r in repos:
            if isinstance(r, dict) and r.get("repoId"):
                rid = str(r["repoId"])
                rev = str(r.get("revision") or "")
                out.append((rid, rev))
        if out:
            return out
    return [
        ("ACE-Step/acestep-v15-xl-base", ""),
        ("ACE-Step/acestep-5Hz-lm-4B", ""),
    ]


volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==1.6.0"),
    volumes={"/models": volume},
    timeout=1800,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    for repo_id, revision in _repo_dirs():
        model_dir = f"/models/{repo_id}"
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"Model already exists at {model_dir}, skipping")
            continue
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            revision=revision or None,
        )
        print(f"Model downloaded to {model_dir}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
