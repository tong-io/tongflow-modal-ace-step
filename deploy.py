"""Modal deploy entry for ace-step.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, cast

import modal
from tongflow import deploy




_cfg: dict[str, Any] = {}
_ace = _cfg.get("aceStep") if isinstance(_cfg.get("aceStep"), dict) else {}
REPO_URL = str(
    _ace.get("gitUrl") or "https://github.com/ACE-Step/ACE-Step-1.5.git",
)
REPO_DIR = str(_ace.get("repoDir") or "/app/ACE-Step-1.5")


def _dit_lm_ids() -> tuple[str, str]:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and len(repos) >= 2:
        a = repos[0] if isinstance(repos[0], dict) else {}
        b = repos[1] if isinstance(repos[1], dict) else {}
        if a.get("repoId") and b.get("repoId"):
            return str(a["repoId"]), str(b["repoId"])
    return "ACE-Step/acestep-v15-xl-base", "ACE-Step/acestep-5Hz-lm-4B"


DIT_REPO_ID, LM_REPO_ID = _dit_lm_ids()
DIT_MODEL_DIR = f"/models/{DIT_REPO_ID}"
LM_MODEL_DIR = f"/models/{LM_REPO_ID}"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.gen_music import GenMusicInput, GenMusicOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset
from tongflow.slots import node_slot


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1")
    .pip_install("tongflow==0.1.0")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"pip install --no-deps -e {REPO_DIR}/acestep/third_parts/nano-vllm",
        f"grep -viE '^(flash-attn|triton)' {REPO_DIR}/requirements.txt | pip install -r /dev/stdin",
        f"pip install -e {REPO_DIR} --no-deps",
    )
)

with image.imports():
    import io
    import torch
    import soundfile as sf
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import generate_music, GenerationParams, GenerationConfig


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    timeout=600,
)
class Inference:
    @modal.enter()
    def load(self):
        ckpt_dir = os.path.join(REPO_DIR, "checkpoints")
        if os.path.exists(ckpt_dir):
            os.remove(ckpt_dir) if os.path.islink(ckpt_dir) else None
        os.symlink(DIT_MODEL_DIR, ckpt_dir)

        self.dit_handler = AceStepHandler()
        self.dit_handler.initialize_service(
            project_root=REPO_DIR,
            config_path="acestep-v15-xl-base",
            device="cuda",
        )
        self.llm_handler = LLMHandler()
        self.llm_handler.initialize(
            checkpoint_dir="/models",
            lm_model_path=LM_REPO_ID,
            backend="vllm",
            device="cuda",
        )

    def _generate_raw(
        self,
        lyrics: str = "",
        tags: str = "",
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: str = "",
        language: str = "zh",
        seed: int = -1,
    ) -> bytes:
        params = GenerationParams(
            lyrics=lyrics,
            caption=tags,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            vocal_language=language,
            seed=seed,
        )
        config = GenerationConfig(batch_size=1)
        result = generate_music(self.dit_handler, self.llm_handler, params, config)

        if not result.success or not result.audios:
            raise RuntimeError(result.error or result.status_message)

        audio = result.audios[0]
        tensor = audio["tensor"]
        sr = audio["sample_rate"]

        buf = io.BytesIO()
        sf.write(buf, tensor.cpu().numpy().T, sr, format="FLAC")
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        lyrics: str = "",
        tags: str = "",
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: str = "",
        language: str = "zh",
        seed: int = -1,
    ) -> bytes:
        return self._generate_raw(
            lyrics=lyrics,
            tags=tags,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            language=language,
            seed=seed,
        )

    @modal.method()
    @node_slot(NodeSlots.GEN_MUSIC)
    def gen_music(self, input: GenMusicInput) -> GenMusicOutput:
        lyrics = input.lyrics or input.text or ""
        tags = input.tags or ""
        try:
            raw = self._generate_raw(
                lyrics=lyrics,
                tags=tags,
                duration=input.duration if input.duration is not None else 30.0,
                bpm=int(input.bpm) if input.bpm is not None else None,
                keyscale=input.keyscale or "",
                language=input.language or "zh",
                seed=int(input.seed) if input.seed is not None else -1,
            )
        except Exception as e:
            return GenMusicOutput(success=False, error=str(e))
        return GenMusicOutput(success=True, audio=asset(raw, mime="audio/wav"))
