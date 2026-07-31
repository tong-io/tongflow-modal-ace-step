"""Modal deploy entry for ace-step.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Optional, cast

import modal
from tongflow import deploy




# Slots this plugin is the default implementation of: the node picker lists
# it first and a newly added node preselects it. Read statically by the
# scanner (never executed), so any SDK version imports this file fine.
TONGFLOW_DEFAULT_SLOTS = ["gen-music"]

_cfg: dict[str, Any] = {}
_ace = _cfg.get("aceStep") if isinstance(_cfg.get("aceStep"), dict) else {}
REPO_URL = str(
    _ace.get("gitUrl") or "https://github.com/ACE-Step/ACE-Step-1.5.git",
)
REPO_DIR = str(_ace.get("repoDir") or "/app/ACE-Step-1.5")
# Pin the upstream revision so redeploys are reproducible (main moves).
REPO_REV = str(_ace.get("gitRev") or "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0")

# DiT variants selectable per node (first entry per slot = default).
# xl-sft: best quality (Model Zoo "Very High"); xl-base: the only variant that
# supports extract/lego/complete; xl-turbo: 8-step distilled, fastest.
# Must stay a pure dict literal — the platform reads it by AST without import.
TONGFLOW_SLOT_MODELS = {
    "gen-music": [
        "acestep-v15-xl-sft",
        "acestep-v15-xl-base",
        "acestep-v15-xl-turbo",
    ],
    "music-repaint": [
        "acestep-v15-xl-sft",
        "acestep-v15-xl-base",
        "acestep-v15-xl-turbo",
    ],
    "music-cover": [
        "acestep-v15-xl-sft",
        "acestep-v15-xl-base",
        "acestep-v15-xl-turbo",
    ],
    "music-extract": ["acestep-v15-xl-base"],
    "music-lego": ["acestep-v15-xl-base"],
    "music-complete": ["acestep-v15-xl-base"],
}

DEFAULT_DIT = "acestep-v15-xl-sft"
# Every variant's weights live in the shared volume as its own HF snapshot.
# sft/base snapshots are full bundles (shared assets + nested variant dir);
# the turbo snapshot is flat — it IS the variant dir.
SFT_SNAP = "/models/ACE-Step/acestep-v15-xl-sft"
BASE_SNAP = "/models/ACE-Step/acestep-v15-xl-base"
TURBO_SNAP = "/models/ACE-Step/acestep-v15-xl-turbo"
LM_REPO_ID = "ACE-Step/acestep-5Hz-lm-4B"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.audio_describe import AudioDescribeInput, AudioDescribeOutput
from tongflow.models.gen_music import GenMusicInput, GenMusicOutput
from tongflow.models.music_brief import MusicBriefInput, MusicBriefOutput
from tongflow.models.music_complete import MusicCompleteInput, MusicCompleteOutput
from tongflow.models.music_cover import MusicCoverInput, MusicCoverOutput
from tongflow.models.music_extract import MusicExtractInput, MusicExtractOutput
from tongflow.models.music_lego import MusicLegoInput, MusicLegoOutput
from tongflow.models.music_repaint import MusicRepaintInput, MusicRepaintOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import current_model, node_slot


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # ffmpeg: torchaudio/torchcodec's decode fallback needs its shared libs —
    # without it, m4a/aac/webm reference audio fails as "Invalid reference audio"
    # (libsndfile alone only covers wav/flac/ogg/mp3).
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("tongflow==0.2.20", "fastapi[standard]")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        f"git -C {REPO_DIR} checkout {REPO_REV}",
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
    from acestep.constants import TASK_INSTRUCTIONS, TRACK_NAMES
    from acestep.inference import (
        GenerationConfig,
        GenerationParams,
        create_sample,
        generate_music,
        understand_music,
    )


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
        self._assemble_checkpoints()
        self.dit_handler = AceStepHandler()
        self._dit_config: Optional[str] = None
        self._ensure_dit(DEFAULT_DIT)
        self.llm_handler = LLMHandler()
        self.llm_handler.initialize(
            checkpoint_dir="/models",
            lm_model_path=LM_REPO_ID,
            backend="vllm",
            device="cuda",
        )

    def _assemble_checkpoints(self) -> None:
        """Build REPO_DIR/checkpoints with all three DiT variants linked in."""
        ckpt = os.path.join(REPO_DIR, "checkpoints")
        if os.path.islink(ckpt):
            os.remove(ckpt)
        os.makedirs(ckpt, exist_ok=True)

        def link(target: str, name: str) -> None:
            dst = os.path.join(ckpt, name)
            if not os.path.exists(dst) and os.path.exists(target):
                os.symlink(target, dst)

        # Shared assets (vae, embedding model, main-model shards) + the sft
        # variant dir all come from the sft bundle.
        for entry in os.listdir(SFT_SNAP):
            link(os.path.join(SFT_SNAP, entry), entry)
        # The base bundle contributes only its variant dir.
        link(os.path.join(BASE_SNAP, "acestep-v15-xl-base"), "acestep-v15-xl-base")
        # The turbo snapshot is flat: the snapshot dir IS the variant dir.
        link(TURBO_SNAP, "acestep-v15-xl-turbo")

    def _ensure_dit(self, config: str) -> None:
        """(Re)initialize the DiT when the requested variant differs."""
        if config == self._dit_config:
            return
        self.dit_handler.initialize_service(
            project_root=REPO_DIR,
            config_path=config,
            device="cuda",
        )
        self._dit_config = config

    def _pick_model(self, slot: str) -> str:
        allowed = TONGFLOW_SLOT_MODELS.get(slot) or [DEFAULT_DIT]
        m = current_model()
        return m if m in allowed else allowed[0]

    def _run(
        self,
        *,
        task_type: str = "text2music",
        lyrics: str = "",
        caption: str = "",
        duration: Optional[float] = None,
        bpm: Optional[int] = None,
        keyscale: str = "",
        language: str = "",
        seed: int = -1,
        reference_audio: Optional[str] = None,
        src_audio: Optional[str] = None,
        instruction: Optional[str] = None,
        repainting_start: float = 0.0,
        repainting_end: float = -1,
        audio_cover_strength: Optional[float] = None,
    ) -> bytes:
        kwargs: dict[str, Any] = dict(
            task_type=task_type,
            lyrics=lyrics,
            caption=caption,
            bpm=bpm,
            keyscale=keyscale,
            seed=seed,
            reference_audio=reference_audio,
            src_audio=src_audio,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
        )
        if duration is not None:
            kwargs["duration"] = duration
        if language:
            kwargs["vocal_language"] = language
        if instruction:
            kwargs["instruction"] = instruction
        if audio_cover_strength is not None:
            kwargs["audio_cover_strength"] = audio_cover_strength

        params = GenerationParams(**kwargs)
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

    @staticmethod
    def _as_wav(stack: contextlib.ExitStack, media: Any) -> str:
        """Materialize an incoming audio Asset as a WAV temp file.

        Upstream's decode fallback (torchaudio/torchcodec) is broken in this
        image (torchcodec wheel targets a different CUDA), so anything
        libsndfile can't read (m4a/aac/webm...) would fail as "Invalid
        reference audio". Transcoding through the ffmpeg CLI first makes the
        soundfile fast path always hit.
        """
        import subprocess
        import tempfile

        src = str(stack.enter_context(asset_as_path(media)))
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        stack.callback(lambda: os.path.exists(wav) and os.unlink(wav))
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", wav],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-300:]
            raise RuntimeError(f"could not decode audio input: {tail}")
        return wav

    @staticmethod
    def _norm_track(track: str) -> str:
        t = (track or "").strip().lower()
        if t not in TRACK_NAMES:
            raise RuntimeError(
                f"unknown track {track!r}; expected one of {', '.join(TRACK_NAMES)}"
            )
        return t

    @modal.method()
    @node_slot(NodeSlots.GEN_MUSIC)
    def gen_music(self, input: GenMusicInput) -> GenMusicOutput:
        try:
            self._ensure_dit(self._pick_model("gen-music"))
            with contextlib.ExitStack() as stack:
                ref_path: Optional[str] = None
                if input.ref_audio is not None:
                    ref_path = self._as_wav(stack, input.ref_audio)
                raw = self._run(
                    lyrics=input.lyrics or input.text or "",
                    caption=input.tags or "",
                    duration=input.duration if input.duration is not None else 30.0,
                    bpm=int(input.bpm) if input.bpm is not None else None,
                    keyscale=input.keyscale or "",
                    language=input.language or "zh",
                    seed=int(input.seed) if input.seed is not None else -1,
                    # text2music consumes reference_audio directly for
                    # style-transfer conditioning (3x10s sampled segments).
                    reference_audio=ref_path,
                )
        except Exception as e:
            return GenMusicOutput(success=False, error=str(e))
        return GenMusicOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_REPAINT)
    def music_repaint(self, input: MusicRepaintInput) -> MusicRepaintOutput:
        try:
            self._ensure_dit(self._pick_model("music-repaint"))
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                raw = self._run(
                    task_type="repaint",
                    instruction=TASK_INSTRUCTIONS["repaint"],
                    src_audio=src_path,
                    caption=input.text or "",
                    lyrics=input.lyrics or "",
                    repainting_start=float(input.start_time),
                    repainting_end=float(input.end_time),
                    audio_cover_strength=float(input.strength)
                    if input.strength is not None
                    else None,
                    seed=int(input.seed) if input.seed is not None else -1,
                )
        except Exception as e:
            return MusicRepaintOutput(success=False, error=str(e))
        return MusicRepaintOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_COVER)
    def music_cover(self, input: MusicCoverInput) -> MusicCoverOutput:
        try:
            self._ensure_dit(self._pick_model("music-cover"))
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                ref_path: Optional[str] = None
                if input.ref_audio is not None:
                    ref_path = self._as_wav(stack, input.ref_audio)
                raw = self._run(
                    task_type="cover",
                    instruction=TASK_INSTRUCTIONS["cover"],
                    src_audio=src_path,
                    reference_audio=ref_path,
                    caption=input.text or "",
                    lyrics=input.lyrics or "",
                    audio_cover_strength=float(input.strength)
                    if input.strength is not None
                    else 0.8,
                    seed=int(input.seed) if input.seed is not None else -1,
                )
        except Exception as e:
            return MusicCoverOutput(success=False, error=str(e))
        return MusicCoverOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_EXTRACT)
    def music_extract(self, input: MusicExtractInput) -> MusicExtractOutput:
        try:
            # extract is a base-only capability upstream.
            self._ensure_dit(self._pick_model("music-extract"))
            track = self._norm_track(input.track)
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                raw = self._run(
                    task_type="extract",
                    instruction=TASK_INSTRUCTIONS["extract"].format(
                        TRACK_NAME=track.upper()
                    ),
                    src_audio=src_path,
                    seed=int(input.seed) if input.seed is not None else -1,
                )
        except Exception as e:
            return MusicExtractOutput(success=False, error=str(e))
        return MusicExtractOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_LEGO)
    def music_lego(self, input: MusicLegoInput) -> MusicLegoOutput:
        try:
            self._ensure_dit(self._pick_model("music-lego"))
            track = self._norm_track(input.track)
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                raw = self._run(
                    task_type="lego",
                    instruction=TASK_INSTRUCTIONS["lego"].format(
                        TRACK_NAME=track.upper()
                    ),
                    src_audio=src_path,
                    caption=input.text or "",
                    lyrics=input.lyrics or "",
                    seed=int(input.seed) if input.seed is not None else -1,
                )
        except Exception as e:
            return MusicLegoOutput(success=False, error=str(e))
        return MusicLegoOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_COMPLETE)
    def music_complete(self, input: MusicCompleteInput) -> MusicCompleteOutput:
        try:
            self._ensure_dit(self._pick_model("music-complete"))
            tracks = [self._norm_track(t) for t in (input.tracks or [])]
            if tracks:
                instruction = TASK_INSTRUCTIONS["complete"].format(
                    TRACK_CLASSES=", ".join(t.upper() for t in tracks)
                )
            else:
                instruction = TASK_INSTRUCTIONS["complete_default"]
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                raw = self._run(
                    task_type="complete",
                    instruction=instruction,
                    src_audio=src_path,
                    caption=input.text or "",
                    seed=int(input.seed) if input.seed is not None else -1,
                )
        except Exception as e:
            return MusicCompleteOutput(success=False, error=str(e))
        return MusicCompleteOutput(success=True, audio=asset(raw, mime="audio/wav"))

    @modal.method()
    @node_slot(NodeSlots.MUSIC_BRIEF)
    def music_brief(self, input: MusicBriefInput) -> MusicBriefOutput:
        try:
            res = create_sample(
                self.llm_handler,
                query=input.text,
                instrumental=bool(input.instrumental),
                vocal_language=input.language or None,
            )
            if not getattr(res, "success", True) and getattr(res, "error", None):
                raise RuntimeError(str(res.error))
        except Exception as e:
            return MusicBriefOutput(success=False, error=str(e))
        return MusicBriefOutput(
            success=True,
            lyrics=getattr(res, "lyrics", None),
            tags=getattr(res, "caption", None),
            bpm=float(res.bpm) if getattr(res, "bpm", None) is not None else None,
            keyscale=getattr(res, "keyscale", None),
            duration=float(res.duration)
            if getattr(res, "duration", None) is not None
            else None,
            language=getattr(res, "language", None),
        )

    @modal.method()
    @node_slot(NodeSlots.AUDIO_DESCRIBE)
    def audio_describe(self, input: AudioDescribeInput) -> AudioDescribeOutput:
        try:
            with contextlib.ExitStack() as stack:
                src_path = self._as_wav(stack, input.audio)
                codes = self.dit_handler.convert_src_audio_to_codes(src_path)
            res = understand_music(self.llm_handler, codes)
            parts: list[str] = []
            caption = getattr(res, "caption", None)
            if caption:
                parts.append(str(caption))
            meta: list[str] = []
            for label, attr in (
                ("BPM", "bpm"),
                ("Key", "keyscale"),
                ("Language", "language"),
                ("Duration", "duration"),
            ):
                v = getattr(res, attr, None)
                if v not in (None, "", "unknown"):
                    meta.append(f"{label}: {v}")
            if meta:
                parts.append("; ".join(meta))
            lyrics = getattr(res, "lyrics", None)
            if lyrics and str(lyrics).strip() not in ("", "[Instrumental]"):
                parts.append(f"Lyrics: {lyrics}")
            text = "\n".join(parts).strip()
            if not text:
                raise RuntimeError("LM returned an empty analysis")
        except Exception as e:
            return AudioDescribeOutput(success=False, error=str(e))
        return AudioDescribeOutput(success=True, text=text)

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
        self._ensure_dit(DEFAULT_DIT)
        return self._run(
            lyrics=lyrics,
            caption=tags,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            language=language,
            seed=seed,
        )

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

