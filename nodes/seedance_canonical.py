"""Seedance 2.0 Reference (Canonical, fal) — mirrors the architecture of
comfy_api_nodes.nodes_bytedance.ByteDance2ReferenceNode but routes through fal.ai.

Surface parity with the official node:
  - Multi-reference: up to 9 images, 3 videos, 3 audios
  - Model tier selector (Seedance 2.0 / Fast / Enterprise)
  - Single VIDEO output (not a STRING URL list — feeds straight into video pipelines)
  - Validation: counts, combined-duration 1.8s..15s, file-size and resolution preflight
  - Asset references: use @Image1..@Image9, @Video1..@Video3, @Audio1..@Audio3 in prompt;
    these map to whichever sockets you connected (in numeric order).

The only material UX delta vs. V3 IO.ComfyNode: optional sockets are all visible at once
(no Autogrow). Autogrow would require a sibling package using comfy_entrypoint, which
would isolate this node from the FalConfig / ImageUtils helpers we already share.
"""

import asyncio
import os
import time

import requests
import folder_paths
import torch  # noqa: F401  (kept for parity with rest of package)

from fal_client import AsyncClient
from .fal_utils import FalConfig, ImageUtils
from .video_node import (  # reuse helpers, fal_config, and validation constants
    fal_config,
    _fetch_result_with_fallbacks,
    _upload_video,
    _upload_audio,
    SEEDANCE_REF_VIDEO_MIN_SEC,
    SEEDANCE_REF_VIDEO_MAX_SEC,
    SEEDANCE_REF_VIDEO_MAX_BYTES,
    SEEDANCE_REF_VIDEO_MAX_LONG_DIM,
    SEEDANCE_REF_AUDIO_MAX_SEC,
)


_FAL_ENDPOINTS_BY_MODEL = {
    "Seedance 2.0": "bytedance/seedance-2.0/reference-to-video",
    "Seedance 2.0 Fast": "bytedance/seedance-2.0/fast/reference-to-video",
    "Seedance 2.0 Enterprise": "bytedance/seedance-2.0/enterprise/reference-to-video",
}

# Resolutions allowed per tier (matches official node's per-model resolution lists).
_RESOLUTIONS_BY_MODEL = {
    "Seedance 2.0": ["480p", "720p", "1080p"],
    "Seedance 2.0 Fast": ["480p", "720p"],
    "Seedance 2.0 Enterprise": ["480p", "720p", "1080p"],
}

_RATIOS = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "auto"]


class Seedance2ReferenceCanonical_NBC:
    """Mirrors ByteDance2ReferenceNode (canonical comfy_api_nodes architecture) on fal.

    Wire references into image_1..image_9, video_1..video_3, audio_1..audio_3 sockets
    (in order — gaps are skipped). Reference them in the prompt as @Image1, @Video1, etc.
    The result is a single VIDEO output ready to plug into preview / save nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Resolution options union across all tiers — runtime check enforces tier-specific
        # subset so the user never gets a surprise 422 from fal.
        all_resolutions = ["480p", "720p", "1080p"]
        d = {
            "required": {
                "prompt": ("STRING", {"default": "@Image1 performs the action from @Video1", "multiline": True}),
                "model": (list(_FAL_ENDPOINTS_BY_MODEL.keys()), {"default": "Seedance 2.0"}),
                "resolution": (all_resolutions, {"default": "720p"}),
                "ratio": (_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1, "display": "slider"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "watermark": ("BOOLEAN", {"default": False}),
                "auto_downscale": ("BOOLEAN", {"default": True}),
            },
        }
        # Numbered ref sockets, parity with reference_images / reference_videos / reference_audios.
        for i in range(1, 10):
            d["optional"][f"image_{i}"] = ("IMAGE",)
        for i in range(1, 4):
            d["optional"][f"video_{i}"] = ("VIDEO",)
        for i in range(1, 4):
            d["optional"][f"audio_{i}"] = ("AUDIO",)
        return d

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", -1)
        return float("nan") if seed == -1 else seed

    async def generate(self, prompt, model, resolution, ratio, duration, generate_audio, seed,
                       watermark=False, auto_downscale=True, **refs):
        # --- Tier validation ---
        if model not in _FAL_ENDPOINTS_BY_MODEL:
            raise ValueError(f"Unknown model tier: {model!r}")
        endpoint = _FAL_ENDPOINTS_BY_MODEL[model]
        allowed_res = _RESOLUTIONS_BY_MODEL[model]
        if resolution not in allowed_res:
            raise ValueError(
                f"Resolution {resolution!r} is not supported by {model!r}. "
                f"Allowed: {allowed_res}."
            )

        # --- Collect connected references in numeric order ---
        images = [refs[f"image_{i}"] for i in range(1, 10) if refs.get(f"image_{i}") is not None]
        videos = [refs[f"video_{i}"] for i in range(1, 4) if refs.get(f"video_{i}") is not None]
        audios = [refs[f"audio_{i}"] for i in range(1, 4) if refs.get(f"audio_{i}") is not None]

        if not images and not videos:
            raise ValueError(
                "At least one reference image or video must be connected "
                "(image_1..image_9 or video_1..video_3)."
            )
        if len(images) > 9:
            raise ValueError(f"Too many reference images ({len(images)}); max 9.")
        if len(videos) > 3:
            raise ValueError(f"Too many reference videos ({len(videos)}); max 3.")
        if len(audios) > 3:
            raise ValueError(f"Too many reference audios ({len(audios)}); max 3.")

        # --- Combined-duration validation (matches official node's 1.8s min, 15.1s max) ---
        total_video_duration = 0.0
        for i, v in enumerate(videos, 1):
            try:
                d_sec = float(v.get_duration())
            except Exception:
                d_sec = 0.0
            if d_sec and d_sec < 1.8:
                raise ValueError(f"Reference video_{i} too short: {d_sec:.2f}s (min 1.8s).")
            total_video_duration += d_sec
        if total_video_duration > 15.1:
            raise ValueError(
                f"Combined reference-video duration is {total_video_duration:.2f}s; "
                f"fal Seedance Reference allows at most 15s combined across @Video refs."
            )

        total_audio_duration = 0.0
        for i, a in enumerate(audios, 1):
            try:
                d_sec = a["waveform"].shape[-1] / int(a["sample_rate"])
            except Exception:
                d_sec = 0.0
            if d_sec and d_sec < 1.8:
                raise ValueError(f"Reference audio_{i} too short: {d_sec:.2f}s (min 1.8s).")
            total_audio_duration += d_sec
        if total_audio_duration > 15.1:
            raise ValueError(
                f"Combined reference-audio duration is {total_audio_duration:.2f}s; "
                f"fal allows at most 15s combined across @Audio refs."
            )

        # --- Upload all references to fal ---
        image_urls = []
        for img in images:
            url = ImageUtils.upload_image(img)
            if url:
                image_urls.append(url)

        video_urls = []
        for v in videos:
            url = _upload_video(
                v,
                min_seconds=SEEDANCE_REF_VIDEO_MIN_SEC,
                max_seconds=SEEDANCE_REF_VIDEO_MAX_SEC,
                max_bytes=SEEDANCE_REF_VIDEO_MAX_BYTES,
                max_long_dim=SEEDANCE_REF_VIDEO_MAX_LONG_DIM if not auto_downscale else None,
            )
            if url:
                video_urls.append(url)

        audio_urls = []
        for a in audios:
            url = _upload_audio(a, max_seconds=SEEDANCE_REF_AUDIO_MAX_SEC)
            if url:
                audio_urls.append(url)

        # --- Build args. duration is sent as STRING per fal's schema. ---
        args = {
            "prompt": prompt,
            "duration": str(duration),
            "aspect_ratio": ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        if image_urls:
            args["image_urls"] = image_urls
        if video_urls:
            args["video_urls"] = video_urls
        if audio_urls:
            args["audio_urls"] = audio_urls
        if seed >= 0:
            args["seed"] = seed
        # watermark is currently a no-op for fal endpoints (the parameter exists on the
        # original ComfyUI node for the comfy.org API). We keep it on the schema for
        # architecture parity; can be wired through if fal exposes it later.

        # --- Submit and fetch ---
        client = AsyncClient(key=fal_config.get_key())
        handler = await client.submit(endpoint, arguments=args)
        result = await _fetch_result_with_fallbacks(handler, endpoint)
        video_url = result["video"]["url"]

        # --- Download into ComfyUI temp dir, return as VideoFromFile (single VIDEO output) ---
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"seedance_canonical_{int(time.time() * 1000)}.mp4")
        resp = requests.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(1 << 16):
                f.write(chunk)

        return (VideoFromFile(out_path),)


NODE_CLASS_MAPPINGS = {
    "Seedance2ReferenceCanonical_NBC": Seedance2ReferenceCanonical_NBC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance2ReferenceCanonical_NBC": "Seedance 2.0 Reference Canonical (NBC, multi-ref, fal)",
}
