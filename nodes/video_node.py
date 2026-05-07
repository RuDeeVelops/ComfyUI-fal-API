import asyncio
import math
import os
import random
import tempfile
import time
import cv2
import httpx
import numpy as np
import torch
import requests
import folder_paths
from fal_client import AsyncClient
from .fal_utils import ApiHandler, FalConfig, ImageUtils

fal_config = FalConfig()

# ============================================================================
# SHARED ENUMS — sourced from fal.ai OpenAPI schemas
# ============================================================================

SEEDANCE_DURATIONS = ["auto", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
SEEDANCE_ASPECT = ["auto", "16:9", "9:16", "1:1", "21:9", "4:3", "3:4"]
SEEDANCE_RESOLUTIONS = ["480p", "720p", "1080p"]
KLING_DURATIONS = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
# Kling I2V API does NOT accept aspect_ratio (output ratio = input image ratio).
# We expose it on the node and center-crop the input image client-side, matching how
# Flora and other UIs work around the API limitation.
KLING_ASPECT = ["auto", "16:9", "9:16", "1:1"]
ASPECT_RATIOS = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "21:9": 21 / 9,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
}

# ============================================================================
# CORE HELPERS
# ============================================================================

def _extract_error_body(resp):
    """Best-effort: pull the most informative slice out of a fal error response."""
    try:
        body = resp.json()
        # fal returns errors as {"detail": [...]} or {"detail": "..."} or similar
        if isinstance(body, dict) and "detail" in body:
            return body["detail"]
        return body
    except Exception:
        return resp.text[:1000] if hasattr(resp, "text") else "<no body>"


async def _fetch_result_with_fallbacks(handler, endpoint):
    """Wait for completion, then fetch the result with fault-tolerant URL fallback.

    fal returns the final result via response_url. For broken queue routing on
    very new endpoints we try a few URL shapes. Critically: 4xx codes that
    indicate AUTHORITATIVE failure (422 = validation error, 401/403 = auth) are
    surfaced immediately with the server's body — those aren't "try another URL"
    situations, they're real errors fal is telling us about.
    """
    # Poll status to completion (status URL always works server-side).
    async for _ in handler.iter_events(with_logs=False, interval=1.5):
        pass

    base = "https://queue.fal.run"
    rid = handler.request_id
    status_base = handler.status_url[:-len("/status")] if handler.status_url.endswith("/status") else handler.status_url
    candidates = [
        handler.response_url,
        f"{base}/{endpoint}/requests/{rid}",
        f"{base}/{endpoint}/requests/{rid}/response",
        f"{status_base}/response",
    ]

    errors = []
    for url in candidates:
        try:
            resp = await handler.client.get(url)
            sc = resp.status_code
            if sc == 200:
                return resp.json()
            # Authoritative failures from fal — don't waste time on other URLs.
            if sc in (401, 403, 422, 400):
                detail = _extract_error_body(resp)
                raise RuntimeError(
                    f"fal rejected the request (HTTP {sc}). This is a real "
                    f"validation/auth error from the server, not a routing bug.\n"
                    f"endpoint: {endpoint}\nrequest_id: {rid}\n"
                    f"server says: {detail}"
                )
            errors.append(f"  {url} -> HTTP {sc}: {(_extract_error_body(resp) or '')!s:.200}")
        except RuntimeError:
            raise
        except Exception as e:
            errors.append(f"  {url} -> {type(e).__name__}: {e}")

    raise RuntimeError(
        f"All result-fetch URLs failed for fal endpoint '{endpoint}' "
        f"(request_id={rid}).\nTried URLs:\n" + "\n".join(errors)
    )


# Patterns that indicate a DETERMINISTIC content_policy_violation — same input will
# always fail. Retrying these is pure waste (each retry = ~15 min on Seedance Reference
# because fal runs moderation AFTER generation, not before).
#
# Probabilistic borderline violations (no specific likeness/IP keyword) DO sometimes
# clear on retry per vicsee/segmind reports — we keep retry behavior for those.
_DETERMINISTIC_POLICY_PATTERNS = (
    "real person", "real people", "likeness", "likenesses",
    "private information", "image_urls",
    "trademark", "copyright", "intellectual property",
    "partner_validation_failed",
)


def _is_deterministic_policy_violation(error_msg: str) -> bool:
    msg = error_msg.lower()
    return "content_policy_violation" in msg and any(
        p in msg for p in _DETERMINISTIC_POLICY_PATTERNS
    )


async def _submit_with_retry(client, endpoint, args, max_retries=0, retry_label=""):
    """Submit + fetch with auto-retry on borderline content_policy_violation.

    Critical refinement: NOT all policy violations are worth retrying.

    DETERMINISTIC violations — likeness of real people, IP/copyright/trademark —
    will fail with the same input every time. fal runs moderation AFTER generation
    (~15 min on Seedance Reference), so retrying these wastes ~15 min per attempt.
    We detect known patterns and short-circuit.

    PROBABILISTIC violations — borderline cases without specific keywords — do
    sometimes clear on retry per vicsee/segmind reports. We keep the original
    backoff retry for these.

    - Retries ONLY on probabilistic content_policy_violation.
    - Backoff: 3s, 6s, 12s.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            handler = await client.submit(endpoint, arguments=args)
            result = await _fetch_result_with_fallbacks(handler, endpoint)
            if attempt > 0:
                print(f"[fal-API] {retry_label}: succeeded on retry {attempt}/{max_retries}")
            return result
        except RuntimeError as e:
            msg = str(e)
            if "content_policy_violation" not in msg.lower():
                raise
            # Deterministic violations: same input -> same failure. Don't waste
            # another 15 min generation cycle on a known-bad input.
            if _is_deterministic_policy_violation(msg):
                print(
                    f"[fal-API] {retry_label}: DETERMINISTIC content policy violation "
                    f"(likeness / IP / copyright) — NOT retrying. Same input always fails. "
                    f"Fix the input refs (Enterprise tier, fewer recognizable faces, "
                    f"AI-generated stylized images, blurred face regions) before re-running."
                )
                raise
            # Probabilistic borderline: retry with backoff.
            last_error = e
            if attempt >= max_retries:
                raise
            backoff = 3 * (2 ** attempt)
            print(
                f"[fal-API] {retry_label}: borderline content_policy_violation "
                f"(attempt {attempt + 1}/{max_retries + 1}) — retrying in {backoff}s"
            )
            await asyncio.sleep(backoff)
    raise last_error  # unreachable in normal flow


async def _run_parallel(endpoint, base_args, count, seed, retry_on_policy=0):
    """Fire `count` parallel requests via fal's queue API.

    seed = -1: generate a fresh random seed PER call PER run. Each variation
    differs from siblings AND from previous runs — exactly what users expect
    when they want to "queue a few attempts and see them."

    seed >= 0: deterministic. Variations get seed, seed+1, seed+2, ... so the
    same seed value reproduces the same set of variations.

    retry_on_policy: max retries per variation on content_policy_violation only.
    """
    client = AsyncClient(key=fal_config.get_key())

    async def one(i):
        args = dict(base_args)
        # Each retry must pick a fresh random seed when seed=-1, so we pass a sentinel
        # via args["seed"] = -1 and let _submit_with_retry build it per attempt below.
        # Simpler: roll the seed here for each variation, retries reuse same args
        # (same seed) — but with probabilistic moderation the SAME seed often clears
        # on retry, so reusing it is fine. Borderline=jitter, not seed-dependent.
        args["seed"] = (seed + i) if seed >= 0 else random.randint(0, 2**31 - 1)
        result = await _submit_with_retry(
            client, endpoint, args,
            max_retries=retry_on_policy,
            retry_label=f"variation {i + 1}",
        )
        return result["video"]["url"]

    return await asyncio.gather(*(one(i) for i in range(count)))


def _upload_image_batch(images, limit=9):
    """Upload a ComfyUI IMAGE batch and return URLs (up to `limit`)."""
    urls = []
    if images is None:
        return urls
    if images.ndim == 4:
        for i in range(min(images.shape[0], limit)):
            url = ImageUtils.upload_image(images[i].unsqueeze(0))
            if url:
                urls.append(url)
    else:
        url = ImageUtils.upload_image(images.unsqueeze(0))
        if url:
            urls.append(url)
    return urls


# Fal Seedance 2.0 reference-to-video constraints (from the live OpenAPI schema).
SEEDANCE_REF_VIDEO_MIN_SEC = 2.0
SEEDANCE_REF_VIDEO_MAX_SEC = 15.0
SEEDANCE_REF_AUDIO_MAX_SEC = 15.0
SEEDANCE_REF_VIDEO_MAX_BYTES = 50 * 1024 * 1024  # 50 MB combined cap; we enforce per-file
SEEDANCE_REF_VIDEO_MAX_LONG_DIM = 1280            # OpenAPI says ~480p-720p; 1080p+ commonly 422s


def _probe_video_meta(path):
    """Read duration + max dimension of a video file via OpenCV. Returns dict or None."""
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if fps > 0 and frames > 0:
            return {"duration": frames / fps, "width": w, "height": h, "long_dim": max(w, h)}
    except Exception:
        pass
    return None


# Backward-compatible alias used elsewhere in the module.
def _probe_video_duration_seconds(path):
    meta = _probe_video_meta(path)
    return meta["duration"] if meta else None


def _upload_video(video, min_seconds=None, max_seconds=None,
                  max_bytes=None, max_long_dim=None):
    """Upload a ComfyUI VIDEO with full preflight (duration + size + resolution).

    All checks fail FAST with a clear, actionable message — no fal credits wasted on
    requests we already know will be rejected by fal's queue validator.
    """
    if video is None:
        return None
    path = video.get_stream_source()

    # Duration + resolution check (single OpenCV probe).
    if any(x is not None for x in (min_seconds, max_seconds, max_long_dim)):
        meta = _probe_video_meta(path)
        if meta is None:
            print(f"[fal-API] WARNING: could not probe metadata of {path}; skipping preflight checks.")
        else:
            d = meta["duration"]
            if min_seconds is not None and d < min_seconds:
                raise ValueError(
                    f"Reference video is too short: {d:.2f}s "
                    f"(fal requires at least {min_seconds:.0f}s combined across @Video refs)."
                )
            if max_seconds is not None and d > max_seconds:
                raise ValueError(
                    f"Reference video is too long: {d:.2f}s "
                    f"(fal allows at most {max_seconds:.0f}s combined across @Video refs). "
                    f"Trim the clip or split it into chunks of {max_seconds:.0f}s or less."
                )
            if max_long_dim is not None and meta["long_dim"] > max_long_dim:
                raise ValueError(
                    f"Reference video resolution is too high: {meta['width']}x{meta['height']} "
                    f"(fal Seedance Reference expects roughly 480p-720p; longest side must be "
                    f"at most {max_long_dim}px). Downscale the source — e.g. "
                    f"`ffmpeg -i in.mp4 -vf scale=-2:720 -c:v libx264 -crf 23 out.mp4` — "
                    f"or feed a 720p export."
                )

    # File size check.
    if max_bytes is not None:
        try:
            sz = os.path.getsize(path)
            if sz > max_bytes:
                raise ValueError(
                    f"Reference video file is too large: {sz / 1024 / 1024:.1f} MB "
                    f"(fal Seedance Reference allows at most {max_bytes / 1024 / 1024:.0f} MB). "
                    f"Re-encode at lower bitrate or 720p — e.g. "
                    f"`ffmpeg -i in.mp4 -vf scale=-2:720 -c:v libx264 -crf 28 out.mp4`."
                )
        except OSError:
            pass

    try:
        return ImageUtils.upload_file(path)
    except Exception as e:
        print(f"[fal-API] failed to upload VIDEO: {e}")
        return None


def _upload_audio(audio, max_seconds=None):
    """Upload a ComfyUI AUDIO dict ({waveform, sample_rate}) as a WAV file and return its URL.

    Uses Python's stdlib `wave` module to write 16-bit PCM — zero new dependencies,
    works regardless of whether torchaudio's backend is sox / soundfile / torchcodec.
    (Recent torchaudio defaults to torchcodec which isn't always installed; we sidestep
    the whole question by writing the WAV ourselves.)

    If `max_seconds` is set, validates duration before upload and raises ValueError on overflow.
    """
    if audio is None:
        return None
    try:
        import wave
        import numpy as np

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        # ComfyUI AUDIO is shape (B, C, T); we want (C, T) for writing.
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if max_seconds is not None:
            duration = waveform.shape[-1] / sample_rate
            if duration > max_seconds:
                raise ValueError(
                    f"Reference audio is too long: {duration:.2f}s "
                    f"(fal Seedance Reference allows at most {max_seconds:.0f}s combined across @Audio refs)."
                )

        # Convert float32 [-1, 1] -> int16 PCM, then write interleaved bytes.
        arr = waveform.detach().cpu().numpy()
        if arr.dtype.kind == "f":
            arr = np.clip(arr, -1.0, 1.0)
            arr_int16 = (arr * 32767.0).astype(np.int16)
        elif arr.dtype == np.int16:
            arr_int16 = arr
        else:
            # int8/int32/etc — normalize through float
            max_val = float(np.iinfo(arr.dtype).max) if arr.dtype.kind == "i" else 1.0
            arr_int16 = (arr.astype(np.float32) / max_val * 32767.0).clip(-32768, 32767).astype(np.int16)

        channels = arr_int16.shape[0]
        # WAV expects interleaved samples: (T, C) in memory layout, then to bytes.
        interleaved = arr_int16.T.tobytes()

        tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes(interleaved)

        try:
            return ImageUtils.upload_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except ValueError:
        raise  # surface validation errors to ComfyUI
    except Exception as e:
        print(f"[fal-API] failed to upload AUDIO: {e}")
        return None


def _center_crop_to_aspect(image, target_ratio):
    """Center-crop a ComfyUI IMAGE tensor (B,H,W,C) to match `target_ratio` (w/h)."""
    if image is None or target_ratio is None:
        return image
    h, w = image.shape[1], image.shape[2]
    current = w / h
    if abs(current - target_ratio) < 1e-3:
        return image
    if current > target_ratio:
        new_w = int(round(h * target_ratio))
        offset = (w - new_w) // 2
        return image[:, :, offset:offset + new_w, :]
    new_h = int(round(w / target_ratio))
    offset = (h - new_h) // 2
    return image[:, offset:offset + new_h, :, :]


# ============================================================================
# NBC-APPROVED MODELS
# ============================================================================

class Seedance2TextToVideo_NBC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (SEEDANCE_ASPECT, {"default": "16:9"}),
                "duration": (SEEDANCE_DURATIONS, {"default": "5"}),
                "resolution": (SEEDANCE_RESOLUTIONS, {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "retry_on_policy_violation": ("INT", {"default": 2, "min": 0, "max": 5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_urls",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # When seed = -1 (random), force ComfyUI to re-execute on every queue.
        # Otherwise the cached result is reused and the user can't get a new variation
        # without manually editing the seed. seed >= 0 is treated as deterministic.
        seed = kwargs.get("seed", -1)
        return float("nan") if seed == -1 else seed

    async def generate(self, prompt, aspect_ratio, duration, resolution, generate_audio,
                       variations, seed, retry_on_policy_violation=2):
        args = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        return (await _run_parallel(
            "bytedance/seedance-2.0/text-to-video", args, variations, seed,
            retry_on_policy=retry_on_policy_violation,
        ),)


class Seedance2ImageToVideo_NBC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "aspect_ratio": (SEEDANCE_ASPECT, {"default": "auto"}),
                "duration": (SEEDANCE_DURATIONS, {"default": "5"}),
                "resolution": (SEEDANCE_RESOLUTIONS, {"default": "720p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "end_image": ("IMAGE",),
                "retry_on_policy_violation": ("INT", {"default": 2, "min": 0, "max": 5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_urls",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # When seed = -1 (random), force ComfyUI to re-execute on every queue.
        # Otherwise the cached result is reused and the user can't get a new variation
        # without manually editing the seed. seed >= 0 is treated as deterministic.
        seed = kwargs.get("seed", -1)
        return float("nan") if seed == -1 else seed

    async def generate(self, prompt, image, aspect_ratio, duration, resolution, generate_audio,
                       variations, seed, end_image=None, retry_on_policy_violation=2):
        args = {
            "prompt": prompt,
            "image_url": ImageUtils.upload_image(image),
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        if end_image is not None:
            end_url = ImageUtils.upload_image(end_image)
            if end_url:
                args["end_image_url"] = end_url
        return (await _run_parallel(
            "bytedance/seedance-2.0/image-to-video", args, variations, seed,
            retry_on_policy=retry_on_policy_violation,
        ),)


# NOTE: The legacy Seedance2ReferenceToVideo_NBC and Seedance2ReferenceToVideoEnterprise_NBC
# nodes have been REMOVED. They were single-image / single-video / single-audio nodes.
# Use Seedance2ReferenceCanonical_NBC instead — it has the same I/O shape (and more):
# multi-reference (up to 9 images, 3 videos, 3 audios), built-in model tier selector
# (Standard / Fast / Enterprise), single VIDEO output. See nodes/seedance_canonical.py.


class KlingV3Standard_NBC:
    """Kling V3 Standard. Auto-switches to I2V when `image` is connected, otherwise T2V."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": (KLING_DURATIONS, {"default": "5"}),
                "aspect_ratio": (KLING_ASPECT, {"default": "16:9"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "variations": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": "blur, distort, and low quality", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_urls",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # When seed = -1 (random), force ComfyUI to re-execute on every queue.
        # Otherwise the cached result is reused and the user can't get a new variation
        # without manually editing the seed. seed >= 0 is treated as deterministic.
        seed = kwargs.get("seed", -1)
        return float("nan") if seed == -1 else seed

    async def generate(self, prompt, duration, aspect_ratio, cfg_scale, variations, seed,
                       image=None, end_image=None, negative_prompt="blur, distort, and low quality"):
        if image is not None:
            # I2V: API has no aspect_ratio param — output ratio = input image ratio.
            # If user picked a specific ratio, center-crop the input image first (Flora-style).
            target = ASPECT_RATIOS.get(aspect_ratio)
            cropped = _center_crop_to_aspect(image, target) if target else image
            cropped_end = _center_crop_to_aspect(end_image, target) if (end_image is not None and target) else end_image

            endpoint = "fal-ai/kling-video/v3/standard/image-to-video"
            args = {
                "prompt": prompt,
                "start_image_url": ImageUtils.upload_image(cropped),
                "duration": duration,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
            }
            if cropped_end is not None:
                end_url = ImageUtils.upload_image(cropped_end)
                if end_url:
                    args["end_image_url"] = end_url
        else:
            # T2V branch — aspect_ratio is a real API field. "auto" isn't valid here, default to 16:9.
            endpoint = "fal-ai/kling-video/v3/standard/text-to-video"
            args = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio if aspect_ratio != "auto" else "16:9",
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
            }
        return (await _run_parallel(endpoint, args, variations, seed),)


# ============================================================================
# UTILITIES
# ============================================================================

class LoadVideoURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com/video.mp4"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "VHS_VIDEOINFO")
    RETURN_NAMES = ("frames", "frame_count", "video_info")
    FUNCTION = "load"
    CATEGORY = "video"

    def load(self, url, frame_load_cap):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            resp = requests.get(url, stream=True)
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
            path = tmp.name
        try:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (frame_load_cap > 0 and len(frames) >= frame_load_cap):
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame).float() / 255.0)
            cap.release()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return (torch.stack(frames), len(frames), {"fps": fps, "width": w, "height": h})


class UploadVideo_fal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "upload"
    CATEGORY = "video"

    def upload(self, video):
        return (ImageUtils.upload_file(video.get_stream_source()),)


# ============================================================================
# AUDIO EXTRACTION (companion to BlurFacesInVideo for full A/V Seedance pipelines)
# ============================================================================

class ExtractAudioFromVideo_NBC:
    """Extract the audio track from a VIDEO and output as ComfyUI's standard AUDIO type.

    Exact compatibility with native ComfyUI AUDIO consumers (Save Audio, the AUDIO
    socket on Seedance Reference, etc.). Format: {waveform: float32 tensor [B,C,T],
    sample_rate: int}.

    Built specifically for the Seedance restyle workflow per fal/Segmind docs:
    pass the same video into BOTH `video_1` (visual + baked-in timing) AND `audio_1`
    (explicit @Audio1 lip-sync directive). This node is the bridge — wire its output
    straight into the AUDIO socket without intermediate save/load steps.

    Internally uses PyAV with libswresample for clean format conversion (handles s16,
    s32, fltp, all standard codecs). Optional resampling and mono downmix.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "target_sample_rate": ("INT", {
                    "default": 0, "min": 0, "max": 192000, "step": 1,
                }),
                "force_mono": ("BOOLEAN", {"default": False}),
                "fail_silently": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "extract"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Output depends on source bytes; ComfyUI can't hash a VIDEO object reliably,
        # so re-extract on every queue. Cheap operation.
        return float("nan")

    def extract(self, video, target_sample_rate=0, force_mono=False, fail_silently=False):
        import av
        import numpy as np

        src = video.get_stream_source()
        # Handle BytesIO input (rare but possible per the VideoFromFile API).
        if not isinstance(src, str):
            tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            with open(tmp_in, "wb") as f:
                if hasattr(src, "read"):
                    src.seek(0)
                    f.write(src.read())
            src_path = tmp_in
        else:
            src_path = src

        try:
            container = av.open(src_path)
        except Exception as e:
            if fail_silently:
                return self._silent_audio()
            raise RuntimeError(f"Could not open video for audio extraction: {e}")

        try:
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                if fail_silently:
                    return self._silent_audio()
                raise ValueError(
                    "Source video has no audio track. Either pick a clip with sound, "
                    "or set fail_silently=True to get a 1-second mono silence stub."
                )

            in_rate = int(audio_stream.rate or 44100)
            out_rate = int(target_sample_rate) if target_sample_rate > 0 else in_rate
            in_channels = int(audio_stream.channels or 1)
            out_channels = 1 if force_mono else in_channels

            # Pick a layout PyAV understands. Channels >2 keeps source layout.
            if out_channels == 1:
                out_layout = "mono"
            elif out_channels == 2:
                out_layout = "stereo"
            else:
                out_layout = audio_stream.layout.name if audio_stream.layout else "stereo"

            # Resample to clean planar float, target rate, target layout. libswresample
            # handles all the fixed/float, planar/packed, channel-layout mess.
            resampler = av.AudioResampler(format="fltp", layout=out_layout, rate=out_rate)

            chunks = []
            for frame in container.decode(audio=0):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray())  # shape (C, T) for planar
            # Flush
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray())

            if not chunks:
                if fail_silently:
                    return self._silent_audio(rate=out_rate, channels=out_channels)
                raise ValueError("No audio frames could be decoded from source.")

            arr = np.concatenate(chunks, axis=-1)  # (C, T)
            if arr.ndim == 1:
                arr = arr[None, :]
            # Force float32 in [-1, 1]; PyAV's fltp output is already float32 normalized.
            arr = arr.astype(np.float32, copy=False)

            # ComfyUI AUDIO is shape (B, C, T) with B=1 typically.
            waveform = torch.from_numpy(arr).unsqueeze(0).contiguous()

            duration = arr.shape[-1] / out_rate
            print(f"[fal-API] ExtractAudioFromVideo: {arr.shape[0]}ch x {arr.shape[-1]} samples "
                  f"({duration:.2f}s @ {out_rate}Hz)")

            return ({"waveform": waveform, "sample_rate": out_rate},)
        finally:
            container.close()

    def _silent_audio(self, rate=44100, channels=1, seconds=1.0):
        """Return a tiny silent AUDIO stub. Used when fail_silently=True and source
        has no audio — keeps the workflow flowing without raising."""
        import numpy as np
        n_samples = max(1, int(rate * seconds))
        arr = np.zeros((max(1, channels), n_samples), dtype=np.float32)
        waveform = torch.from_numpy(arr).unsqueeze(0).contiguous()
        return ({"waveform": waveform, "sample_rate": rate},)


class PitchShiftAudio_NBC:
    """Pitch-shift an AUDIO without changing duration.

    Use case: defeat audio-fingerprint content filters while preserving lip-sync
    timing. Wire ExtractAudioFromVideo -> PitchShiftAudio (semitones=-2) -> the
    audio_1 socket on Seedance Reference. The model uses the pitched audio as a
    timing reference for lip-sync; you mux the ORIGINAL (un-shifted) audio over
    the silent restyled video at the end. The fingerprinter sees a different
    spectral signature than the original copyrighted track and lets it through.

    Implementation: PyAV filter graph (asetrate + atempo + aresample). The
    asetrate stage changes both pitch and duration; atempo with the inverse ratio
    restores duration while keeping the new pitch; aresample renormalizes back
    to the original sample rate. No external deps.

    Quality is good for lip-sync timing reference (which is what we need it for).
    The output audio is discarded after Seedance reads it — what you ship is the
    original un-shifted audio muxed in at the end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "semitones": ("FLOAT", {
                    "default": -2.0, "min": -12.0, "max": 12.0, "step": 0.5,
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "shift"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def shift(self, audio, semitones):
        # Zero-shift: passthrough untouched.
        if abs(semitones) < 0.001:
            return (audio,)

        import av
        from fractions import Fraction
        import numpy as np

        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])

        # ComfyUI AUDIO is (B, C, T); take the first batch.
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        channels = int(waveform.shape[0])
        layout = "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels}c")

        # ratio > 1 = pitch up, < 1 = pitch down
        ratio = 2.0 ** (semitones / 12.0)
        intermediate_sr = max(1, int(round(sr * ratio)))

        # Build the filter chain. atempo accepts 0.5-100; for ratios outside that range
        # we'd need to chain multiple atempos. Within ±12 semitones we're between
        # 0.5 and 2.0, so a single atempo is sufficient.
        graph = av.filter.Graph()
        src = graph.add_abuffer(
            sample_rate=sr, format="fltp", layout=layout, time_base=Fraction(1, sr),
        )
        f_set = graph.add("asetrate", str(intermediate_sr))
        f_atempo = graph.add("atempo", f"{1.0 / ratio:.6f}")
        f_resample = graph.add("aresample", str(sr))
        sink = graph.add("abuffersink")
        src.link_to(f_set)
        f_set.link_to(f_atempo)
        f_atempo.link_to(f_resample)
        f_resample.link_to(sink)
        graph.configure()

        # Push input as a single audio frame.
        arr_in = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
        # Ensure planar shape (C, T).
        if arr_in.ndim == 1:
            arr_in = arr_in[None, :]
        in_frame = av.AudioFrame.from_ndarray(arr_in, format="fltp", layout=layout)
        in_frame.sample_rate = sr
        in_frame.pts = 0
        graph.push(in_frame)
        graph.push(None)  # signal EOF

        # Pull output frames.
        out_chunks = []
        while True:
            try:
                out_frame = graph.pull()
            except (BlockingIOError, av.error.BlockingIOError):
                break
            except av.error.EOFError:
                break
            except av.AVError:
                break
            arr_out = out_frame.to_ndarray()  # (C, T) for fltp
            if arr_out.ndim == 1:
                arr_out = arr_out[None, :]
            out_chunks.append(arr_out)

        if not out_chunks:
            print("[fal-API] PitchShiftAudio: filter graph produced no output, returning input unchanged.")
            return (audio,)

        out_arr = np.concatenate(out_chunks, axis=-1)  # (C, T)
        out_tensor = torch.from_numpy(out_arr.astype(np.float32, copy=False)).unsqueeze(0).contiguous()

        in_dur = arr_in.shape[-1] / sr
        out_dur = out_arr.shape[-1] / sr
        print(
            f"[fal-API] PitchShiftAudio: {semitones:+.1f} semitones (ratio {ratio:.4f}) | "
            f"in {arr_in.shape[0]}ch x {arr_in.shape[-1]} samples ({in_dur:.2f}s) -> "
            f"out {out_arr.shape[0]}ch x {out_arr.shape[-1]} samples ({out_dur:.2f}s @ {sr}Hz)"
        )

        return ({"waveform": out_tensor, "sample_rate": sr},)


# ============================================================================
# FACE BLUR (preprocess to bypass face-detection content filters)
# ============================================================================

# YuNet face detector — production-grade CNN that handles frontal/profile/3-quarter/
# tilted/occluded faces. Built into cv2 (>=4.5.4). The ONNX model is ~340 KB, fetched
# once and cached in the package's models/ folder.
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def _yunet_model_path():
    """Cache the YuNet ONNX next to the package so the user doesn't have to manage it."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(pkg_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, YUNET_FILENAME)


def _ensure_yunet_model():
    path = _yunet_model_path()
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    print(f"[fal-API] Downloading YuNet face detector (one-time, ~340 KB) ...")
    resp = requests.get(YUNET_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(1 << 16):
            f.write(chunk)
    print(f"[fal-API] YuNet cached at {path}")
    return path


def _detect_faces_yunet(detector, frame_bgr):
    """Run YuNet on a BGR frame. Returns list of dicts:
        {"box": (x, y, w, h), "mouth": (x1, y1, x2, y2) or None}
    sorted left-to-right by box.x.

    The mouth region is derived from YuNet's 5-point landmarks (right + left mouth
    corners). It's intentionally generous (2x mouth-corner distance wide, 1x tall,
    biased slightly downward) so that jaw motion is included — that's the data
    Seedance reads to drive lip-sync.
    """
    h_img, w_img = frame_bgr.shape[:2]
    detector.setInputSize((w_img, h_img))
    ret_val, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return []
    out = []
    for f in faces:
        x = max(0, int(round(f[0])))
        y = max(0, int(round(f[1])))
        fw = max(1, int(round(f[2])))
        fh = max(1, int(round(f[3])))
        # YuNet landmarks (15-element row): box[0:4], r_eye[4:6], l_eye[6:8],
        # nose[8:10], r_mouth[10:12], l_mouth[12:14], score[14].
        rmx, rmy = float(f[10]), float(f[11])
        lmx, lmy = float(f[12]), float(f[13])
        mouth = None
        if rmx > 0 and lmx > 0 and rmy > 0 and lmy > 0:
            cx = (rmx + lmx) / 2.0
            cy = (rmy + lmy) / 2.0
            corner_dist = abs(rmx - lmx)
            mw = corner_dist * 2.0  # box ~ 2x wider than mouth corners
            mh = corner_dist * 1.0  # ~ as tall, captures jaw motion
            cy_adj = cy + corner_dist * 0.2  # bias down toward jaw
            mx1 = max(0, int(round(cx - mw / 2)))
            mx2 = min(w_img, int(round(cx + mw / 2)))
            my1 = max(0, int(round(cy_adj - mh / 2)))
            my2 = min(h_img, int(round(cy_adj + mh / 2)))
            if mx2 > mx1 and my2 > my1:
                mouth = (mx1, my1, mx2, my2)
        out.append({"box": (x, y, fw, fh), "mouth": mouth})
    out.sort(key=lambda d: d["box"][0])
    return out


class BlurFacesInVideo_NBC:
    """Detect faces in a VIDEO and blur all of them, or just specific ones by index.
    Uses YuNet (CNN, handles profile / 3-quarter / side / occluded faces).

    Use case: bypass face-detection-based content filters (e.g. Seedance Reference's
    likeness filter) by removing recognizable faces from the source video. The
    composition, motion, and timing are preserved — Seedance restyles the blurred
    regions into clay along with everything else.

    Workflow:
      1. Connect LoadVideo → `video`. Run with mode="all" to blur every face.
      2. Look at the `preview_frame` IMAGE output: faces are boxed + labeled #0,
         #1, #2 (left-to-right in the first frame).
      3. For surgical blur, switch mode="specific" and set blur_indices, e.g. "0"
         (or "0,2" for multiple).
      4. Wire `video` output straight into Seedance Reference's `ref_video`.

    Tracking: in "specific" mode, faces in subsequent frames are matched to the
    first-frame indices by left-to-right ordering. Works well for static-camera
    footage. For tricky shots, use mode="all".

    Tuning if a face is missed:
      - Lower `confidence` (default 0.6) toward 0.4-0.5 — more aggressive detection.
      - Increase `padding` if the box is tight and the blur leaves edges visible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "mode": (["all", "specific"], {"default": "all"}),
                "blur_indices": ("STRING", {"default": "0"}),
                "blur_strength": ("INT", {"default": 51, "min": 5, "max": 151, "step": 2}),
                "padding": ("INT", {"default": 20, "min": 0, "max": 200}),
                "confidence": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05}),
                "preserve_mouth": ("BOOLEAN", {"default": True}),
                "blur_method": (["mosaic", "gaussian", "noise"], {"default": "mosaic"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE")
    RETURN_NAMES = ("video", "preview_frame")
    FUNCTION = "blur"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def blur(self, video, mode, blur_indices, blur_strength, padding, confidence,
             preserve_mouth=True, blur_method="mosaic"):
        import av
        from comfy_api.latest._input_impl.video_types import VideoFromFile

        try:
            indices_to_blur = {int(x.strip()) for x in blur_indices.split(",") if x.strip()}
        except ValueError:
            indices_to_blur = {0}

        if blur_strength % 2 == 0:
            blur_strength += 1

        src = video.get_stream_source()
        if not isinstance(src, str):
            tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            with open(tmp_in, "wb") as f:
                if hasattr(src, "read"):
                    src.seek(0)
                    f.write(src.read())
            src_path = tmp_in
        else:
            src_path = src

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for face blur: {src_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize YuNet with the actual frame size.
        model_path = _ensure_yunet_model()
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(w, h),
            score_threshold=float(confidence),
            nms_threshold=0.3,
            top_k=5000,
        )

        out_dir = folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        out_path = os.path.join(out_dir, f"blurfaces_{ts}.mp4")

        from fractions import Fraction
        rate = Fraction(round(fps * 1000), 1000)

        container = av.open(out_path, mode="w")
        stream = container.add_stream("libx264", rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}

        # Audio passthrough: open the source separately to copy audio packets without
        # re-encoding (fast, lossless). Skip silently if the source has no audio.
        audio_in_container = None
        audio_in_stream = None
        audio_out_stream = None
        try:
            audio_in_container = av.open(src_path)
            audio_in_stream = next(
                (s for s in audio_in_container.streams if s.type == "audio"), None
            )
            if audio_in_stream is not None:
                # PyAV 14+ uses add_stream_from_template (older `template=` kwarg removed).
                audio_out_stream = container.add_stream_from_template(audio_in_stream)
        except Exception as e:
            print(f"[fal-API] BlurFacesInVideo: audio passthrough skipped ({e})")
            if audio_in_container is not None:
                audio_in_container.close()
                audio_in_container = None
            audio_in_stream = None
            audio_out_stream = None

        def _apply_obscuring(roi):
            """Three obscuring methods, picked via blur_method. The choice matters because
            Seedance reads its reference VISUALLY: a smooth gray blob can be interpreted
            as 'render this region blank in the output' (the bake-in failure). Mosaic
            and noise are CLEARLY stylized — the model reads them as existing visual
            treatments to restyle, not as voids."""
            if roi.size == 0:
                return roi
            if blur_method == "mosaic":
                # Pixelate: downsample then nearest-upsample. Block size scales with strength.
                rh, rw = roi.shape[:2]
                block = max(4, blur_strength // 3)
                small = cv2.resize(
                    roi, (max(1, rw // block), max(1, rh // block)),
                    interpolation=cv2.INTER_LINEAR,
                )
                return cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            if blur_method == "noise":
                # High-frequency noise mixed with the local color so it's NOT featureless;
                # Seedance can't interpret it as "leave this blank".
                noise = np.random.randint(0, 255, roi.shape, dtype=np.uint8)
                base = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
                return cv2.addWeighted(noise, 0.55, base, 0.45, 0)
            # gaussian (legacy)
            return cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)

        def _blur_face_keep_mouth(frame, face_box, mouth_box):
            """Save mouth pixels first, obscure the face region, restore the mouth.
            Kills identifying features (eyes/nose/cheeks/jawline) while keeping mouth
            motion data so Seedance reads lip-sync from the reference."""
            x, y, fw, fh = face_box
            mouth_pixels = None
            if preserve_mouth and mouth_box is not None:
                mx1, my1, mx2, my2 = mouth_box
                mouth_pixels = frame[my1:my2, mx1:mx2].copy()

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + fw + padding)
            y2 = min(frame.shape[0], y + fh + padding)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = _apply_obscuring(roi)

            if mouth_pixels is not None and mouth_pixels.size > 0:
                mx1, my1, mx2, my2 = mouth_box
                frame[my1:my2, mx1:mx2] = mouth_pixels

        preview_frame_bgr = None
        frame_idx = 0
        total_faces_blurred = 0
        audio_packets_copied = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = _detect_faces_yunet(detector, frame)

                if frame_idx == 0:
                    preview_frame_bgr = frame.copy()
                    for i, det in enumerate(faces):
                        x, y, fw, fh = det["box"]
                        will_blur = (mode == "all") or (i in indices_to_blur)
                        color = (0, 0, 255) if will_blur else (0, 200, 0)  # red=blur, green=keep
                        cv2.rectangle(preview_frame_bgr, (x, y), (x + fw, y + fh), color, 3)
                        # Mouth-preserve region (yellow), if detected and we're keeping it.
                        if preserve_mouth and det["mouth"] is not None and will_blur:
                            mx1, my1, mx2, my2 = det["mouth"]
                            cv2.rectangle(preview_frame_bgr, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
                        label = f"#{i}"
                        cv2.rectangle(preview_frame_bgr, (x, y - 30), (x + 55, y), color, -1)
                        cv2.putText(
                            preview_frame_bgr, label, (x + 6, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
                        )

                for i, det in enumerate(faces):
                    if mode == "all" or i in indices_to_blur:
                        _blur_face_keep_mouth(frame, det["box"], det["mouth"])
                        total_faces_blurred += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                frame_idx += 1

            for packet in stream.encode():
                container.mux(packet)

            # Audio passthrough: demux source audio packets and remux into output.
            # template= reuses the input codec/timebase, so packets pass through as-is.
            if audio_in_container is not None and audio_in_stream is not None and audio_out_stream is not None:
                try:
                    for packet in audio_in_container.demux(audio_in_stream):
                        if packet.dts is None:
                            continue
                        packet.stream = audio_out_stream
                        container.mux(packet)
                        audio_packets_copied += 1
                except Exception as e:
                    print(f"[fal-API] audio remux warning: {e}")
        finally:
            container.close()
            cap.release()
            if audio_in_container is not None:
                try:
                    audio_in_container.close()
                except Exception:
                    pass

        audio_msg = f", audio: {audio_packets_copied} packets copied" if audio_in_stream is not None else ", audio: none in source"
        print(f"[fal-API] BlurFacesInVideo: processed {frame_idx} frames, "
              f"blurred {total_faces_blurred} face regions{audio_msg}, output -> {out_path}")

        # Build the IMAGE preview tensor (B,H,W,C) float in [0,1].
        if preview_frame_bgr is None:
            preview_frame_bgr = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
        rgb_preview = cv2.cvtColor(preview_frame_bgr, cv2.COLOR_BGR2RGB)
        preview_tensor = torch.from_numpy(rgb_preview.astype(np.float32) / 255.0).unsqueeze(0)

        return (VideoFromFile(out_path), preview_tensor)


def _grid_layout(n):
    """Pick a sensible (cols, rows) grid for n cells: prefer slightly wider than tall."""
    if n <= 1:
        return (1, 1)
    if n == 2:
        return (2, 1)
    if n <= 4:
        return (2, 2)
    if n <= 6:
        return (3, 2)
    if n <= 9:
        return (3, 3)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return (cols, rows)


def _build_grid_video(paths, out_path, cell_w=480, cell_h=270, fps=24):
    """Tile multiple videos side-by-side into a single h264 mp4 for at-a-glance preview.
    Each cell labeled with its index (#00, #01, ...) so the user can map back to the
    individual file in the output folder. No audio in the grid (use individual files
    for that). Returns out_path on success or None on failure."""
    import av  # PyAV is bundled with modern ComfyUI for video work

    if not paths:
        return None

    cols, rows = _grid_layout(len(paths))
    grid_w, grid_h = cols * cell_w, rows * cell_h

    caps = []
    durations = []
    try:
        for p in paths:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                cap.release()
                continue
            f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            r = cap.get(cv2.CAP_PROP_FPS) or fps
            caps.append(cap)
            durations.append((f / r) if r > 0 else 0)

        if not caps or not any(d > 0 for d in durations):
            return None

        target_dur = min(d for d in durations if d > 0)
        target_frames = max(1, int(target_dur * fps))

        # Encode with h264 via PyAV — produces standard MP4 that plays inline in ComfyUI.
        from fractions import Fraction
        rate = Fraction(round(fps * 1000), 1000)
        container = av.open(out_path, mode="w")
        stream = container.add_stream("libx264", rate=rate)
        stream.width = grid_w
        stream.height = grid_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}

        for fi in range(target_frames):
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                # Cell label so user can match grid tiles to individual files (#00, #01, ...)
                cv2.rectangle(frame, (5, 5), (70, 30), (0, 0, 0), -1)
                cv2.putText(
                    frame, f"#{i:02d}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
                )
                row = i // cols
                col = i % cols
                y, x = row * cell_h, col * cell_w
                # cv2 reads BGR; convert to RGB for PyAV.
                canvas[y:y + cell_h, x:x + cell_w] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            av_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush the encoder.
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return out_path
    except Exception as e:
        print(f"[fal-API] grid build failed: {e}")
        return None
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


class PreviewVideosFromURLs:
    """Download fal video URLs to ComfyUI's output folder and preview the first inline.
    All variations are saved with sequential numbering (_00, _01, ...). Open the
    output folder to view individual files at full resolution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_urls": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "fal_video"}),
                "save_to_output": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "FAL/NBC_Approved"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def preview(self, video_urls, filename_prefix, save_to_output):
        def _scalar(x, default):
            return (x[0] if x else default) if isinstance(x, list) else x

        filename_prefix = _scalar(filename_prefix, "fal_video")
        save_to_output = _scalar(save_to_output, True)

        out_dir = folder_paths.get_output_directory() if save_to_output else folder_paths.get_temp_directory()
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        type_label = "output" if save_to_output else "temp"
        paths = []
        previews = []

        for i, url in enumerate(video_urls or []):
            if not url:
                continue
            filename = f"{filename_prefix}_{ts}_{i:02d}.mp4"
            full_path = os.path.join(out_dir, filename)
            try:
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(full_path, "wb") as f:
                    for chunk in resp.iter_content(1 << 16):
                        f.write(chunk)
            except Exception as e:
                print(f"[fal-API] download failed for {url}: {e}")
                continue
            paths.append(full_path)
            previews.append({"filename": filename, "subfolder": "", "type": type_label})

        return {
            "ui": {"images": previews[:1], "animated": (True,)},
            "result": (paths,),
        }


# ============================================================================
# REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Seedance2TextToVideo_NBC": Seedance2TextToVideo_NBC,
    "Seedance2ImageToVideo_NBC": Seedance2ImageToVideo_NBC,
    "KlingV3Standard_NBC": KlingV3Standard_NBC,
    "BlurFacesInVideo_NBC": BlurFacesInVideo_NBC,
    "ExtractAudioFromVideo_NBC": ExtractAudioFromVideo_NBC,
    "PitchShiftAudio_NBC": PitchShiftAudio_NBC,
    "PreviewVideosFromURLs": PreviewVideosFromURLs,
    "LoadVideoURL": LoadVideoURL,
    "UploadVideo_fal": UploadVideo_fal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance2TextToVideo_NBC": "Seedance 2.0 T2V Parallel (NBC)",
    "Seedance2ImageToVideo_NBC": "Seedance 2.0 I2V Parallel (NBC)",
    "KlingV3Standard_NBC": "Kling V3 Standard Parallel (NBC)",
    "BlurFacesInVideo_NBC": "Blur Faces in Video (NBC)",
    "ExtractAudioFromVideo_NBC": "Extract Audio from Video (NBC)",
    "PitchShiftAudio_NBC": "Pitch Shift Audio (NBC, defeats fingerprinting)",
    "PreviewVideosFromURLs": "Preview Videos from URLs (fal)",
    "LoadVideoURL": "Load Video from URL",
    "UploadVideo_fal": "Upload Video to Fal",
}
