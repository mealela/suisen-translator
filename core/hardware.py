"""硬件探测与三档预设(低/中/高)。启动时异步探测并套用一次, 设置→杂项可重新检测或手动指定。"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

from config import (ALT_MT_TAG, DEFAULT_MT_TAG, DEFAULT_VLM_TAG,
                    LIGHT_MT_TAG, LIGHT_VLM_TAG)

log = logging.getLogger("suisen.hardware")

TIER_NAMES = {"high": "高配", "mid": "中配", "low": "低配"}

PRESETS = {
    "high": dict(
        locator_rapid_version="PP-OCRv6", locator_rapid_type="server",
        locator_det_limit_side_len=1280, sample_interval=0.25,
        precise_timing=True, locator_flash_probe=True,
        ocr_engine="vlm", vlm_detail_width=1024, conf_threshold=0.6,
        locator_vlm_verify=True, locator_verify_mode="all", locator_verify_budget=200,
        batch_size=8, ctx_translate=True, filter_llm=True,
        erase_engine="lama", erase_chunk=60, erase_dilate=8, erase_crf=16,
        _vlm=DEFAULT_VLM_TAG, _mt=DEFAULT_MT_TAG,
    ),
    "mid": dict(
        locator_rapid_version="PP-OCRv5", locator_rapid_type="server",
        locator_det_limit_side_len=1280, sample_interval=0.40,
        precise_timing=True, locator_flash_probe=True,
        ocr_engine="vlm", vlm_detail_width=896, conf_threshold=0.6,
        locator_vlm_verify=True, locator_verify_mode="ambiguous", locator_verify_budget=100,
        batch_size=6, ctx_translate=True, filter_llm=True,
        erase_engine="lama", erase_chunk=40, erase_dilate=8, erase_crf=18,
        _vlm=LIGHT_VLM_TAG, _mt=LIGHT_MT_TAG,
    ),
    "low": dict(
        locator_rapid_version="PP-OCRv5", locator_rapid_type="mobile",
        locator_det_limit_side_len=960, sample_interval=0.60,
        precise_timing=False, locator_flash_probe=True,
        ocr_engine="local", vlm_detail_width=640, conf_threshold=0.5,
        locator_vlm_verify=False, locator_verify_mode="off", locator_verify_budget=0,
        batch_size=4, ctx_translate=False, filter_llm=False,
        erase_engine="opencv", erase_chunk=24, erase_dilate=6, erase_crf=20,
        _vlm=LIGHT_VLM_TAG, _mt=LIGHT_MT_TAG,
    ),
}

_ALT_MT_TIERS = ("high",)


def _flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        exe = cand if os.path.isfile(cand) else ""
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=6, creationflags=_flags())
    except Exception as exc:
        log.debug("nvidia-smi 不可用: %s", exc)
        return None
    best = None
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mb = float(parts[1])
        except ValueError:
            continue
        if best is None or mb > best[1]:
            best = (parts[0], mb)
    if best is None:
        return None
    return {"gpu": best[0], "vram_gb": best[1] / 1024.0, "backend": "cuda"}


def _has_nvidia_driver():
    if os.name != "nt":
        return False
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.isfile(os.path.join(root, "System32", "nvcuda.dll"))


def cuda_capable():
    """是否值得安装 CUDA 版 torch: 有 NVIDIA 驱动(nvcuda.dll)或 nvidia-smi 能列出显卡。"""
    return _has_nvidia_driver() or bool(_nvidia_smi())


def _torch_gpu():
    if not _has_nvidia_driver() and os.name == "nt":
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": p.name, "vram_gb": p.total_memory / 2 ** 30, "backend": "cuda"}
    except Exception:
        pass
    return None


def _other_gpu():
    if os.name != "nt":
        return ""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController).Name -join ';'"],
                             capture_output=True, text=True, timeout=5, creationflags=_flags())
        names = [n.strip() for n in out.stdout.split(";") if n.strip()]
        return " / ".join(n for n in names if "nvidia" not in n.lower())[:60]
    except Exception:
        return ""


def probe():
    from core.model_manager import _total_ram_gb
    info = {"ram_gb": round(float(_total_ram_gb()), 1), "cpu": int(os.cpu_count() or 4),
            "gpu": "", "vram_gb": 0.0, "backend": "cpu"}
    gpu = _nvidia_smi() or _torch_gpu()
    if gpu:
        info.update(gpu)
        info["vram_gb"] = round(float(info["vram_gb"]), 1)
    else:
        info["gpu_other"] = _other_gpu()
    return info


def classify(info):
    vram, ram, cpu = info["vram_gb"], info["ram_gb"], info["cpu"]
    if vram >= 12 and ram >= 24:
        return "high"
    if vram >= 6 or (ram >= 16 and cpu >= 8):
        return "mid"
    return "low"


def describe(info):
    if info.get("gpu"):
        gpu = f"{info['gpu']} {info['vram_gb']:.0f}GB"
    elif info.get("gpu_other"):
        gpu = f"{info['gpu_other']}(非NVIDIA, 不计入分级)"
    else:
        gpu = "无独显/未识别"
    return f"GPU {gpu} · 内存 {info['ram_gb']:.0f}GB · CPU {info['cpu']}核"


def summary(info, tier):
    return f"[{TIER_NAMES.get(tier, tier)}] {describe(info)}"


def apply_preset(cfg, tier):
    from core.model_manager import is_managed
    preset = dict(PRESETS[tier])
    vlm_tag = preset.pop("_vlm")
    mt_tag = preset.pop("_mt")
    for key, value in preset.items():
        setattr(cfg, key, value)
    if is_managed(cfg.ep_vlm):
        cfg.ep_vlm = {**cfg.ep_vlm, "ollama_model": vlm_tag}
    if is_managed(cfg.ep_translate):
        cur = str(cfg.ep_translate.get("ollama_model", ""))
        if not (tier in _ALT_MT_TIERS and cur == ALT_MT_TAG):
            cfg.ep_translate = {**cfg.ep_translate, "ollama_model": mt_tag}
    return tier


def detect_and_apply(cfg, force_tier=None, info=None):
    """探测(或用给定的 info)并写入配置。GUI 在后台线程调 probe() 后再调本函数(传 info), 不阻塞界面。"""
    info = info or probe()
    tier = force_tier or (cfg.hw_tier if cfg.hw_tier in PRESETS else classify(info))
    apply_preset(cfg, tier)
    cfg.hw_applied = True
    cfg.hw_summary = summary(info, tier)
    cfg.save()
    log.info("硬件预设: %s", cfg.hw_summary)
    return info, tier
