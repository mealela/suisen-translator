from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from core.v6_common import (WORK, FrameCursor, _iter_frames_overlapped, check_abort, fmt_mmss, publish,
                            run_folder, save_image, source_quad, video_meta)
from core.v6_paint import paint_frame

INTERMEDIATE_CRF = 12   # 工作区擦除中间视频: 导出时还要再编码一次, 这里用高质量保存


# 新增(模块级)
_ENC_CACHE = {}


def _probe_encoder(ffmpeg, name):
    """真实试编 4 帧: 编码器在 -encoders 里列出不等于能用(无 N 卡时 nvenc 会失败)。进程内缓存。"""
    if name in _ENC_CACHE:
        return _ENC_CACHE[name]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", "nullsrc=s=256x256:d=0.2:r=25", "-frames:v", "4",
                            "-c:v", name, "-f", "null", "-"],
                           capture_output=True, timeout=30, creationflags=flags)
        ok = r.returncode == 0
    except Exception:
        ok = False
    _ENC_CACHE[name] = ok
    return ok


def encoder_args(cfg, ffmpeg, crf):
    """成品编码参数: auto 时依次试 NVENC/QSV/AMF, 都不可用退 libx264。返回 (参数列表, 编码器名)。"""
    want = str(getattr(cfg, "video_encoder", "auto") or "auto")
    cands = ["h264_nvenc", "h264_qsv", "h264_amf"] if want == "auto" else ([] if want == "libx264" else [want])
    for name in cands:
        if not _probe_encoder(ffmpeg, name):
            continue
        if name == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                    "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"], name
        if name == "h264_qsv":
            return ["-c:v", "h264_qsv", "-global_quality", str(crf), "-preset", "medium", "-pix_fmt", "nv12"], name
        if name == "h264_amf":
            return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp",
                    "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p"], name
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"], "libx264"


def _erase_quad(ev, t):
    if getattr(ev, "mask_custom", False) and ev.mask and len(ev.mask) == 4:
        x1, y1, x2, y2 = ev.mask
        return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    q = source_quad(ev, t)
    if q is None:
        ev.ensure_mask()
        x1, y1, x2, y2 = ev.mask
        q = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return q


def _erase_target(ev):
    return bool(ev.text.strip() or ev.translated.strip())


def _expand_quad(q, ratio, w, h):
    """擦除范围在定位框基础上绕框心外扩 ratio(默认 5%), 防止原字边缘被漏擦。"""
    if ratio <= 0:
        return q
    q = np.asarray(q, np.float32).copy()
    c = q.mean(axis=0)
    q = c + (q - c) * (1.0 + float(ratio))
    q[:, 0] = np.clip(q[:, 0], 0, w)
    q[:, 1] = np.clip(q[:, 1], 0, h)
    return q


def frame_mask(events, masks, t, w, h, cfg):
    """t 时刻需要擦除的区域(0/255)。只包含带文字的字幕框; 右键遮挡块(masks)不参与擦除。
    当前时刻没有字幕时直接返回全零, 不做形态学。"""
    output = np.zeros((h, w), np.uint8)
    pad = max(0.0, float(cfg.erase_time_pad))
    expand = max(0.0, float(getattr(cfg, "erase_expand_ratio", 0.05)))
    hit = False
    for ev in events:
        if not (ev.start - pad <= t < ev.end + pad) or not _erase_target(ev):
            continue
        q = _erase_quad(ev, t)
        q = _expand_quad(q, expand, w, h)
        cv2.fillPoly(output, [np.rint(q).astype(np.int32)], 255)
        hit = True
    if not hit:
        return output
    dilation = max(0, int(cfg.erase_dilate))
    if dilation:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        output = cv2.dilate(output, kernel)
    return output


def erase_targets_exist(events, cfg):
    """是否有需要擦除的字幕(带文字)。签名与 erase_items 保持 (events, cfg) 形式。"""
    return any(_erase_target(ev) for ev in events)


def _rounded(value, digits=1):
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, np.ndarray):
        return _rounded(value.tolist(), digits)
    return value


def erase_params(cfg):
    return {"engine": str(cfg.erase_engine), "dilate": int(cfg.erase_dilate),
            "expand": round(float(getattr(cfg, "erase_expand_ratio", 0.05)), 3),
            "pad": round(float(cfg.erase_time_pad), 3)}


def erase_items(events, cfg):
    """影响擦除结果的全部字幕要素: [start, end, geom, track, custom, mask, bbox]"""
    items = []
    for ev in events:
        if not _erase_target(ev):
            continue
        custom = bool(getattr(ev, "mask_custom", False))
        geom = getattr(ev, "geom", None)
        items.append([
            round(float(ev.start), 3), round(float(ev.end), 3),
            _rounded(geom), _rounded(getattr(ev, "geom_track", None)),
            custom, _rounded(list(ev.mask)) if custom and ev.mask else None,
            _rounded(list(ev.bbox)) if not geom and ev.bbox else None,
        ])
    items.sort(key=lambda x: json.dumps(x, default=str))
    return items


def file_identity(video):
    try:
        st = os.stat(str(video))
        return [int(st.st_size), int(st.st_mtime)]
    except (OSError, TypeError):
        return None


def erase_signature(events, cfg, video=None, t0=0.0, t1=None):
    """擦除结果签名: 字幕时间/几何/自定义遮罩 + 擦除参数 + 视频文件身份。签名变 → 缓存不能直接复用。"""
    payload = {"v": 2, "events": erase_items(events, cfg), **erase_params(cfg),
               "range": [round(float(t0 or 0.0), 3), None if t1 is None else round(float(t1), 3)],
               "video": file_identity(video) if video else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _union(intervals):
    out = []
    for a, b in sorted([list(x) for x in intervals]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _sym_diff(a, b):
    """两组区间并集的对称差: 只属于其中一方的时间。"""
    a, b = _union(a), _union(b)
    points = sorted({p for seg in a + b for p in seg})
    out = []
    for lo, hi in zip(points[:-1], points[1:]):
        mid = (lo + hi) / 2
        if any(x <= mid < y for x, y in a) != any(x <= mid < y for x, y in b):
            out.append([lo, hi])
    return _union(out)


def affected_ranges(old_items, new_items, cfg, fps=25.0):
    """新旧擦除项的差异 → 必须重新擦除的时间段。同一几何只重擦时间上有出入的部分。"""
    pad = max(0.0, float(cfg.erase_time_pad)) + 2.0 / max(1.0, float(fps))
    groups = {}
    for side, items in (("old", old_items), ("new", new_items)):
        for item in items:
            key = json.dumps(item[2:], default=str)
            groups.setdefault(key, {"old": [], "new": []})[side].append([float(item[0]), float(item[1])])
    out = []
    for g in groups.values():
        for a, b in _sym_diff(g["old"], g["new"]):
            out.append([a - pad, b + pad])
    return [(max(0.0, a), b) for a, b in _union(out)]


def _fmt_eta(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _color_args(meta):
    c = meta.get("color") or {}
    args = []
    for flag, key in (("-colorspace", "space"), ("-color_primaries", "primaries"),
                      ("-color_trc", "trc"), ("-color_range", "range")):
        if c.get(key):
            args += [flag, c[key]]
    return args


def render_video(video, events, masks, out_path, cfg, progress, abort,
                 t0=0.0, t1=None, erase_only=False, force_erase=False,
                 frames_source=None, erase=None, intermediate=False,
                 patch_source=None, patch_ranges=None):
    """
    video          原视频(音轨与时间轴来源)
    frames_source  工作区里已擦除的中间视频: 给出时画面从这里读, 不再擦除
    erase          None=按 cfg(placement/pipe_erase) 决定; True/False=显式指定
    erase_only     只输出擦除后的画面(不嵌字、不叠遮罩) → 工作区中间产物
    intermediate   工作区中间视频: 高质量(crf 12 / 4:4:4)、无音轨、快速预设
    patch_source   增量擦除: 旧的擦除缓存; patch_ranges 之外的时间直接沿用旧结果, 只重擦范围内的帧
    """
    from core.inpaint import _masked_change, ensure_ffmpeg, get_eraser
    from core.model_manager import REGISTRY, ROOT

    meta = video_meta(video)
    src_meta = video_meta(frames_source) if frames_source else meta
    fps, w, h = src_meta["fps"], src_meta["w"], src_meta["h"]
    duration = src_meta["duration"] or meta["duration"]
    t0 = max(0.0, float(t0 or 0))
    t1 = min(float(t1), duration) if t1 is not None else duration
    if t1 <= t0:
        raise RuntimeError("视频时长未知或导出范围无效")
    for source in (video, frames_source, patch_source):
        if source and Path(source).resolve() == Path(out_path).resolve():
            raise RuntimeError("输出文件不能覆盖输入视频")
    ffmpeg = ensure_ffmpeg(cfg, progress, abort)

    if erase is None:
        erase = force_erase or erase_only or (
            cfg.placement == "erase" and bool(getattr(cfg, "pipe_erase", True)))
    do_erase = bool(erase) and frames_source is None
    patch = None
    if do_erase and patch_source and patch_ranges is not None:
        try:
            pm = video_meta(patch_source)
            if (pm["w"], pm["h"]) == (w, h):
                patch = (patch_source, [(float(a), float(b)) for a, b in patch_ranges])
        except Exception:
            patch = None
    eraser, mode = None, "frame"
    if do_erase:
        if cfg.erase_engine == "lama":
            from core.model_manager import is_installed
            item = next(x for x in REGISTRY if x["id"] == "lama")
            if not is_installed(item):
                raise RuntimeError(f"擦除模型 [{item['name']}] 尚未安装: 请到[模型库]安装(无痕擦除需要 LAMA)")
            import importlib.util
            if importlib.util.find_spec("simple_lama_inpainting"):
                os.environ["LAMA_MODEL"] = str(ROOT / "inpaint/lama/big-lama.pt")
        eraser, mode = get_eraser(cfg, fps)

    job = run_folder("render-", cfg)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = None
    if do_erase:
        # 任务预览帧: 擦除过程中把"已处理帧"的缩略 JPEG 写到 .work/task-preview/<视频键>/,
        # 供 GUI 在关闭[跟随任务画面]时按进度展示(已擦除区=擦除后+叠加层嵌字, 未到进度=原画面)。
        # 整段重擦(非增量)时先清掉上次残留; 增量擦除保留已有预览, 只覆盖重擦范围。
        from core.workspace import video_key
        preview_dir = WORK / "task-preview" / video_key(video)
        if patch is None and preview_dir.is_dir():
            shutil.rmtree(preview_dir, ignore_errors=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial-" + uuid.uuid4().hex[:8] + output.suffix)
    keep_audio = bool(cfg.keep_audio) and not intermediate
    has_audio = bool(keep_audio and meta["audio"])
    full_range = t0 <= 1e-6 and t1 >= duration - 1e-3
    audio_copy = has_audio and full_range and str(meta.get("audio_codec", "")) == "aac"
    video_filters = []
    if intermediate:                        # 工作区中间视频: 还要再编一次, 保持 4:4:4 近无损; libx264 对 444 支持最稳
        enc_args, enc_name = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(INTERMEDIATE_CRF),
                              "-pix_fmt", "yuv444p"], "libx264"
    else:
        enc_args, enc_name = encoder_args(cfg, ffmpeg, int(cfg.erase_crf))
        if w % 2 or h % 2:
            video_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.9f}", "-i", "pipe:0"]
    if has_audio:
        command += ["-ss", f"{t0:.6f}", "-t", f"{t1 - t0:.6f}", "-i", str(Path(video).resolve())]
    command += ["-map", "0:v:0"]
    if has_audio:
        command += ["-map", "1:a:0"] + (["-c:a", "copy"] if audio_copy else ["-c:a", "aac", "-af", "apad"])
    if video_filters:
        command += ["-vf", ",".join(video_filters)]
    command += enc_args
    command += _color_args(meta)
    command += ["-t", f"{t1 - t0:.6f}", str(temporary.resolve())]

    stop_watch = threading.Event()
    writing = [False, 0.0]
    stats = {
        "engine": cfg.erase_engine if do_erase else ("pre-erased" if frames_source else "none"),
        "erase_executed": do_erase, "patched": bool(patch),
        "dirty_frames": 0, "changed_frames": 0, "reused_frames": 0, "output_frames": 0,
        "static_reused_frames": 0,
        "notes": [],
    }
    stats["encoder_video"] = enc_name
    if meta.get("hdr") or meta.get("high_depth"):
        stats["notes"].append("源视频为 10bit/HDR, 成品会转为 8bit(已保留色彩标签)")
    if keep_audio and not meta["audio"]:
        stats["notes"].append("源视频本身没有音轨, 输出将无声")
    debug = bool(getattr(cfg, "debug_mode", False))
    label = "擦除" if erase_only else ("擦除→嵌字" if do_erase else ("嵌字(复用已擦除画面)" if frames_source else "嵌字"))
    process = None
    success = False
    started = time.monotonic()
    zero_mask = np.zeros((h, w), np.uint8)

    from concurrent.futures import ThreadPoolExecutor
    jpg_pool = ThreadPoolExecutor(max_workers=1) if preview_dir is not None else None
    jpg_pending = []

    def _write_preview(img, path):
        if img.shape[1] > 1280:
            img = cv2.resize(img, (1280, int(round(img.shape[0] * 1280.0 / img.shape[1]))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    with (job / "ffmpeg.log").open("wb") as logfile:
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=logfile, stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))

            def watchdog():
                while not stop_watch.wait(0.2):
                    if writing[0] and time.monotonic() - writing[1] > 90 and process.poll() is None:
                        process.terminate()
                        return

            threading.Thread(target=watchdog, daemon=True).start()

            def output_frame(t, original, repaired, mask, reused=False):
                check_abort(abort)
                if reused:
                    stats["reused_frames"] += 1
                elif np.any(mask):
                    stats["dirty_frames"] += 1
                    delta = cv2.absdiff(original, repaired)[mask > 0]
                    change = float(delta.mean()) if delta.size else 0.0
                    if change >= float(cfg.erase_min_change):
                        stats["changed_frames"] += 1
                    if stats["dirty_frames"] == 1 and debug:
                        save_image(job / "before.png", original)
                        save_image(job / "after.png", repaired)
                        save_image(job / "mask.png", mask)
                final = repaired if erase_only else paint_frame(repaired, events, masks, t, cfg)
                publish(video, "erase" if do_erase else "render", t, final, baked=True)
                writing[:] = [True, time.monotonic()]
                try:
                    process.stdin.write(np.ascontiguousarray(final).tobytes())
                finally:
                    writing[0] = False
                stats["output_frames"] += 1
                # 任务预览帧: 只写有实际画面的帧(有掩码或沿用旧缓存); 60fps 等高帧率隔帧写,
                # 宽度压到 1280、JPEG 80——每帧额外成本约 5~10ms, 对擦除总时长影响可忽略。
                if jpg_pool is not None and (reused or np.any(mask)) \
                        and (fps <= 40 or stats["output_frames"] % 2 == 0):
                    jpg_pending[:] = [f for f in jpg_pending if not f.done()]
                    if len(jpg_pending) < 8:                 # 磁盘慢时宁可丢预览帧也不拖慢擦除
                        jpg_pending.append(jpg_pool.submit(
                            _write_preview, final, str(preview_dir / ("%08d.jpg" % int(round(t * 1000))))))
                if stats["output_frames"] % 10 == 0:
                    done = t - t0
                    elapsed = time.monotonic() - started
                    eta = ""
                    if elapsed > 3 and done > 0:
                        eta = f" · 剩余约 {_fmt_eta((t1 - t) * elapsed / done)}"
                    progress(min(99, int(done * 100 / (t1 - t0))),
                             f"{label} {fmt_mmss(done)}/{fmt_mmss(t1 - t0)}{eta}")

            buffer, history = [], []
            body_size = max(16, int(cfg.erase_chunk))
            overlap = 4 if mode == "temporal" else 0
            batch_n = body_size if mode == "temporal" else min(body_size, 16)

            # ---- 静态字幕帧复用: 掩码相同且掩码区域与上一帧几乎一致时, 直接沿用上一帧的修复结果 ----
            # MV 等静止字幕场景下大部分帧不用再过 LAMA; 阈值保守(掩码区平均色差 < 2.0/255),
            # 背景一动就自动回到引擎修复。temporal(时序模型)模式不走复用, 保持序列完整。
            STATIC_REUSE_DIFF = 2.0
            prev_static = [None]            # (上一帧原图, 掩码, 修复结果)

            def try_reuse(frame, mask):
                p = prev_static[0]
                if p is None:
                    return None
                pf, pm, pr = p
                if pm.shape != mask.shape or not np.array_equal(pm, mask):
                    return None
                if _masked_change(frame, pf, mask) >= STATIC_REUSE_DIFF:
                    return None
                out = frame.copy()
                sel = mask > 0
                out[sel] = pr[sel]
                return out

            def note_repair(frame, mask, repaired):
                prev_static[0] = (frame, mask, repaired)

            def run_chunk(work):
                originals = [x[1] for x in work]
                mask_list = [x[2] for x in work]
                if not any(np.any(m) for m in mask_list):
                    return originals
                if mode != "temporal":
                    results = [None] * len(work)
                    todo_i, todo_f, todo_m = [], [], []
                    reuse_of = {}                 # 组内复用: results[i] 沿用 todo 第 k 帧的修复
                    for i, (frame, mask) in enumerate(zip(originals, mask_list)):
                        if not np.any(mask):
                            todo_i.append(i)
                            todo_f.append(frame)
                            todo_m.append(mask)
                            continue
                        r = try_reuse(frame, mask)              # 与上一个已修复帧(跨块状态)比较
                        if r is not None:
                            results[i] = r
                            stats["static_reused_frames"] += 1
                            note_repair(frame, mask, r)
                            continue
                        if todo_f and np.array_equal(todo_m[-1], mask) \
                                and _masked_change(frame, todo_f[-1], mask) < STATIC_REUSE_DIFF:
                            reuse_of[i] = len(todo_f) - 1       # 与本块内最近一个待修复帧同属一个静止组
                            stats["static_reused_frames"] += 1
                            continue
                        todo_i.append(i)
                        todo_f.append(frame)
                        todo_m.append(mask)
                    if todo_f:
                        inputs, input_masks = todo_f, todo_m
                        if len(inputs) == 1:
                            inputs, input_masks = inputs + [inputs[-1]], input_masks + [input_masks[-1]]
                        outs = eraser.erase_chunk(inputs, input_masks, abort)
                        for k, i in enumerate(todo_i):
                            results[i] = outs[k]
                            note_repair(todo_f[k], todo_m[k], outs[k])
                        for i, k in reuse_of.items():
                            g = originals[i].copy()
                            sel = mask_list[i] > 0
                            g[sel] = outs[k][sel]
                            results[i] = g
                            note_repair(originals[i], mask_list[i], g)
                    if any(r is None for r in results):
                        raise RuntimeError("擦除引擎输出帧数不匹配")
                    return results
                inputs, input_masks = originals, mask_list
                if len(inputs) == 1:
                    inputs, input_masks = inputs + [inputs[-1]], input_masks + [input_masks[-1]]
                results = eraser.erase_chunk(inputs, input_masks, abort)[:len(work)]
                if len(results) != len(work):
                    raise RuntimeError("擦除引擎输出帧数不匹配")
                return results

            def flush(final=False):
                nonlocal buffer, history
                while buffer and (final or len(buffer) >= batch_n + overlap):
                    n = min(batch_n, len(buffer))
                    work = history + buffer[:n + overlap]
                    results = run_chunk(work)
                    offset = len(history)
                    for i in range(n):
                        t, original, mask, reused = buffer[i]
                        output_frame(t, original, results[offset + i], mask, reused)
                    history = (history + buffer[:n])[-overlap:] if overlap else []
                    buffer = buffer[n:]

            cache = FrameCursor(patch[0], t0, t1, abort) if patch else None
            ranges = patch[1] if patch else []

            def in_ranges(t):
                return any(a <= t < b for a, b in ranges)

            next_output = t0
            # VFR 按平均帧率重采样为 CFR; 音频按同一时间窗裁剪
            prefetch = max(2, min(12, int(300_000_000 // max(1, w * h * 3))))
            for start, end, frame in _iter_frames_overlapped(frames_source or video, t0, t1, abort,
                                                             prefetch=prefetch):
                while next_output < min(end, t1) - 1e-8:
                    check_abort(abort)
                    t = next_output
                    src, reused = frame, False
                    if cache is not None and not in_ranges(t):
                        cached = cache.frame_at(t)
                        if cached is not None and cached.shape == frame.shape:
                            src, reused = cached, True
                    mask = frame_mask(events, masks, t, w, h, cfg) if (do_erase and not reused) else zero_mask
                    if mode == "frame" or reused and not buffer:
                        if do_erase and not reused and np.any(mask):
                            repaired = try_reuse(src, mask)
                            if repaired is not None:
                                stats["static_reused_frames"] += 1
                            else:
                                repaired = eraser.erase(src, mask)
                                note_repair(src, mask, repaired)
                        else:
                            repaired = src
                        output_frame(t, src, repaired, mask, reused)
                    elif mode == "batch" and not buffer and not np.any(mask):
                        output_frame(t, src, src, mask, reused)
                    else:
                        buffer.append((t, src.copy(), mask, reused))
                        flush()
                    next_output = t0 + (round((next_output - t0) * fps) + 1) / fps
            flush(final=True)
            if cache is not None:
                cache.close()

            process.stdin.close()
            try:
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise RuntimeError("FFmpeg 编码收尾超时") from exc
            if code:
                raise RuntimeError(f"FFmpeg 失败, 请查看 {job / 'ffmpeg.log'}")
            if not stats["output_frames"]:
                raise RuntimeError("没有输出视频帧")
            if do_erase and not stats["dirty_frames"] and not stats["reused_frames"]:
                stats["notes"].append("所选时间范围内没有需要擦除的字幕, 画面未改动")
            elif do_erase and stats["dirty_frames"] and not stats["changed_frames"]:
                stats["notes"].append("擦除区域几乎没有像素变化, 请检查效果(引擎可能未生效)")
            check_abort(abort)
            os.replace(temporary, output)
            success = True
            tail = ("; 注意: " + "; ".join(stats["notes"])) if stats["notes"] else ""
            if do_erase:
                progress(100, f"完成, 擦除生效 {stats['changed_frames']}/{stats['dirty_frames']} 帧"
                              + (f", 沿用旧结果 {stats['reused_frames']} 帧" if stats["reused_frames"] else "")
                              + (f", 静态复用 {stats['static_reused_frames']} 帧" if stats["static_reused_frames"] else "")
                              + tail)
            else:
                progress(100, "完成" + tail)
        finally:
            if jpg_pool is not None:
                jpg_pool.shutdown(wait=False)
            stop_watch.set()
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stats["success"] = success
            stats["opencv_fallback_frames"] = int(getattr(eraser, "fallback_frames", 0) or 0)
            if success and preview_dir is not None:
                shutil.rmtree(preview_dir, ignore_errors=True)   # 任务预览只供任务进行中查看, 完成即清
            if debug:
                (job / "report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            if not success:
                temporary.unlink(missing_ok=True)
            if eraser is not None and hasattr(eraser, "close"):
                eraser.close()
    if success and not debug:
        shutil.rmtree(job, ignore_errors=True)
    return str(output)


def erase_to_workspace(video, events, cfg, progress, abort, previous=None, redo=None):
    """[04 原字处理](无痕擦除, 工作区内)。previous: 上次记录 → 只重擦有改动的时间段;
    redo: 强制重擦的时间段列表(选中字幕 / 整段重擦)——其余帧沿用现有缓存。返回 (中间视频路径, 签名, 记录dict)。"""
    from core.workspace import erased_path, purge_erased_for
    sig = erase_signature(events, cfg, video)
    items = erase_items(events, cfg)
    info = {"sig": sig, "items": items, "params": erase_params(cfg), "file": file_identity(video)}
    target = erased_path(video, sig)
    redo = _union([[float(a), float(b)] for a, b in (redo or [])])
    if target.is_file() and not redo:
        progress(100, "画面已擦除过且与当前字幕一致, 直接复用")
        info["path"] = str(target)
        purge_erased_for(video, keep=target)
        return str(target), sig, info
    try:
        meta = video_meta(video)
        fps, duration = meta["fps"], meta["duration"]
    except Exception:
        fps, duration = 25.0, 0.0
    patch_source, patch_ranges = None, None
    if target.is_file() and redo:
        patch_source, patch_ranges = str(target), [(a, b) for a, b in redo]     # 缓存一致: 只重擦 redo
    else:
        prev = dict(previous or {})
        if (prev.get("items") is not None and prev.get("params") == info["params"]
                and prev.get("file") == info["file"] and os.path.isfile(str(prev.get("path", "")))):
            ranges = _union(affected_ranges(prev["items"], items, cfg, fps)
                            + [list(x) for x in redo])
            total = sum(b - a for a, b in ranges)
            if ranges and (not duration or total < 0.9 * duration):
                patch_source, patch_ranges = prev["path"], [(a, b) for a, b in ranges]
    if patch_ranges:
        progress(0, f"增量擦除: 只重擦 {fmt_mmss(sum(b - a for a, b in patch_ranges))}, 其余画面沿用上次结果")
    target.parent.mkdir(parents=True, exist_ok=True)
    out = target.with_name(target.stem + "-redo.mp4") if patch_source == str(target) else target
    render_video(video, events, [], str(out), cfg, progress, abort,
                 erase_only=True, erase=True, intermediate=True,
                 patch_source=patch_source, patch_ranges=patch_ranges)
    if out != target:
        os.replace(out, target)
    purge_erased_for(video, keep=target)
    info["path"] = str(target)
    return str(target), sig, info
