from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path

import av
import cv2
import numpy as np

from core.paths import base_dir

BASE = Path(base_dir())
MODELS = BASE / "models"
LOGS = BASE / "logs"
WORK = BASE / ".work"      # 功能性临时产物(快照/ffmpeg日志/状态/擦除缓存), 非调试模式下绝不写入 logs/


def run_folder(prefix, cfg=None):
    root = LOGS if bool(getattr(cfg, "debug_mode", False)) else WORK
    folder = root / (prefix + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


_live_lock = threading.Lock()
_live_value = None
_live_last = {}

SIG_W, SIG_H = 128, 32      # 墨迹签名分辨率


def check_abort(abort):
    if abort and abort():
        raise RuntimeError("已取消")


def publish(path, phase, t, frame, boxes=None, baked=False, force=False):
    """单槽邮箱: 节流判定和写槽都在锁内, 多个 worker 并发时不会互相踩掉节流状态。"""
    global _live_value
    now = time.monotonic()
    key = (str(path), phase)
    with _live_lock:
        if not force and now - _live_last.get(key, 0.0) < 0.12:
            return
        _live_last[key] = now
    data = {
        "path": os.path.abspath(path), "phase": phase, "t": float(t),
        "frame": np.ascontiguousarray(frame).copy(),
        "boxes": boxes or [], "baked": bool(baked),
    }
    with _live_lock:
        _live_value = data


def take_live():
    global _live_value
    with _live_lock:
        value, _live_value = _live_value, None
    return value


# H.273 色彩编码 → ffmpeg 参数名(只透传认得的; 未知的不写, 让编码器按默认处理)
_CS = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m", 9: "bt2020nc", 10: "bt2020c"}
_PRIM = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020"}
_TRC = {1: "bt709", 6: "smpte170m", 8: "linear", 13: "iec61966-2-1", 14: "bt2020-10",
        15: "bt2020-12", 16: "smpte2084", 18: "arib-std-b67"}
_RANGE = {1: "tv", 2: "pc"}


def _color_info(ctx):
    out = {}
    for key, attr, table in (("space", "colorspace", _CS), ("primaries", "color_primaries", _PRIM),
                             ("trc", "color_trc", _TRC), ("range", "color_range", _RANGE)):
        try:
            v = getattr(ctx, attr, None)
            v = int(v) if v is not None else None
        except (TypeError, ValueError):
            v = None
        out[key] = table.get(v)
    return out


def video_meta(path):
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError("文件没有视频流")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 25.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        else:
            duration = 0.0
        ctx = stream.codec_context
        pix = str(getattr(ctx, "pix_fmt", "") or "")
        color = _color_info(ctx)
        audio_codec = ""
        if container.streams.audio:
            try:
                audio_codec = str(container.streams.audio[0].codec_context.name or "")
            except Exception:
                audio_codec = ""
        return {
            "w": int(ctx.width), "h": int(ctx.height), "fps": fps,
            "duration": max(0.0, float(duration)),
            "audio": bool(container.streams.audio), "audio_codec": audio_codec,
            "pix_fmt": pix, "color": color,
            "high_depth": any(tag in pix for tag in ("10", "12", "16")),
            "hdr": color.get("trc") in ("smpte2084", "arib-std-b67"),
        }


# ================= 硬件解码(可选, 启动期失败自动回退 PyAV 软解) =================

_HW = {}


def _ffmpeg_exe():
    if "ffmpeg" not in _HW:
        try:
            from core.inpaint import ffmpeg_bin
            _HW["ffmpeg"] = ffmpeg_bin() or "ffmpeg"
        except Exception:
            _HW["ffmpeg"] = "ffmpeg"
    return _HW["ffmpeg"]


def hw_kind():
    """探测可用硬解方式: dxva2 → d3d11va → cuda → None(只用软解)。结果进程内缓存。"""
    if "kind" not in _HW:
        kind = None
        ff = _ffmpeg_exe()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        for cand in ("dxva2", "d3d11va", "cuda"):
            try:
                r = subprocess.run(
                    [ff, "-hide_banner", "-loglevel", "error", "-hwaccel", cand,
                     "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                     "-frames:v", "1", "-f", "null", "-"],
                    capture_output=True, timeout=20, creationflags=flags)
                if r.returncode == 0:
                    kind = cand
                    break
            except Exception:
                continue
        _HW["kind"] = kind
    return _HW["kind"]


def _hw_enabled():
    if "enabled" not in _HW:
        try:
            from config import AppConfig
            _HW["enabled"] = bool(getattr(AppConfig.load(), "hw_decode", True))
        except Exception:
            _HW["enabled"] = True
    return _HW["enabled"]


def _grab_frame_hw(path, t, kind):
    """ffmpeg -hwaccel 单发精确寻址取一帧; 失败返回 None。"""
    meta = video_meta(path)
    w, h = int(meta["w"]), int(meta["h"])
    if not w or not h:
        return None
    ff = _ffmpeg_exe()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-hwaccel", kind,
             "-ss", f"{max(0.0, float(t)):.3f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "-dn", "-"],
            capture_output=True, timeout=60, creationflags=flags)
    except Exception:
        return None
    size = w * h * 3
    if r.returncode != 0 or len(r.stdout) < size:
        return None
    return np.frombuffer(r.stdout[:size], np.uint8).reshape(h, w, 3).copy()


def iter_frames(path, t0=0.0, t1=None, abort=None):
    """返回 (帧起点PTS, 下一帧PTS, BGR)。包含覆盖 t0 的那一帧。
    顺序取帧一律走 PyAV 软解(进程内解码): 实测 ffmpeg -hwaccel 子进程管道(4K 裸帧 25MB/帧)
    吞吐是瓶颈, 比进程内解码慢约 18 倍, 是定位/擦除变慢的回归, 故不用于顺序取帧。
    硬件解码只用于 grab_frame 的单发精确寻址(那里瓶颈是寻址, 不是吞吐)。
    t0 > 1s 且容器给出 start_time 时, 先 seek 到 t0 前约 1s 的关键帧再顺序解码。"""
    yield from _iter_frames_sw(path, t0, t1, abort)


def _iter_frames_sw(path, t0=0.0, t1=None, abort=None):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        rate = stream.average_rate or stream.base_rate
        dt = 1.0 / float(rate or 25)
        origin = (float(stream.start_time * stream.time_base)
                  if stream.start_time is not None else None)
        previous = None
        fallback_t = 0.0
        if t0 > 1.0 and origin is not None:
            lead = t0 - 1.0
            try:
                container.seek(int((lead + origin) / float(stream.time_base)),
                               stream=stream, backward=True, any_frame=False)
                fallback_t = lead
            except Exception:
                pass
        for frame in container.decode(stream):
            check_abort(abort)
            stamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
            if origin is None and stamp is not None:
                origin = stamp
            t = stamp - (origin or 0.0) if stamp is not None else fallback_t
            t = max(0.0, float(t))
            fallback_t = t + dt
            if previous is not None:
                pt, pf = previous
                end = max(pt + 1e-6, t)
                if end > t0 and (t1 is None or pt < t1):
                    yield max(pt, t0), min(end, t1) if t1 else end, pf
                if t1 is not None and t >= t1:
                    return
            previous = (t, frame.to_ndarray(format="bgr24"))
        if previous is not None:
            pt, pf = previous
            end = pt + dt
            if end > t0 and (t1 is None or pt < t1):
                yield max(pt, t0), min(end, t1) if t1 else end, pf


def grab_frame(path, t):
    """取覆盖时刻 t 的一帧(与流水线同一时间轴); 失败返回 None。
    有可用硬解时优先 ffmpeg 单发精确寻址(长 GOP 的 4K 也能在约半秒内取到), 失败回退 PyAV。"""
    if _hw_enabled():
        kind = hw_kind()
        if kind:
            frame = _grab_frame_hw(path, t, kind)
            if frame is not None:
                return frame
    try:
        for _start, _end, frame in _iter_frames_sw(path, max(0.0, float(t)), None):
            return frame
    except Exception:
        return None
    return None


def _iter_frames_overlapped(path, t0=0.0, t1=None, abort=None, prefetch=8):
    """后台线程预取解码帧, 让解码与检测/擦除/编码重叠(语义与 iter_frames 一致)。
    消费方提前退出(异常/取消)时通过 stop 事件让生产线程立刻收工, 不再卡在满队列上。"""
    import queue as _queue
    import threading as _threading
    q = _queue.Queue(maxsize=max(2, int(prefetch)))
    SENTINEL = object()
    stop = _threading.Event()

    def _abort():
        return stop.is_set() or bool(abort and abort())

    def _put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except _queue.Full:
                continue
        return False

    def _run():
        try:
            for item in iter_frames(path, t0, t1, _abort):
                if not _put(item):
                    return
            _put(SENTINEL)
        except BaseException as exc:
            if not stop.is_set():
                _put(exc)

    _threading.Thread(target=_run, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def fmt_mmss(sec):
    """秒 → MM:SS(整秒, 用于进度/时长/状态/时间轴)。"""
    sec = max(0, int(round(float(sec))))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def fmt_time_cs(sec):
    """秒 → MM:SS.CC(百分秒, 用于字幕时间列, 保留编辑精度)。"""
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    return f"{m:02d}:{sec - m * 60:05.2f}"


def parse_time(text):
    """解析用户输入的时间: 支持 'MM:SS'/'MM:SS.CC'/'HH:MM:SS.CC'/纯秒数; 失败返回 None。"""
    s = str(text).strip().replace(",", ".")
    if not s:
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                return round(int(parts[0]) * 60 + float(parts[1]), 2)
            if len(parts) == 3:
                return round(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]), 2)
            return None
        return round(float(s), 2)
    except ValueError:
        return None


class FrameCursor:
    """顺序取帧游标: frame_at(t) 返回覆盖时刻 t 的帧(t 只能单调前进)。增量擦除时读旧缓存用。"""

    def __init__(self, path, t0=0.0, t1=None, abort=None):
        self._it = iter_frames(path, t0, t1, abort)
        self._cur = None

    def frame_at(self, t):
        while True:
            if self._cur is not None and t < self._cur[1] - 1e-8:
                return self._cur[2]
            try:
                self._cur = next(self._it)
            except StopIteration:
                return self._cur[2] if self._cur is not None else None

    def close(self):
        try:
            self._it.close()
        except Exception:
            pass


def ordered_quad(points, w=None, h=None):
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("无效四点框")
    center = q.mean(axis=0)
    order = np.argsort(np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0]))
    q = q[order]
    q = np.roll(q, -int(np.argmin(q[:, 0] + q[:, 1])), axis=0)
    if w is not None:
        q[:, 0] = np.clip(q[:, 0], 0, w)
        q[:, 1] = np.clip(q[:, 1], 0, h)
    if abs(cv2.contourArea(q)) < 4 or not cv2.isContourConvex(q):
        raise ValueError("退化或非凸四点框")
    return q


def qbox(q):
    p = np.asarray(q, dtype=float)
    return (float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max()))


def qsize(q):
    p = np.asarray(q, dtype=float)
    width = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    height = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    return max(1.0, float(width)), max(1.0, float(height))


def qangle(q):
    p = np.asarray(q, dtype=float)
    return math.degrees(math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))


def qiou(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
        union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
        return float(inter / max(union, 1e-6))
    except cv2.error:
        return 0.0


def qcontain(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
        smaller = min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b)))
        return float(inter / max(smaller, 1e-6))
    except cv2.error:
        return 0.0


def track_quad(samples, t, fallback=None):
    if samples:
        lo, hi = 0, len(samples)
        while lo < hi:
            mid = (lo + hi) // 2
            if float(samples[mid][0]) <= t:
                lo = mid + 1
            else:
                hi = mid
        return np.asarray(samples[max(0, lo - 1)][1], np.float32)
    return None if fallback is None else np.asarray(fallback, np.float32)


def crop_quad(frame, quad, border=2, scale=1.0):
    q = ordered_quad(quad)
    width, height = qsize(q)
    width = max(4, int(round(width * scale)))
    height = max(4, int(round(height * scale)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(q, dst)
    crop = cv2.warpPerspective(frame, transform, (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if border:
        crop = cv2.copyMakeBorder(crop, border, border, border, border, cv2.BORDER_REPLICATE)
    return crop


def downscale(frame, max_w=1280):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    k = max_w / float(w)
    return cv2.resize(frame, (max_w, max(2, int(round(h * k)))),
                      interpolation=cv2.INTER_AREA), k


def ink_signature(frame, q):
    try:
        crop = crop_quad(frame, q, border=0)
    except (ValueError, cv2.error):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 2.5)
    hp -= hp.mean()
    n = float(np.linalg.norm(hp))
    return hp / n if n > 1e-6 else None


def _ncc(a, b):
    return float(np.clip(float((a * b).sum()), -1.0, 1.0))


def sig_similarity(a, b):
    if a is None or b is None:
        return 0.0
    best = _ncc(a, b)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(a, b)
    except cv2.error:
        return best
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return best
    if abs(dx) > SIG_W * 0.15 or abs(dy) > SIG_H * 0.3 or (abs(dx) < 0.3 and abs(dy) < 0.3):
        return best
    for sx, sy in ((-dx, -dy), (dx, dy)):
        m = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(b, m, (SIG_W, SIG_H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shifted -= shifted.mean()
        n = float(np.linalg.norm(shifted))
        if n > 1e-6:
            best = max(best, _ncc(a, shifted / n))
    return best


def ema_signature(old, new, alpha):
    if new is None:
        return old
    if old is None:
        return new
    mixed = old * (1.0 - alpha) + new * alpha
    mixed -= mixed.mean()
    n = float(np.linalg.norm(mixed))
    return mixed / n if n > 1e-6 else old


def source_quad(ev, t=None):
    track = getattr(ev, "geom_track", None) or []
    if track and t is not None:
        return track_quad(track, t)
    q = getattr(ev, "geom", None)
    if q is not None and len(q) == 4:
        return np.asarray(q, np.float32)
    bbox = getattr(ev, "bbox", None)
    if bbox:
        x1, y1, x2, y2 = bbox
        return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return None


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path.name}")
    data.tofile(str(path))


def load_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def cleanup_run_folders(prefixes=("locate-",), max_age_hours=0.0, root=None, keep=()):
    """删除运行留下的临时目录; keep 里的目录(被工作区状态引用的定位快照)不删。"""
    root = Path(root) if root is not None else LOGS
    if not root.is_dir():
        return 0
    keep = {Path(k).resolve() for k in keep}
    n, now = 0, time.time()
    for p in root.iterdir():
        if not p.is_dir() or not any(p.name.startswith(x) for x in prefixes):
            continue
        if p.resolve() in keep:
            continue
        try:
            if max_age_hours and now - p.stat().st_mtime < max_age_hours * 3600:
                continue
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        except OSError:
            pass
    return n


def cache_key(*parts):
    value = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str,
                       separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def endpoint_identity(ep):
    return {k: ep.get(k) for k in (
        "provider", "api_base", "api_model", "ollama_url", "ollama_model",
        "local_model", "api_extra", "ollama_options")}


# ---- 结果缓存: 进程内单连接(多线程共用, 以锁串行), 不再每次新建连接 ----
_db_lock = threading.Lock()
_db_conn = None


def _db():
    global _db_conn
    if _db_conn is None:
        folder = MODELS / "cache"
        folder.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(folder / "suisen-v6.sqlite"), timeout=30,
                               check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit()
        _db_conn = conn
    return _db_conn


def cache_get(key):
    with _db_lock:
        row = _db().execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(key, value):
    with _db_lock:
        conn = _db()
        conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                     (key, json.dumps(value, ensure_ascii=False)))
        conn.commit()
