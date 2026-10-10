from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core import ai
from core.enhance import estimate_bg_color, estimate_text_color
from core.frames import decode_jpg, encode_jpg
from core.geometry import quad_area, quad_axis, quad_from_rect
from core.models import SubtitleEvent, flag_growing_tracks, stabilize_tracks
from core.ocr_protocols import is_glm_ocr, validate_ocr
from core.pipeline_quality import norm_text as _norm_text
from core.prompts import get_prompt
from core.v6_common import (
    LOGS, _iter_frames_overlapped, cache_get, cache_key, cache_put, check_abort,
    crop_quad, downscale, ema_signature, endpoint_identity, fmt_mmss, grab_frame,
    ink_signature, iter_frames, ordered_quad, publish, qangle, qbox, qcontain,
    qiou, qsize, run_folder, sig_similarity, track_quad, video_meta,
)

log = logging.getLogger("suisen.pipeline")

SNAP_MAX = 6
SNAP_EVERY = 0.5


@dataclass
class SegRecord:
    id: str
    start: float
    end: float
    bbox: Tuple[float, float, float, float]
    angle: float = 0.0
    vertical: bool = False
    conf: float = 0.8
    snaps: List[list] = field(default_factory=list)   # [t, 裁剪图路径(或未落盘的jpg字节), 偏移x, 偏移y]
    text: str = ""
    rconf: float = 0.0
    source: str = ""
    verdict: str = ""
    hits: int = 0
    score: float = 0.0
    detector: str = ""
    verified: bool = False
    verify_conf: float = 0.0
    verify_reason: str = ""
    evidence: dict = field(default_factory=dict)
    quad_pts: list = field(default_factory=list)
    sig: object = field(default=None, repr=False, compare=False)

    def quad(self):
        if len(self.quad_pts) == 4:
            try:
                return [[float(p[0]), float(p[1])] for p in self.quad_pts]
            except Exception:
                pass
        x1, y1, x2, y2 = self.bbox
        if abs(self.angle) >= 2.0 and not self.vertical:
            try:
                return quad_from_rect((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, self.angle)
            except Exception:
                pass
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def quad_at(self, t):
        return track_quad(self.evidence.get("quad_track"), t, self.quad())

    def set_quad(self, quad):
        """手动调整定位框: 四点框成为唯一几何(逐帧轨迹清掉), bbox/angle 同步; 识别时按新框读字。"""
        pts = [[float(p[0]), float(p[1])] for p in quad]
        self.quad_pts = pts
        self.bbox = qbox(pts)
        self.angle = 0.0 if self.vertical else float(qangle(pts))
        self.evidence.pop("quad_track", None)

    def brief(self):
        return {
            "id": self.id,
            "snaps": [list(x) for x in self.snaps
                      if isinstance(x, (list, tuple)) and len(x) >= 4 and isinstance(x[1], str)],
            "t0": round(self.start, 2), "t1": round(self.end, 2),
            "bbox": [round(v, 1) for v in self.bbox],
            "angle": round(self.angle, 2), "vertical": self.vertical,
            "conf": round(self.conf, 3), "hits": self.hits, "score": round(self.score, 3),
            "detector": self.detector, "verified": self.verified,
            "verify_conf": round(self.verify_conf, 3), "verify_reason": self.verify_reason,
            "text": self.text, "rconf": round(self.rconf, 3), "src": self.source,
            "verdict": self.verdict,
            "evidence": {k: v for k, v in self.evidence.items()},
            "quad": [[round(p[0], 1), round(p[1], 1)] for p in self.quad()],
        }


def segs_brief(segs):
    return [s.brief() if isinstance(s, SegRecord) else dict(s) for s in segs]


def segs_from_brief(items):
    segs = []
    for s in items or []:
        if isinstance(s, SegRecord):
            segs.append(s)
            continue
        if not isinstance(s, dict):
            continue
        try:
            q = s.get("quad") or []
            qpts = [p for p in q if isinstance(p, (list, tuple)) and len(p) == 2][:4]
            quad_pts = [[float(p[0]), float(p[1])] for p in qpts] if len(qpts) == 4 else []
            segs.append(SegRecord(
                id=str(s.get("id", "")), snaps=[list(x) for x in s.get("snaps", [])],
                start=float(s.get("t0", 0.0)), end=float(s.get("t1", 0.0)),
                bbox=tuple(float(v) for v in s.get("bbox", (0, 0, 0, 0))),
                angle=float(s.get("angle", 0.0)), vertical=bool(s.get("vertical", False)),
                conf=float(s.get("conf", 0.8)), text=str(s.get("text", "")),
                rconf=float(s.get("rconf", 0.0)), source=str(s.get("src", "")),
                verdict=str(s.get("verdict", "")), hits=int(s.get("hits", 0)),
                score=float(s.get("score", 0.0)), detector=str(s.get("detector", "")),
                verified=bool(s.get("verified", False)), verify_conf=float(s.get("verify_conf", 0.0)),
                verify_reason=str(s.get("verify_reason", "")),
                evidence=dict(s.get("evidence", {}) or {}), quad_pts=quad_pts,
            ))
        except Exception:
            continue
    return segs


# ================= 工具 =================

def _sim(a, b):
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return 1.0 if na == nb else 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _part_sim(na, nb):
    if not na or not nb:
        return 0.0
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    matcher = SequenceMatcher(None, short, long_)
    return sum(b.size for b in matcher.get_matching_blocks()) / len(short)


def _area_ratio(a, b):
    try:
        aa, bb = quad_area(a), quad_area(b)
    except (ValueError, TypeError):
        return 1.0
    return min(aa, bb) / max(1e-6, max(aa, bb))


def _pick_snaps(snaps, n=3):
    snaps = [s for s in snaps if isinstance(s, (list, tuple)) and len(s) >= 4]
    if len(snaps) <= n:
        return snaps
    idx = sorted(set(np.linspace(0, len(snaps) - 1, n).round().astype(int).tolist()))
    return [snaps[i] for i in idx]


def _thin_snaps(snaps, keep=SNAP_MAX):
    if len(snaps) <= keep:
        return snaps
    idx = set(np.linspace(0, len(snaps) - 1, keep).round().astype(int).tolist())
    kept = []
    for i, s in enumerate(snaps):
        if i in idx:
            kept.append(s)
        elif isinstance(s[1], str):
            try:
                Path(s[1]).unlink(missing_ok=True)
            except Exception:
                pass
    return kept


def _snapshot(seg, t, frame, q):
    if seg.snaps and t - seg.snaps[-1][0] < SNAP_EVERY:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = qbox(q)
    pad = max(12.0, 0.6 * (y2 - y1), 0.12 * (x2 - x1))
    xa, ya = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
    xb, yb = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
    if xb - xa < 4 or yb - ya < 4:
        return
    data = encode_jpg(frame[ya:yb, xa:xb], 88)
    if not data:
        return
    seg.snaps.append([float(t), data, xa, ya])
    if len(seg.snaps) > SNAP_MAX:
        seg.snaps = _thin_snaps(seg.snaps)


def _flush_snaps(seg, folder):
    n = int(seg.evidence.get("snap_n", 0))
    for s in seg.snaps:
        if isinstance(s[1], (bytes, bytearray)):
            path = folder / f"{seg.id}-{n:02d}.jpg"
            n += 1
            path.write_bytes(s[1])
            s[1] = str(path)
    seg.evidence["snap_n"] = n


def _light(segs):
    out = []
    for s in segs:
        c = copy.copy(s)
        c.sig = None
        c.snaps = [list(x) for x in s.snaps if isinstance(x[1], str)]
        c.quad_pts = [list(p) for p in s.quad_pts]
        c.evidence = {}
        out.append(c)
    return out


def _merge_into(target, s):
    target.start = min(target.start, s.start)
    target.end = max(target.end, s.end)
    total = max(1, target.hits + s.hits)
    target.score = (target.score * target.hits + s.score * s.hits) / total
    target.conf = max(target.conf, s.conf)
    qt = target.evidence.setdefault("quad_track", [])
    qt.extend(s.evidence.get("quad_track", []))
    qt.sort(key=lambda x: x[0])
    target.snaps = _thin_snaps(sorted(target.snaps + s.snaps, key=lambda x: x[0]))
    if target.sig is not None and s.sig is not None:
        target.sig = ema_signature(target.sig, s.sig, s.hits / total)
    elif s.sig is not None:
        target.sig = s.sig
    target.hits = total
    target.evidence["merged"] = int(target.evidence.get("merged", 0)) + 1


def _fragment_of(tr, q, t, thr, step):
    if qcontain(tr["q"], q) < 0.6:
        return False
    if tr["score"] >= thr:
        return True
    if t - tr["last_seen"] > step + 1e-6:
        return False
    _, _, h_tr = quad_axis(tr["q"])
    _, _, h_q = quad_axis(q)
    return (quad_area(q) < 0.7 * quad_area(tr["q"]) and abs(h_tr - h_q) <= 0.6 * max(h_tr, h_q))


# ================= 阶段① 定位 =================

def stage_locate(video, cfg, progress, abort, t0=0.0, t1=None,
                 partial=None, pbase=0, pspan=100):
    from core.locator import get_locator
    progress(pbase, "定位: 加载检测器(首次使用会自动下载 ONNX 模型)")
    detector = get_locator(cfg)
    meta = video_meta(video)
    fps = float(meta["fps"] or 25.0)
    w, h = meta["w"], meta["h"]
    t1 = min(t1, meta["duration"]) if t1 is not None else (meta["duration"] or None)

    step = max(0.08, float(cfg.sample_interval))
    thr = float(np.clip(getattr(cfg, "track_similarity", 0.5), 0.2, 0.95))
    precise = bool(getattr(cfg, "precise_timing", True))
    gap_tol = max(1.5 / fps, float(getattr(cfg, "locator_gap_tolerance", 0.35)))
    if not precise:
        gap_tol = max(gap_tol, step * 1.5)
    probe = bool(getattr(cfg, "locator_flash_probe", True))
    cut_thr = float(getattr(cfg, "locator_cut_threshold", 22.0))
    min_hits = max(1, int(getattr(cfg, "locator_min_hits", 2)))
    min_dur = max(0.0, float(getattr(cfg, "min_seg_dur", 0.25)))
    max_active = max(4, int(getattr(cfg, "locator_max_active", 40)))
    folder = run_folder("locate-", cfg)

    ring = deque(maxlen=int(math.ceil((step + gap_tol) * fps)) + 2)
    active, finished = [], []
    stats = {"noise": 0, "single": 0, "overflow": 0}
    next_detect = float(t0)
    prev_small = None
    n_det = 0
    last_partial = last_progress = 0.0
    started = time.monotonic()
    duration = max(0.01, (t1 or max(meta["duration"], 1.0)) - t0)

    def confirmed(tr, now_t):
        seg = tr["seg"]
        return seg.hits >= min_hits or (now_t - seg.start) >= min_dur

    def finish(tr):
        seg = tr["seg"]
        dur = float(tr["seen_end"]) - seg.start
        if dur < min_dur:
            stats["noise"] += 1
            return
        if seg.hits < min_hits and tr["det_miss"] >= 2:
            stats["single"] += 1
            return
        seg.end = max(float(tr["seen_end"]), seg.start + 0.02)
        q = np.asarray(tr["q"], np.float32)
        seg.quad_pts = q.tolist()
        seg.bbox = qbox(q)
        seg.angle = float(qangle(q)) if not seg.vertical else 0.0
        seg.sig = tr["sig"]
        _flush_snaps(seg, folder)
        finished.append(seg)

    # 定位阶段的进度拆分: 扫描占前 75%(按视频时间线性), 候选核验占后 25%(按已核验数线性)——
    # 不再把核验钉在 99%, 进度条全程线性。
    scan_span = max(1, int(pspan * 0.75))
    verify_span = pspan - scan_span - 1
    for t, end, frame in _iter_frames_overlapped(video, t0, t1, abort):   # 解码与检测重叠(~1.6×)
        check_abort(abort)
        work, k = downscale(frame, 1280)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = prev_small is not None and float(np.abs(small - prev_small).mean()) >= cut_thr
        prev_small = small
        run_det = t + 1e-6 >= next_detect or (probe and cut)

        if precise or run_det:
            for tr in active:
                now_sig = ink_signature(work, tr["q"] * k)
                s = sig_similarity(tr["sig"], now_sig)
                tr["score"] = s
                if s >= thr:
                    tr["last_seen"], tr["seen_end"] = t, end
                    tr["seg"].end = end
                    if s >= 0.7 and now_sig is not None:
                        tr["sig"] = ema_signature(tr["sig"], now_sig, 0.06)

        kept = []
        for tr in active:
            if t - tr["last_seen"] > gap_tol:
                finish(tr)
            else:
                kept.append(tr)
        active = kept

        if run_det:
            n_det += 1
            next_detect = t + step
            try:
                detections = detector.detect(frame)
            except Exception as exc:
                log.warning("检测失败 t=%.2f: %s", t, exc)
                detections = []
            prepared = []
            for d in detections:
                try:
                    prepared.append((d, ordered_quad(d["quad"], w, h)))
                except (ValueError, KeyError, TypeError):
                    continue
            cands = []
            for ti, tr in enumerate(active):
                for di, (d, q) in enumerate(prepared):
                    iou = qiou(tr["q"], q)
                    cont = qcontain(tr["q"], q)
                    if iou < 0.45 and cont < 0.75:
                        continue
                    app = tr["score"] if tr["score"] >= thr else \
                        sig_similarity(tr["sig"], ink_signature(work, q * k))
                    if app < thr:
                        continue
                    cands.append((iou + 0.5 * cont + app, ti, di, iou))
            cands.sort(reverse=True)
            used_t, used_d = set(), set()
            for _, ti, di, iou in cands:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                tr = active[ti]
                d, q = prepared[di]
                weak = bool(d.get("weak"))
                seg = tr["seg"]
                tr["last_seen"], tr["seen_end"] = t, end
                tr["score"] = max(tr["score"], thr)
                seg.end = end
                if not weak:
                    seg.hits += 1
                    seg.score += (float(d.get("conf", 0.8)) - seg.score) / seg.hits
                    if not seg.detector:
                        seg.detector = str(d.get("source", ""))
                    if iou >= 0.6:
                        newq = (tr["q"] * 0.7 + q * 0.3).astype(np.float32)
                        if float(np.max(np.abs(newq - tr["q"]))) >= 0.5:
                            tr["q"] = newq
                            seg.evidence["quad_track"].append([float(t), newq.tolist()])
                        tr["sig"] = ema_signature(tr["sig"], ink_signature(work, tr["q"] * k), 0.25)
                _snapshot(seg, t, frame, tr["q"])
            for ti, tr in enumerate(active):
                if ti not in used_t and tr["score"] >= thr:
                    tr["det_miss"] += 1
            fresh = []
            for di, (d, q) in enumerate(prepared):
                if di in used_d or d.get("weak"):
                    continue
                if any(_fragment_of(tr, q, t, thr, step) for tr in active):
                    continue
                sig = ink_signature(work, q * k)
                if sig is None:
                    continue
                start = float(t)
                for bt, _bend, bwork in reversed(ring):
                    if bt >= t:
                        continue
                    if sig_similarity(sig, ink_signature(bwork, q * k)) >= thr:
                        start = float(bt)
                    else:
                        break
                seg = SegRecord(
                    id=uuid.uuid4().hex[:12], start=start, end=float(end),
                    bbox=qbox(q), angle=float(qangle(q)),
                    vertical=bool(d.get("vertical") and cfg.vertical_support),
                    conf=float(d.get("conf", 0.8)), hits=1, score=float(d.get("conf", 0.8)),
                    detector=str(d.get("source", "")), quad_pts=q.tolist(),
                    evidence={"quad_track": [[start, q.tolist()]]},
                )
                _snapshot(seg, t, frame, q)
                fresh.append({"seg": seg, "q": q.astype(np.float32), "sig": sig,
                              "score": 1.0, "last_seen": t, "seen_end": end, "det_miss": 0})
            active.extend(fresh)
            if len(active) > max_active:
                active.sort(key=lambda x: (x["seg"].hits, x["seg"].score), reverse=True)
                stats["overflow"] += len(active) - max_active
                active = active[:max_active]

        ring.append((float(t), float(end), work))
        publish(video, "locate", t, frame, [
            {"quad": tr["q"].tolist(), "ok": tr["score"] >= thr, "label": tr["seg"].id[:6]}
            for tr in active], baked=True)
        now = time.monotonic()
        if now - last_progress >= 0.25:
            last_progress = now
            done = t - t0
            elapsed = now - started
            eta = f" · 剩余约 {int((duration - done) * elapsed / done) // 60:02d}:{int((duration - done) * elapsed / done) % 60:02d}" \
                if elapsed > 3 and done > 0 else ""
            progress(pbase + min(scan_span, int(done * scan_span / duration)),
                     f"定位 {fmt_mmss(t)}/{fmt_mmss(t0 + duration)} · 已找到 {len(finished)} 段, 跟踪中 {len(active)}{eta}")
        if partial and now - last_partial >= 1.5:
            last_partial = now
            live = []
            for tr in active:
                if confirmed(tr, t):
                    tr["seg"].quad_pts = tr["q"].tolist()
                    live.append(tr["seg"])
            partial(_light(finished + live))

    for tr in active:
        finish(tr)
    segs, n_merged, n_frag = _postprocess_segments(finished, cfg)

    try:
        from core.pipeline_quality import verify_segments_vlm
        verify_segments_vlm(segs, cfg,
                            lambda p, m: progress(
                                pbase + scan_span + min(
                                    verify_span, int(p * verify_span / 100)), m),
                            abort, meta)
    except RuntimeError:
        raise
    except Exception as exc:
        log.warning("候选核验跳过: %s", exc)

    segs.sort(key=lambda s: (s.start, s.bbox[1], s.bbox[0]))
    if partial:
        partial(_light(segs))
    ok = sum(1 for s in segs if not s.verdict)
    dropped = stats["noise"] + stats["single"] + stats["overflow"]
    progress(pbase + pspan,
             f"定位完成: {len(segs)} 个候选段(通过 {ok}) · 已排除噪声 {dropped + n_frag} · 接回断句 {n_merged}")
    return segs, meta


def _postprocess_segments(segs, cfg):
    segs = [s for s in segs if s.end > s.start and s.snaps]
    segs.sort(key=lambda s: (s.start, s.bbox[1], s.bbox[0]))
    thr = float(np.clip(getattr(cfg, "track_similarity", 0.5), 0.2, 0.95))
    bridge = max(0.0, float(getattr(cfg, "seg_bridge_gap", 1.2)))
    merged, n_merged = [], 0
    for s in segs:
        target = None
        for k in reversed(merged[-60:]):
            if s.start - k.end > bridge or k.vertical != s.vertical:
                continue
            if qiou(k.quad(), s.quad()) < 0.55 or sig_similarity(k.sig, s.sig) < thr:
                continue
            target = k
            break
        if target is None:
            merged.append(s)
        else:
            _merge_into(target, s)
            n_merged += 1
    static_max = float(getattr(cfg, "static_text_max_dur", 60.0) or 0.0)
    for s in merged:
        if static_max > 0 and (s.end - s.start) >= static_max:
            s.verdict = f"静止≥{fmt_mmss(static_max)}: 疑似台标/水印"
    merged.sort(key=lambda s: (s.start, s.bbox[1], s.bbox[0]))

    n_frag = 0
    for s in merged:
        if s.verdict:
            continue
        sq = s.quad()
        s_area, s_dur = quad_area(sq), max(1e-3, s.end - s.start)
        _, _, s_h = quad_axis(sq)
        for k in merged:
            if k is s or k.verdict:
                continue
            if k.start >= s.end:
                break
            if k.end <= s.start or k.vertical != s.vertical:
                continue
            if (min(k.end, s.end) - max(k.start, s.start)) / s_dur < 0.6:
                continue
            kq = k.quad()
            if s_area >= 0.7 * quad_area(kq):
                continue
            _, _, k_h = quad_axis(kq)
            if abs(k_h - s_h) > 0.6 * max(k_h, s_h) or qcontain(kq, sq) < 0.8:
                continue
            s.verdict = "碎片: 同一行文字的局部(已由完整框覆盖)"
            n_frag += 1
            break
    return merged, n_merged, n_frag


# ================= 阶段② 读字 =================

class LocalLineReader:
    """本地读字: RapidOCR 单行识别(ONNX)。"""

    def __init__(self, cfg):
        from core.locator import RapidLineReader
        self.model = RapidLineReader(str(getattr(cfg, "ocr_lang", "ch")), cfg)
        log.info("本地读字引擎: RapidOCR")

    def read_line(self, crop):
        return self.model.read_line(crop)

    def read(self, crop, vertical=False):
        from core.pipeline_quality import split_chars, split_text_lines, stitch_chars
        texts, scores = [], []
        if vertical:
            for column in split_text_lines(crop, True, max_lines=8):
                line = stitch_chars(split_chars(column))
                if line is None:
                    continue
                text, score = self.read_line(line)
                if text:
                    texts.append(text.replace(" ", ""))
                    scores.append(score)
        else:
            for part in split_text_lines(crop, False):
                text, score = self.read_line(part)
                if text:
                    texts.append(text)
                    scores.append(score)
        if not texts:
            return "", 0.0
        return validate_ocr("\n".join(texts), multiline=True), float(np.mean(scores))


def _vlm_schema_echo(text):
    """模型把提示词里的字段名/占位符当内容输出(schema 回显): 形如 {"text":"text"} 或 "可见文字"。
    没有字幕的裁剪图上小模型常犯这个错; 这类值一律视为"无结果", 不落成字幕。"""
    return str(text or "").strip().lower() in ("text", "可见文字")


def _read_vlm(client, crop, cfg, vertical=False):
    h, w = crop.shape[:2]
    if w < 640:
        s = min(3.0, 640.0 / max(1, w))
        crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    budget = int(getattr(cfg, "ocr_max_tokens", 192))
    if is_glm_ocr(cfg.ep("vlm")):
        raw = client.ask(crop, "Text Recognition:", max_tokens=budget)
        return validate_ocr(raw, multiline=True)
    prompt = get_prompt(cfg, "vlm_read")
    if vertical:
        prompt += "本图是竖排文字: 按列从右到左、每列从上到下的顺序转写, 列与列之间用换行分隔。"
    raw = client.ask(crop, prompt, max_tokens=budget)
    data = ai.extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("视觉读字未返回 text 字符串")
    text = validate_ocr(data["text"], multiline=True)
    if _vlm_schema_echo(text):
        return ""
    return text


def read_region(crop, vertical, cfg, client, get_local, diagnostics=None, budget=None, seg_id=""):
    """一张裁剪图的读字路由(自动流程与手动[识别选中]共用):
    local/auto 先本地; vlm 或 auto 的竖排/低置信/无结果 → VLM(带结果缓存与调用预算);
    VLM 不可用/失败且允许回退 → 本地。budget=[已用, 上限]。返回 (text, score, source)。"""
    route = str(cfg.ocr_engine or "vlm")
    conf_thr = float(getattr(cfg, "conf_threshold", 0.6))
    diag = diagnostics if diagnostics is not None else []
    text, score, source = "", 0.0, ""
    if route in ("auto", "local"):
        try:
            text, score = get_local().read(crop, vertical)
            source = "local"
        except Exception as exc:
            diag.append({"id": seg_id, "local": str(exc)})
    need_vlm = route == "vlm" or (route == "auto" and (vertical or not text or score < conf_thr))
    fallback_ok = route == "vlm" and bool(getattr(cfg, "ocr_local_fallback", True))
    if need_vlm and client is not None:
        digest = hashlib.sha256(crop.tobytes()).hexdigest()
        key = cache_key("ocr-v7", digest, endpoint_identity(cfg.ep("vlm")))
        cached = cache_get(key)
        try:
            if cached is None:
                if budget is not None:
                    if budget[0] >= budget[1]:
                        raise RuntimeError("达到本视频 AI 读字调用上限")
                    budget[0] += 1
                candidate = _read_vlm(client, crop, cfg, vertical=vertical)
                cache_put(key, candidate or "")
            else:
                candidate = validate_ocr(cached, multiline=True) if cached else ""
            if candidate and _vlm_schema_echo(candidate):
                candidate = ""        # 新调用与缓存命中都拦: schema 回显不算读字结果
            if candidate:
                text, score, source = candidate, 0.8, "vlm"
        except Exception as exc:
            diag.append({"id": seg_id, "vlm": str(exc)})
            if not text and fallback_ok:
                try:
                    text, score = get_local().read(crop, vertical)
                    source = "local-fallback"
                except Exception as fe:
                    diag.append({"id": seg_id, "fallback": str(fe)})
    elif need_vlm and not text and fallback_ok:
        try:
            text, score = get_local().read(crop, vertical)
            source = "local-fallback"
        except Exception as fe:
            diag.append({"id": seg_id, "fallback": str(fe)})
    return text, float(score), source


def _vote(votes):
    texts = [v[0] for v in votes]
    counts = Counter(_norm_text(t) for t in texts)
    key, count = counts.most_common(1)[0]
    if count >= 2:
        best = max((t for t in texts if _norm_text(t) == key), key=len)
        return best, count, 0.95

    def rank(i):
        others = [_sim(texts[i], texts[j]) for j in range(len(texts)) if j != i]
        return (float(np.mean(others)) if others else 0.0) + 0.3 * float(votes[i][1])

    i = max(range(len(texts)), key=rank)
    return texts[i], 1, min(0.65, max(0.4, float(votes[i][1])))


def _seg_bg_color(seg):
    if not seg.snaps or not isinstance(seg.snaps[0][1], str):
        return "#101010"
    img = decode_jpg(seg.snaps[0][1])
    if img is None:
        return "#101010"
    ox, oy = float(seg.snaps[0][2]), float(seg.snaps[0][3])
    x1, y1, x2, y2 = seg.bbox
    return estimate_bg_color(img, x1 - ox, y1 - oy, x2 - ox, y2 - oy)


def _seg_text_color(seg):
    if not seg.snaps or not isinstance(seg.snaps[0][1], str):
        return None
    img = decode_jpg(seg.snaps[0][1])
    if img is None:
        return None
    ox, oy = float(seg.snaps[0][2]), float(seg.snaps[0][3])
    x1, y1, x2, y2 = seg.bbox
    return estimate_text_color(img, x1 - ox, y1 - oy, x2 - ox, y2 - oy)


def _make_event(seg, text, confidence, cfg):
    q = np.asarray(seg.quad(), np.float32)
    center = q.mean(axis=0)
    width, height = qsize(q)
    rows = [r for r in re.split(r"\\N|\r?\n", text) if r.strip()]
    n = max(1, len(rows))
    unit = (width if seg.vertical else height) / n
    color = cfg.subtitle_safe_color
    outline = cfg.subtitle_safe_outline
    if bool(getattr(cfg, "auto_text_color", False)):
        got = _seg_text_color(seg)
        if got:
            color, outline = got
    ev = SubtitleEvent(
        start=seg.start, end=seg.end, text="\n".join(rows) if rows else text,
        x=float(center[0]), y=float(center[1]),
        font_size=max(8, min(300, int(unit * 0.85))),
        rotation=float(qangle(q)) if not seg.vertical else 0.0,
        vertical=bool(seg.vertical), bbox=tuple(seg.bbox),
        geom=q.tolist(), geom_anchored=True,
        geom_track=copy.deepcopy(seg.evidence.get("quad_track", [])),
        fit_mode=str(getattr(cfg, "layout_fit", "contain")),
        mask=list(seg.bbox), mask_custom=False, mask_color=_seg_bg_color(seg),
        color=color, outline_color=outline,
        source=seg.source, source_seg_id=seg.id,
        conf=confidence, locator_score=seg.score,
        sample_image=seg.snaps[0][1] if seg.snaps else "",
        sample_time=seg.snaps[0][0] if seg.snaps else seg.start,
    )
    if confidence < 0.75:
        ev.review_reason = "多张快照读字不一致, 请复核"
    return ev


def stage_read(video, segs, cfg, progress, abort, partial=None, pbase=0, pspan=100):
    meta = video_meta(video)
    route = str(cfg.ocr_engine or "vlm")
    diagnostics = []
    client = None
    if route in ("auto", "vlm"):
        try:
            from core.model_manager import prepare_endpoint
            prepare_endpoint(cfg.ep("vlm"))      # 只校验+拉起服务; 模型未安装就报错去[模型库], 绝不静默下载
            client = ai.VLMClient(cfg.ep("vlm"), cfg.vlm_detail_width, abort=abort,
                                  timeout=getattr(cfg, "request_timeout", 300))
        except Exception as exc:
            diagnostics.append({"vlm_init": str(exc)})
            if route == "vlm" and not getattr(cfg, "ocr_local_fallback", True):
                raise RuntimeError(f"AI 读字不可用: {exc}")
    local = [None]

    def get_local():
        if local[0] is None:
            local[0] = LocalLineReader(cfg)
        return local[0]

    events, snapmap = [], {}
    budget = [0, int(getattr(cfg, "ocr_request_budget", 1000))]
    gap = float(getattr(cfg, "event_merge_gap", 1.6))

    todo = [s for s in segs if not s.verdict]
    for index, seg in enumerate(todo):
        check_abort(abort)
        votes = []
        for snap in _pick_snaps(seg.snaps, 3):
            check_abort(abort)
            t, data, ox, oy = float(snap[0]), snap[1], float(snap[2]), float(snap[3])
            frame = decode_jpg(data)
            if frame is None:
                continue
            q_abs = np.asarray(seg.quad_at(t), np.float32)
            q = q_abs - np.float32([ox, oy])
            sh, sw = frame.shape[:2]
            if (q[:, 0].min() < -1 or q[:, 1].min() < -1
                    or q[:, 0].max() > sw + 1 or q[:, 1].max() > sh + 1):
                full = grab_frame(video, t)          # 定位框被手动放大到快照之外: 回原视频截整帧
                if full is not None:
                    frame, q = full, q_abs
            try:
                crop = crop_quad(frame, q, border=3)
            except (ValueError, cv2.error):
                continue
            text, score, source = read_region(crop, seg.vertical, cfg, client, get_local,
                                              diagnostics, budget, seg.id)
            if text:
                votes.append((text, float(score), source))
                if len(votes) >= 2 and _norm_text(votes[-1][0]) == _norm_text(votes[-2][0]):
                    break
        if not votes:
            seg.text = ""
            seg.verdict = "读字失败/无有效文字"
            continue
        text, _agree, confidence = _vote(votes)
        seg.text, seg.rconf = text, confidence
        seg.source = ",".join(sorted({v[2] for v in votes}))
        seg.verdict = ""
        ev = _make_event(seg, text, confidence, cfg)
        previous = next((e for e in reversed(events[-12:])
                         if _sim(e.text, ev.text) >= 0.92 and ev.start <= e.end + gap
                         and qiou(e.geom, ev.geom) >= 0.6), None)
        if previous is not None:
            previous.end = max(previous.end, ev.end)
            previous.start = min(previous.start, ev.start)
            previous.geom_track.extend(ev.geom_track)
            previous.geom_track.sort(key=lambda x: x[0])
            previous.conf = max(previous.conf, ev.conf)
            snapmap[previous.id].extend(seg.snaps)
            seg.verdict = "同位置同文连续: 已并入前一条"
        else:
            events.append(ev)
            snapmap[ev.id] = list(seg.snaps)
        progress(pbase + int((index + 1) * pspan * 0.85 / max(1, len(todo))),
                 f"读字 {index + 1}/{len(todo)} · {len(events)} 条 · AI调用 {budget[0]}")
        if partial:
            partial(copy.deepcopy(events))

    dropped = _postfilter_events(
        events, segs, cfg, meta,
        lambda p, m: progress(pbase + int(pspan * 0.85 + p * pspan * 0.14 / 100), m), abort)

    tail = ""
    if getattr(cfg, "debug_mode", False):
        path = LOGS / ("ocr-" + uuid.uuid4().hex[:12] + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"route": route, "calls": budget[0], "dropped": dropped,
                                    "diagnostics": diagnostics}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        tail = f" · 诊断 {path.name}"
    events.sort(key=lambda e: (e.start, e.y, e.x))
    fixed = stabilize_tracks(events, lock_all=bool(getattr(cfg, "subtitle_lock_position", True)))
    tail += f" · 位置固定 {fixed} 条" if fixed else ""
    flagged = flag_growing_tracks(events)
    tail += f" · 框漂移待复核 {flagged} 条" if flagged else ""
    progress(pbase + pspan, f"读字完成: {len(events)} 条 · 后置过滤 {len(dropped)} 条{tail}")
    return events, snapmap, budget[0]


def _richness(ev):
    q = np.asarray(ev.geom, np.float32)
    try:
        area = abs(cv2.contourArea(q))
    except cv2.error:
        area = 0.0
    return (len(_norm_text(ev.text)), area, ev.end - ev.start)


def _postfilter_events(events, segs, cfg, meta, progress, abort):
    from core.pipeline_quality import is_symbol_only, llm_filter_events
    seg_by_id = {s.id: s for s in segs}
    dropped = []

    def drop(ev, why):
        dropped.append({"t0": round(ev.start, 2), "t1": round(ev.end, 2), "text": ev.text, "why": why})
        seg = seg_by_id.get(ev.source_seg_id)
        if seg is not None and not seg.verdict:
            seg.verdict = why

    if bool(getattr(cfg, "filter_symbol_only", True)):
        for ev in list(events):
            if is_symbol_only(ev.text):
                drop(ev, "纯符号/标点, 非台词")
                events.remove(ev)

    events.sort(key=lambda e: (e.start, e.y, e.x))
    keep = []
    for ev in events:
        dup = None
        for k in keep[-40:]:
            ov = min(k.end, ev.end) - max(k.start, ev.start)
            if ov <= 0:
                continue
            shorter = max(1e-3, min(k.end - k.start, ev.end - ev.start))
            if ov / shorter < 0.3:
                continue
            contain = qcontain(k.geom, ev.geom)
            if contain < 0.6:
                continue
            na, nb = _norm_text(k.text), _norm_text(ev.text)
            if not na or not nb:
                continue
            if _sim(k.text, ev.text) >= 0.6 or na in nb or nb in na:
                dup = k
                break
            if (min(len(na), len(nb)) >= 2 and contain >= 0.75
                    and _area_ratio(k.geom, ev.geom) < 0.75 and _part_sim(na, nb) >= 0.7):
                dup = k
                break
        if dup is None:
            keep.append(ev)
            continue
        winner, loser = (ev, dup) if _richness(ev) > _richness(dup) else (dup, ev)
        winner.start = min(winner.start, loser.start)
        winner.end = max(winner.end, loser.end)
        winner.conf = max(winner.conf, loser.conf)
        if winner is ev:
            keep[keep.index(dup)] = ev
        drop(loser, f"重复候选: 已并入 [{winner.text[:12]}]")
    events[:] = keep

    if bool(getattr(cfg, "filter_llm", True)) and events:
        try:
            bad = llm_filter_events(events, cfg, meta, progress, abort)
        except RuntimeError:
            raise
        except Exception as exc:
            log.warning("LLM 语义过滤跳过: %s", exc)
            bad = []
        for ev in bad:
            if ev in events:
                drop(ev, "AI 判定非字幕(效果音/水印/UI/招牌)")
                events.remove(ev)

    events.sort(key=lambda e: (e.start, e.y, e.x))
    for i, ev in enumerate(events):
        for k in events[max(0, i - 12):i]:
            if k.end <= ev.start + 1e-6 or qiou(k.geom, ev.geom) < 0.5 or _sim(k.text, ev.text) >= 0.6:
                continue
            if ev.start - k.start >= 0.1:
                k.end = round(ev.start, 3)
    events[:] = [e for e in events if e.end - e.start >= 0.05]
    return dropped


def run_auto_pipeline(video, cfg, progress, abort, partial=None, t0=0.0, t1=None, locate_partial=None, progress_read=None):
    if progress_read is None:                      # 兼容: 定位 0-55 / 识别 55-100 合并成一条
        segs, meta = stage_locate(video, cfg, progress, abort, t0, t1,
                                  partial=locate_partial, pbase=0, pspan=55)
        read_progress, rbase, rspan = progress, 55, 45
    else:                                          # 分环节: 定位/识别各自线性 0~100
        segs, meta = stage_locate(video, cfg, progress, abort, t0, t1,
                                  partial=locate_partial, pbase=0, pspan=100)
        read_progress, rbase, rspan = progress_read, 0, 100
    check_abort(abort)
    events, snapmap, calls = stage_read(
        video, segs, cfg, read_progress, abort,
        partial=(lambda evs: partial(-1.0, evs)) if partial else None, pbase=rbase, pspan=rspan)
    report = {"vlm_read": calls,
              "dropped": [{"t0": s.start, "text": s.text, "why": s.verdict} for s in segs if s.verdict]}
    if partial:
        partial(-1.0, copy.deepcopy(events))
    return events, meta, segs, report
