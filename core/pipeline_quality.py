"""质量层: 候选核验(VLM) / 语义过滤(LLM) / 纯符号过滤 / 多行投影切分 / 拼图。"""
import json
import logging
import re
import unicodedata

import cv2
import numpy as np

from core import ai
from core.frames import decode_jpg
from core.ocr_protocols import is_glm_ocr
from core.prompts import get_prompt, render
from core.v6_common import check_abort

log = logging.getLogger("suisen.quality")

_NORM_RE = re.compile(r"[\s\W_]+", re.U)


def norm_text(text):
    return _NORM_RE.sub("", str(text or "")).lower()


def is_symbol_only(text):
    """没有任何字母/汉字/假名/数字 → 纯符号(♥ … !!) 不是台词"""
    for ch in str(text or ""):
        cat = unicodedata.category(ch)
        if cat in ("Lo", "Ll", "Lu", "Nd"):
            return False
    return True


def split_text_lines(crop, vertical=False, max_lines=4):
    """按墨迹投影把'多行合成一框'的裁剪图切成逐行子图(横排切行, 竖排切列并右→左)。
    切不出 2~max_lines 条可信条带时原样返回。"""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hp = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3.0))
    _, mask = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = (mask > 0).sum(axis=0 if vertical else 1).astype(np.float32)
    length = len(profile)
    if length < 16 or not np.any(profile > 0):
        return [crop]
    win = max(3, int(round(length * 0.03)))
    smooth = np.convolve(profile, np.ones(win) / win, mode="same")
    positive = smooth[smooth > 0]
    thr = max(1.0, float(np.percentile(positive, 40)) * 0.3)
    active = smooth >= thr
    idx = np.flatnonzero(active)
    max_gap = max(2, int(length * 0.02))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            active[a:b + 1] = True
    runs, i = [], 0
    while i < length:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < length and active[j]:
            j += 1
        runs.append((i, j))
        i = j
    min_span = max(6, int(length * 0.12))
    bands = [(a, b) for a, b in runs if b - a >= min_span]
    if len(bands) < 2 or len(bands) > max_lines:
        return [crop]
    tallest = max(b - a for a, b in bands)
    if any((b - a) < 0.4 * tallest for a, b in bands):
        return [crop]
    out = []
    for a, b in bands:
        a0, b0 = max(0, a - 2), min(length, b + 2)
        out.append(crop[:, a0:b0] if vertical else crop[a0:b0, :])
    if vertical:
        out.reverse()
    return out


def split_chars(column):
    """竖排单列 → 逐字子图(上→下)。"""
    h, w = column.shape[:2]
    if h < 8 or w < 6:
        return []
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    hp = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3.0))
    _, mask = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = (mask > 0).sum(axis=1).astype(np.float32)
    positive = profile[profile > 0]
    if positive.size == 0:
        return []
    active = profile >= max(1.0, float(np.percentile(positive, 40)) * 0.25)
    runs, i = [], 0
    while i < h:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < h and active[j]:
            j += 1
        runs.append([i, j])
        i = j
    merged = []
    for a, b in runs:                      # 碎片合并: 合并后高度 ≤ 1.15 列宽 且 间隙 ≤ 0.35 列宽
        if merged and b - merged[-1][0] <= 1.15 * w and a - merged[-1][1] <= 0.35 * w:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a < max(3, 0.08 * w):       # 只丢明显噪点; '一'这种扁字要保留
            continue
        out.append(column[max(0, a - 2):min(h, b + 2), :])
    return out


def stitch_chars(chars, height=48, gap=6):
    """单字子图横向拼成一行(扁字先补成方块再等高缩放), 供单行识别模型一次读完。"""
    if not chars:
        return None
    tiles = []
    for c in chars:
        h, w = c.shape[:2]
        if h < w:
            top = (w - h) // 2
            c = cv2.copyMakeBorder(c, top, w - h - top, 0, 0, cv2.BORDER_REPLICATE)
            h = w
        nw = max(4, int(round(w * height / max(1, h))))
        tile = cv2.resize(c, (nw, height), interpolation=cv2.INTER_CUBIC)
        tiles.append(cv2.copyMakeBorder(tile, 0, 0, gap // 2, gap // 2, cv2.BORDER_REPLICATE))
    return cv2.hconcat(tiles)


def _contact_sheet(seg, max_frames=3):
    """把候选段的多张裁剪快照(已含上下文边距)横向拼成一张图给 VLM 判别"""
    snaps = [s for s in (getattr(seg, "snaps", None) or [])
             if isinstance(s, (list, tuple)) and len(s) >= 2]
    if not snaps:
        return None
    n = min(max_frames, len(snaps))
    idxs = sorted(set(np.linspace(0, len(snaps) - 1, n).round().astype(int).tolist()))
    tiles = []
    for j, idx in enumerate(idxs):
        crop = decode_jpg(snaps[idx][1])
        if crop is None or crop.size == 0:
            continue
        scale = min(1.0, 420.0 / max(1, crop.shape[1]))
        if crop.shape[0] * scale < 48:
            scale = min(2.5, 48.0 / max(1, crop.shape[0]))
        if abs(scale - 1.0) > 1e-3:
            crop = cv2.resize(crop, (max(2, int(crop.shape[1] * scale)),
                                     max(2, int(crop.shape[0] * scale))))
        crop = cv2.copyMakeBorder(crop, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(12, 12, 12))
        cv2.putText(crop, f"frame {j + 1}", (7, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (60, 220, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    if not tiles:
        return None
    target_h = max(t.shape[0] for t in tiles)
    padded = [
        cv2.copyMakeBorder(t, 0, target_h - t.shape[0], 0, 0,
                           cv2.BORDER_CONSTANT, value=(12, 12, 12))
        if t.shape[0] < target_h else t for t in tiles
    ]
    return cv2.hconcat(padded)


def verify_segments_vlm(segs, cfg, progress, abort, meta=None):
    """用通用 VLM 判别候选是否为字幕(台标/招牌/UI → 拒绝)。GLM-OCR 这类纯OCR模型无法判别, 自动跳过。
    方向(横/竖排)只信定位框自身的几何——VLM 常把斜排文字报成竖排, 这里不再覆盖。"""
    if not bool(getattr(cfg, "locator_vlm_verify", True)):
        return 0
    mode = str(getattr(cfg, "locator_verify_mode", "ambiguous"))
    if mode == "off":
        return 0
    ep = cfg.ep("vlm")
    if is_glm_ocr(ep):
        log.info("VLM 为纯OCR模型(GLM-OCR), 不能做字幕/非字幕判别; 跳过核验, 交给读字后的LLM语义过滤")
        return 0
    try:
        vlm = ai.VLMClient(ep, cfg.vlm_detail_width, abort=abort,
                           timeout=getattr(cfg, "request_timeout", 300))
    except Exception as exc:
        log.warning("VLM 核验不可用: %s", exc)      # 模型未下载也会走到这里: 不静默下载, 直接跳过核验
        return 0
    w = float((meta or {}).get("w", 0) or 0)
    h = float((meta or {}).get("h", 0) or 0)
    candidates = []
    for seg in segs:
        if seg.verdict:
            continue
        need = mode == "all"
        if mode == "ambiguous":
            duration = seg.end - seg.start
            central = False
            if w and h:
                x1, y1, x2, y2 = seg.bbox
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                central = (0.25 < cx < 0.75) if seg.vertical else (0.12 < cy < 0.72)
            need = central or duration > 20.0 or duration < 0.6 \
                or seg.score < 0.7 or seg.hits <= 2
        if need:
            candidates.append(seg)
    budget = max(0, int(getattr(cfg, "locator_verify_budget", 100)))
    candidates = candidates[:budget]
    threshold = float(getattr(cfg, "locator_verify_threshold", 0.82))
    used = 0
    for i, seg in enumerate(candidates):
        check_abort(abort)
        sheet = _contact_sheet(seg)
        if sheet is None:
            continue
        try:
            raw = vlm.ask(sheet, get_prompt(cfg, "locator_verify"), max_tokens=120)
            data = ai.extract_json(raw)
        except Exception as exc:
            log.debug("核验调用失败 %s: %s", seg.id, exc)
            continue
        if not isinstance(data, dict):
            continue
        used += 1
        is_sub = ai.coerce_bool(data.get("subtitle"))
        conf = ai.coerce_num(data.get("confidence"), 0.5) or 0.5
        if conf > 1.0:
            conf /= 100.0
        conf = float(np.clip(conf, 0.0, 1.0))
        reason = str(data.get("reason") or "").strip()
        seg.verify_conf, seg.verify_reason = conf, reason
        if is_sub:
            seg.verified = True
        elif conf >= threshold:
            seg.verdict = "VLM判非字幕" + (f": {reason}" if reason else "")
        progress(int((i + 1) * 100 / max(1, len(candidates))),
                 f"候选字幕核验 {i + 1}/{len(candidates)}")
    return used


def llm_filter_events(events, cfg, meta, progress, abort):
    """用文本 LLM(翻译端点)对读出的文本做语义过滤: 效果音/拟声词特效字/水印/UI/招牌 → 丢弃。
    返回需要丢弃的事件列表; 任何解析失败都'宁可保留', 不会误删。"""
    if not events or not bool(getattr(cfg, "filter_llm", True)):
        return []
    try:
        client = ai.TextClient(cfg.ep("translate"), abort=abort,
                               timeout=getattr(cfg, "request_timeout", 300))
    except Exception as exc:
        log.warning("LLM 语义过滤不可用(翻译端点): %s", exc)
        return []
    w = float(meta.get("w", 1) or 1)
    h = float(meta.get("h", 1) or 1)
    tpl = get_prompt(cfg, "subtitle_filter")
    system = "你是视频字幕清洗器。只输出JSON数组，不要解释。"
    batch = max(5, int(getattr(cfg, "filter_llm_batch", 40)))
    dropped = []
    for i in range(0, len(events), batch):
        check_abort(abort)
        chunk = events[i:i + batch]
        payload = [{
            "t": ev.text, "sec": round(ev.end - ev.start, 2),
            "y": round(ev.y / h, 2), "x": round(ev.x / w, 2),
            "hr": round(float(ev.font_size) / h, 3),
            "c": round(float(ev.conf), 2), "src": ev.source,
        } for ev in chunk]
        user = render(tpl, payload=json.dumps(payload, ensure_ascii=False))
        try:
            raw = client.chat(system, user, temperature=0.0,
                              max_tokens=max(256, 12 * len(chunk)))
            arr = ai.extract_json(raw)
        except Exception as exc:
            log.warning("LLM过滤本批失败, 全部保留: %s", exc)
            continue
        if not isinstance(arr, list) or len(arr) != len(chunk):
            log.warning("LLM过滤返回长度不符(%s/%s), 本批保留", len(arr) if isinstance(arr, list) else "?", len(chunk))
            continue
        for ev, keep in zip(chunk, arr):
            if not ai.coerce_bool(keep):
                dropped.append(ev)
        progress(int(min(len(events), i + batch) * 100 / len(events)),
                 f"语义过滤 {min(len(events), i + batch)}/{len(events)} · 剔除{len(dropped)}")
    return dropped
