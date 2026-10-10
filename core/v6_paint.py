"""文字排布与栅格化——预览 / 成品 MP4 / ASS 的唯一真相。
规则: 字号只由定位框的高(竖排: 宽)决定——文字连描边刚好占满一行, 不越界、不缩小;
宽度不受限, 按 align 在框内靠左/居中/靠右(竖排: 靠上/居中/靠下)。"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetricsF, QImage, QPainter,
                           QPainterPath, QPainterPathStroker, QPolygonF, QTransform)

from core.models import cover_quad, event_covered
from core.v6_common import qangle, qbox, qsize, source_quad

BILINGUAL_SCALE = 0.55      # 双语: 原文行相对译文的字号比例(画面合成与 ASS 一致)
ALIGNS = ("left", "center", "right")
FILL = 0.98                 # 文字(含描边)占框高的比例, 留 2% 防越界
BELOW_GAP = 8.0


@lru_cache(maxsize=256)
def _font(family, size, bold):
    font = QFont()
    font.setFamilies([family, "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC"])
    font.setPixelSize(max(8, int(round(size))))
    font.setBold(bool(bold))
    return font


@lru_cache(maxsize=2048)
def _glyphs(rows, family, size, bold, vertical, outline, align="center"):
    """rows=((文字, 字号比例), ...)。横排: 逐行按 align 对齐叠放; 竖排: 每行当一列, 右→左, 列内按 align 对齐(上/中/下)。
    返回 (填充路径, 描边∪填充路径, 每行/列在局部坐标的中心)。"""
    path = QPainterPath()
    centers = []
    if vertical:
        cols = []
        for text, scale in rows:
            font = _font(family, size * scale, bold)
            m = QFontMetricsF(font)
            chars = [c for c in text if not c.isspace()] or [" "]
            cols.append((font, m, chars, scale, m.height() * len(chars)))
        total = max(c[4] for c in cols)
        x = 0.0
        for font, m, chars, scale, length in cols:
            step = m.height()
            y0 = {"left": 0.0, "right": total - length}.get(align, (total - length) / 2.0)
            for i, ch in enumerate(chars):
                path.addText(x, y0 + i * step + m.ascent(), font, ch)
            centers.append((x + size * scale / 2.0, y0 + length / 2.0))
            x -= step
    else:
        metrics = [(QFontMetricsF(_font(family, size * s, bold)), _font(family, size * s, bold), text)
                   for text, s in rows]
        width = max(m.horizontalAdvance(text) for m, _, text in metrics)
        y = 0.0
        for m, font, text in metrics:
            adv = m.horizontalAdvance(text)
            x = {"left": 0.0, "right": width - adv}.get(align, (width - adv) / 2.0)
            path.addText(x, y + m.ascent(), font, text)
            centers.append((x + adv / 2.0, y + m.height() / 2.0))
            y += m.height()
    stroker = QPainterPathStroker()
    stroker.setWidth(max(0.0, float(outline)) * 2)
    stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    edge = stroker.createStroke(path).united(path)
    return path, edge, tuple(centers)


def _rows_for(ev, bilingual):
    text = ev.display_text()
    rows = [(r, 1.0) for r in text.splitlines() if r] or [(" ", 1.0)]
    if bilingual and not ev.vertical and ev.translated.strip() and ev.text.strip():
        rows.append((" ".join(ev.text.split()), BILINGUAL_SCALE))
    return tuple(rows)


def _layout(ev, default_font, t, placement, bilingual, align):
    """字幕排布的唯一实现: 画面合成(event_paths)与 ASS(ass_layout)都从这里取结果。
    有定位框且 fit_mode 为 contain/stretch → 字号由框高(竖排: 框宽)决定, 宽度按 align 对齐;
    否则(自由文字 / manual)按 ev.x, ev.y, font_size, rotation 自由摆放。"""
    family = ev.font.strip() or default_font
    align = align if align in ALIGNS else "center"
    size, bold, vertical, outline = ev.font_size, ev.bold, ev.vertical, ev.outline_w()
    rows = _rows_for(ev, bilingual)
    fill, edge, centers = _glyphs(rows, family, size, bold, vertical, outline, align)
    bounds = edge.boundingRect()
    if bounds.isEmpty():
        return None
    common = {"rows": rows, "centers": centers, "size": size, "family": family, "bold": bold,
              "vertical": vertical, "align": align, "rotation": float(ev.rotation)}
    q = source_quad(ev, t)
    reference = source_quad(ev, None)
    mode = getattr(ev, "fit_mode", "manual")
    if q is None or reference is None or mode not in ("contain", "stretch"):
        tf = QTransform()
        tf.translate(ev.x, ev.y)
        tf.rotate(ev.rotation)
        tf.translate(-bounds.center().x(), -bounds.center().y())
        return {**common, "fill": tf.map(fill), "edge": tf.map(edge), "transform": tf,
                "scale": (1.0, 1.0), "anchor": bounds.center(), "an": 5}

    width, height = qsize(q)
    q = np.asarray(q, float)
    center = q.mean(axis=0)
    ref_center = np.asarray(reference, float).mean(axis=0)
    rad = math.radians(ev.rotation - qangle(reference))
    matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    q = (q - center) @ matrix.T + center
    q += np.array([ev.x, ev.y]) - ref_center            # 用户拖动译文 = 相对框的偏移
    if placement == "below":
        q[:, 1] += height + BELOW_GAP
    if vertical:
        s = width * FILL / max(1e-6, bounds.width())
        sx, sy = s, (height * FILL / max(1e-6, bounds.height()) if mode == "stretch" else s)
        ax_l, ax_b = bounds.center().x(), width / 2.0
        ay_l = {"left": bounds.top(), "right": bounds.bottom()}.get(align, bounds.center().y())
        ay_b = {"left": 0.0, "right": height}.get(align, height / 2.0)
        an = 5
    else:
        s = height * FILL / max(1e-6, bounds.height())
        sx, sy = (width * FILL / max(1e-6, bounds.width()) if mode == "stretch" else s), s
        ay_l, ay_b = bounds.center().y(), height / 2.0
        ax_l = {"left": bounds.left(), "right": bounds.right()}.get(align, bounds.center().x())
        ax_b = {"left": 0.0, "right": width}.get(align, width / 2.0)
        an = {"left": 4, "right": 6}.get(align, 5)
    local = QTransform()
    local.translate(ax_b, ay_b)
    local.scale(sx, sy)
    local.translate(-ax_l, -ay_l)
    src = QPolygonF([QPointF(0, 0), QPointF(width, 0), QPointF(width, height), QPointF(0, height)])
    dst = QPolygonF([QPointF(float(x), float(y)) for x, y in q])
    perspective = QTransform()
    if not QTransform.quadToQuad(src, dst, perspective):
        raise ValueError("字幕四点框无法建立投影变换")
    tf = local * perspective
    return {**common, "fill": tf.map(fill), "edge": tf.map(edge), "transform": tf,
            "scale": (sx, sy), "anchor": QPointF(ax_l, ay_l), "an": an}


def event_paths(ev, default_font, t=None, placement="erase", bilingual=False, align="center"):
    lay = _layout(ev, default_font, t, placement, bilingual, align)
    if lay is None:
        return QPainterPath(), QPainterPath()
    return lay["fill"], lay["edge"]


def ass_layout(ev, default_font, placement="erase", bilingual=False, align="center"):
    """给 ASS 生成器: 与画面合成完全相同的排布(有效字号、拉伸、对齐锚点、竖排每列中心)。
    返回 None 表示该条无可见文字。"""
    lay = _layout(ev, default_font, None, placement, bilingual, align)
    if lay is None:
        return None
    sx, sy = lay["scale"]
    metrics = QFontMetricsF(_font(lay["family"], lay["size"], lay["bold"]))
    fs = max(1, int(round(metrics.height() * sy)))          # libass \fs 是格高(ascent+descent)
    tf = lay["transform"]
    items = []
    for (cx, cy), (text, scale) in zip(lay["centers"], lay["rows"]):
        p = tf.map(QPointF(cx, cy))
        items.append((p.x(), p.y(), text, scale))
    a = tf.map(lay["anchor"])
    return {"fs": fs, "scale": (sx, sy), "rotation": lay["rotation"], "vertical": lay["vertical"],
            "items": items, "center": (a.x(), a.y()), "an": lay["an"]}


def ass_font_ratio(family, size=40, bold=True):
    """导入 ASS 时把 \\fs(格高) 换回 Qt pixelSize 的比例。"""
    try:
        return max(0.5, QFontMetricsF(_font(family, size, bold)).height() / float(size))
    except Exception:
        return 1.0


def fitted_size(ev, default_font, align="center"):
    """当前框下实际使用的像素字号(属性面板显示用)。"""
    lay = _layout(ev, default_font, None, "erase", False, align)
    return float(ev.font_size) if lay is None else float(ev.font_size) * lay["scale"][1]


def prewarm_glyphs(events, default_font, placement="erase", bilingual=False, align="center"):
    """在 GUI 线程预先生成字形路径填满缓存: 工作线程合成时只读缓存。"""
    for ev in events:
        try:
            _layout(ev, default_font, None, placement, bilingual, align)
        except Exception:
            pass


# ---------------- 遮罩样式 ----------------

def _hex_bgr(hex_color):
    h = str(hex_color or "#101010").lstrip("#")
    if len(h) != 6:
        h = "101010"
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _cover_regions(events, masks, t, cfg):
    regions = []
    for m in masks or []:
        if m.start <= t < m.end:
            regions.append((np.float32([[m.x1, m.y1], [m.x2, m.y1], [m.x2, m.y2], [m.x1, m.y2]]), m.color))
    placement = str(getattr(cfg, "placement", "erase"))
    pipe_erase = bool(getattr(cfg, "pipe_erase", True))
    for ev in events:
        if not (ev.start <= t < ev.end) or not event_covered(ev, placement, pipe_erase):
            continue
        q = None
        if not getattr(ev, "mask_custom", False) and getattr(ev, "geom_track", None):
            q = source_quad(ev, t)
        if q is None:
            q = cover_quad(ev)
        regions.append((np.asarray(q, np.float32), ev.mask_color))
    return regions


def _cover_patch(roi, regions, style, cfg):
    h, w = roi.shape[:2]
    hole = np.zeros((h, w), np.uint8)
    for q, _ in regions:
        cv2.fillPoly(hole, [np.rint(q).astype(np.int32)], 255)
    if style == "blur":
        strength = int(np.clip(getattr(cfg, "mask_blur", 25), 15, 45))
        blurred = cv2.GaussianBlur(roi, (0, 0), strength / 3.0)
        # 纯模糊对亮字会留残影, 轻微压暗进一步隐藏底字, 嵌字更可读; 不压成硬块, 画面主体附近仍柔和
        blurred = np.clip(blurred.astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
        soft = cv2.GaussianBlur(hole, (0, 0), max(2.0, strength / 4.0))
        alpha = (soft.astype(np.float32) / 255.0)[..., None]
        out = roi.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    base = float(np.clip(getattr(cfg, "mask_bar_alpha", 55), 40, 70)) / 100.0
    alpha_map = np.zeros((h, w), np.float32)
    for q, _ in regions:
        x1, y1, x2, y2 = qbox(q)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        hh = y2 - y1
        fade = max(2, int(hh * 0.28))
        ys = np.arange(hh, dtype=np.float32)
        prof = np.clip(np.minimum(ys / fade, (hh - 1 - ys) / fade), 0.0, 1.0) * base
        block = np.repeat(prof[:, None], x2 - x1, axis=1)
        clip = hole[y1:y2, x1:x2] > 0
        cur = alpha_map[y1:y2, x1:x2]
        alpha_map[y1:y2, x1:x2] = np.where(clip, np.maximum(cur, block), cur)
    alpha_map = cv2.GaussianBlur(alpha_map, (0, 0), 2.0)[..., None]
    return np.clip(roi.astype(np.float32) * (1.0 - alpha_map), 0, 255).astype(np.uint8)


def _bg_ring_color(frame, q):
    """取定位框外圈的背景色(中值): 比框内取色更稳, 即使框几乎被文字占满也能取到真正的底色;
    只读取框外像素, 填充仍严格落在框内。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = qbox(q)
    pad = max(4, int(round((y2 - y1) * 0.35)))
    xa, ya = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
    xb, yb = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
    if xb - xa < 4 or yb - ya < 4:
        return "#101010"
    xi1, yi1 = max(0, int(math.ceil(x1))), max(0, int(math.ceil(y1)))
    xi2, yi2 = min(w, int(math.floor(x2))), min(h, int(math.floor(y2)))
    outer = frame[ya:yb, xa:xb]
    ring = np.ones(outer.shape[:2], bool)
    ring[yi1 - ya:yi2 - ya, xi1 - xa:xi2 - xa] = False
    px = outer[ring]
    if px.size == 0:
        return "#101010"
    b, g, r = np.median(px, axis=0).astype(int)
    return "#%02X%02X%02X" % (int(r), int(g), int(b))


def apply_cover_regions(frame, regions, cfg):
    if not regions:
        return frame
    h, w = frame.shape[:2]
    style = str(getattr(cfg, "mask_style", "bar"))
    out = frame.copy()
    if style == "solid":
        for q, color in regions:
            cv2.fillPoly(out, [np.rint(q).astype(np.int32)], _hex_bgr(color))
        return out
    if style == "bg":
        # 智能底色: 按每个区域外圈的背景色填满, 完全盖住底字; 边缘羽化过渡, 实心而不像硬块
        for q, _color in regions:
            color = _hex_bgr(_bg_ring_color(frame, q))
            quad = np.rint(q).astype(np.int32)
            fill = np.zeros_like(out)
            cv2.fillPoly(fill, [quad], color)
            m = np.zeros(out.shape[:2], np.uint8)
            cv2.fillPoly(m, [quad], 255)
            alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 3.0)[..., None]
            out[:] = np.clip(out.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha,
                             0, 255).astype(np.uint8)
        return out
    margin = 48 if style == "blur" else 8
    boxes = [qbox(q) for q, _ in regions]
    x1 = max(0, int(math.floor(min(b[0] for b in boxes))) - margin)
    y1 = max(0, int(math.floor(min(b[1] for b in boxes))) - margin)
    x2 = min(w, int(math.ceil(max(b[2] for b in boxes))) + margin)
    y2 = min(h, int(math.ceil(max(b[3] for b in boxes))) + margin)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame
    shifted = [(np.asarray(q, np.float32) - np.float32([x1, y1]), c) for q, c in regions]
    out[y1:y2, x1:x2] = _cover_patch(frame[y1:y2, x1:x2], shifted, style, cfg)
    return out


def apply_cover_masks(frame, events, masks, t, cfg):
    return apply_cover_regions(frame, _cover_regions(events, masks, t, cfg), cfg)


def paint_frame(frame, events, masks, t, cfg):
    """导出与预览共用的栅格化: 先叠遮罩, 再嵌字; 嵌字只在字形外接矩形内走 QPainter。"""
    regions = _cover_regions(events, masks, t, cfg)
    active = [ev for ev in events if ev.start <= t < ev.end and ev.display_text()]
    if not regions and not active:
        return frame
    out = apply_cover_regions(frame, regions, cfg) if regions else frame
    if not active:
        return out
    align = str(getattr(cfg, "subtitle_align", "center"))
    h, w = out.shape[:2]
    paths = []
    x1 = y1 = float("inf")
    x2 = y2 = float("-inf")
    for ev in active:
        fill, edge = event_paths(ev, cfg.subtitle_font, t, cfg.placement, cfg.out_bilingual, align)
        b = edge.boundingRect()
        if b.isEmpty():
            continue
        paths.append((ev, fill, edge))
        x1, y1 = min(x1, b.left()), min(y1, b.top())
        x2, y2 = max(x2, b.right()), max(y2, b.bottom())
    if not paths:
        return out
    x1, y1 = max(0, int(math.floor(x1)) - 2), max(0, int(math.floor(y1)) - 2)
    x2, y2 = min(w, int(math.ceil(x2)) + 2), min(h, int(math.ceil(y2)) + 2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return out
    if out is frame:
        out = frame.copy()
    roi = np.ascontiguousarray(cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    image = QImage(roi.data, roi.shape[1], roi.shape[0], roi.strides[0], QImage.Format.Format_RGB888).copy()
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.translate(-x1, -y1)
    for ev, fill, edge in paths:
        painter.fillPath(edge, QColor(ev.outline_color))
        painter.fillPath(fill, QColor(ev.color))
    painter.end()
    data = np.frombuffer(image.bits(), np.uint8, count=image.sizeInBytes()).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 3].reshape(
        image.height(), image.width(), 3)
    out[y1:y2, x1:x2] = cv2.cvtColor(np.ascontiguousarray(data), cv2.COLOR_RGB2BGR)
    return out
