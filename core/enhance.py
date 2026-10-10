"""画面取色: 遮罩底色估计 与 字幕文字颜色估计。"""
import cv2
import numpy as np


def _clip_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def _lum(b, g, r):
    return 0.114 * b + 0.587 * g + 0.299 * r


def estimate_bg_color(frame, x1, y1, x2, y2):
    """Otsu多数派像素中值 = 原字幕底色(用于遮罩), 失败时取外圈中值"""
    x1, y1, x2, y2 = _clip_box(frame, x1, y1, x2, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return "#101010"
    region = frame[y1:y2, x1:x2]
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg = otsu > 0
        if bg.sum() < gray.size / 2:
            bg = ~bg
        if bg.sum() >= 8:
            px = region[bg]
            b, g, r = np.median(px, axis=0)
            return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
    except Exception:
        pass
    try:
        p = 8
        ex1, ey1, ex2, ey2 = _clip_box(frame, x1 - p, y1 - p, x2 + p, y2 + p)
        ring = frame[ey1:ey2, ex1:ex2].reshape(-1, 3)
        b, g, r = np.median(ring, axis=0)
        return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))
    except Exception:
        return "#101010"


def estimate_text_color(frame, x1, y1, x2, y2):
    """估计框内文字颜色: Otsu少数派像素中值 = 文字色。
    返回 (文字色hex, 建议描边hex); 对比不足或像素太少时返回 None(调用方退安全色)。"""
    x1, y1, x2, y2 = _clip_box(frame, x1, y1, x2, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    region = frame[y1:y2, x1:x2]
    try:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = otsu > 0
        if fg.sum() > gray.size / 2:
            fg = ~fg
        n = int(fg.sum())
        if n < 24 or n > gray.size * 0.6:
            return None
        tb, tg, tr = np.median(region[fg], axis=0)
        bb, bg_, br = np.median(region[~fg], axis=0)
        lt, lb = _lum(tb, tg, tr), _lum(bb, bg_, br)
        if abs(lt - lb) < 40:            # 前后景对比太弱, 取色不可信
            return None
        color = "#{:02X}{:02X}{:02X}".format(int(tr), int(tg), int(tb))
        outline = "#101010" if lt >= 140 else "#F4F4F4"
        return color, outline
    except Exception:
        return None
