# -*- coding: utf-8 -*-
"""旋转框工具: 四点框的生成、长轴方向、竖排判定、面积。"""
import math

import numpy as np


def quad_from_rect(cx, cy, w, h, angle):
    """中心+宽高+角度 → 四点框(顺时针)"""
    rad = math.radians(float(angle))
    ca, sa = math.cos(rad), math.sin(rad)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2),
                   (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append([float(cx + dx * ca - dy * sa),
                    float(cy + dx * sa + dy * ca)])
    return pts


def quad_axis(q):
    """四点框自身的长轴 → (长轴与水平线夹角(-90,90], 长边, 短边)。
    用框自己的边判断方向, 而不是外接矩形: 斜排的横向字幕不会被误判为竖排。"""
    p = np.asarray(q, dtype=float).reshape(4, 2)
    a = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2.0
    b = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2.0
    if a >= b:
        v = (p[1] - p[0]) + (p[2] - p[3])
        long_side, short_side = a, b
    else:
        v = (p[3] - p[0]) + (p[2] - p[1])
        long_side, short_side = b, a
    angle = math.degrees(math.atan2(float(v[1]), float(v[0])))
    angle = (angle + 90.0) % 180.0 - 90.0
    return angle, max(1.0, float(long_side)), max(1.0, float(short_side))


def quad_is_vertical(q, ratio=1.6, min_angle=60.0):
    """竖排 = 长轴明显长于短轴, 且长轴更接近竖直(与水平夹角 > min_angle)。
    60° 以内的斜排文字一律按(带旋转的)横排处理。"""
    try:
        angle, long_side, short_side = quad_axis(q)
    except (ValueError, TypeError):
        return False
    return long_side > short_side * ratio and abs(angle) > min_angle


def quad_area(q):
    p = np.asarray(q, dtype=float).reshape(4, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
