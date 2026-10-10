import math
import re
import statistics
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Tuple

FIT_MODES = ("contain", "stretch", "manual")
_BREAKS = re.compile(r"\\N|\r?\n")


def _nid():
    return uuid.uuid4().hex[:8]


def rect_quad(rect):
    x1, y1, x2, y2 = [float(v) for v in rect]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def quad_center(q):
    return (sum(float(p[0]) for p in q) / 4.0, sum(float(p[1]) for p in q) / 4.0)


def quad_bbox(q):
    xs = [float(p[0]) for p in q]
    ys = [float(p[1]) for p in q]
    return (min(xs), min(ys), max(xs), max(ys))


def quad_angle(q):
    return math.degrees(math.atan2(float(q[1][1]) - float(q[0][1]), float(q[1][0]) - float(q[0][0])))


@dataclass
class SubtitleEvent:
    start: float
    end: float
    text: str
    translated: str = ""

    x: float = 0.0
    y: float = 0.0
    font_size: int = 40                 # 基准字号: 有定位框时只作字形比例的基准, 实际字号由框高自适应
    color: str = "#FFFFFF"
    rotation: float = 0.0
    font: str = ""
    bold: bool = True
    outline_color: str = "#000000"
    outline_width: float = 0.0
    vertical: bool = False

    conf: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None

    # 遮罩: 用户手动改过(mask_custom)时完全听 mask_on; 否则跟随 处理方式+04勾选 自动决定
    mask_on: bool = False
    mask: Optional[List[float]] = None
    mask_color: str = "#101010"

    # geom = 定位框 = 字幕本体: 译文位置/字号基准、擦除范围、自动遮罩都由它推出; 移动框 = 移动字幕
    geom: Optional[List[List[float]]] = None
    geom_anchored: bool = False
    geom_track: list = field(default_factory=list)
    fit_mode: str = "manual"            # contain: 字号按框高自适应 / stretch: 宽高都贴合框 / manual: 自由字号
    mask_custom: bool = False
    review_reason: str = ""
    sample_image: str = ""
    sample_time: float = 0.0

    source: str = ""
    source_seg_id: str = ""
    locator_score: float = 0.0

    id: str = field(default_factory=_nid)

    def display_text(self) -> str:
        t = self.translated.strip() or self.text.strip()
        if not t:
            return ""
        rows = [" ".join(row.split()) for row in _BREAKS.split(t)]
        return "\n".join(row for row in rows if row)

    def has_text(self) -> bool:
        return bool(self.text.strip() or self.translated.strip())

    def needs_review(self) -> bool:
        return bool(self.review_reason) or self.conf < 0.55

    def outline_w(self) -> float:
        if self.outline_width > 0:
            return self.outline_width
        return max(1.0, self.font_size / 16.0)

    # ---------- 定位框 ----------
    def ensure_geom(self):
        """没有四点框时由 bbox / mask 还原; 都没有返回 None(自由文字)。"""
        if self.geom is not None and len(self.geom) == 4:
            return self.geom
        src = self.bbox if (self.bbox and len(self.bbox) == 4) else self.mask
        if src and len(src) == 4:
            self.geom = rect_quad(src)
            return self.geom
        return None

    def set_quad(self, quad):
        """手动调整定位框(字幕框 = 译文位置/字号基准 = 擦除与遮罩范围):
        四点框成为唯一几何(逐帧轨迹清掉), bbox 同步; 自动遮罩随框, 用户改过的遮罩(mask_custom)不动;
        译文随框心平移——自动排布(contain/stretch)会按新框重新套入, 手动字号只跟随平移。"""
        pts = [[float(p[0]), float(p[1])] for p in quad]
        if len(pts) != 4:
            raise ValueError("四点框需要 4 个点")
        cx, cy = quad_center(pts)
        old = self.ensure_geom()
        if old is not None:
            ocx, ocy = quad_center(old)
            self.x += cx - ocx
            self.y += cy - ocy
        else:
            self.x, self.y = cx, cy
        self.geom = pts
        self.geom_anchored = True
        self.geom_track = []
        self.bbox = quad_bbox(pts)
        # 框的旋转即文字的旋转: 旋转定位框后, 译文随框同步旋转(竖排按竖排处理, 不写 rotation)
        self.rotation = 0.0 if self.vertical else float(
            math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])))
        if not self.mask_custom:
            self.mask = list(self.bbox)
        return self

    def ensure_mask(self):
        if self.mask and len(self.mask) == 4:
            return
        q = self.ensure_geom()
        if q is not None:
            x1, y1, x2, y2 = quad_bbox(q)
            p = max(2.0, (y2 - y1) * 0.12)
            self.mask = [x1 - p, y1 - p, x2 + p, y2 + p]
            return
        n = max(1, len(self.display_text().replace("\n", "")))
        if self.vertical:
            w = self.font_size * 1.35
            h = self.font_size * n * 1.05
        else:
            w = self.font_size * n * 1.05
            h = self.font_size * 1.35
        self.mask = [self.x - w / 2, self.y - h / 2, self.x + w / 2, self.y + h / 2]


@dataclass
class MaskEvent:
    start: float
    end: float
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "#101010"
    id: str = field(default_factory=_nid)

    def rect(self):
        return self.x1, self.y1, self.x2, self.y2


def cover_quad(ev):
    """遮罩的静态四点框——预览、成品视频、ASS 三处共用同一规则:
    用户改过的遮罩 → mask 矩形(轴对齐); 否则定位四点框 geom; 再退 bbox; 最后按字号估一个。"""
    if getattr(ev, "mask_custom", False) and ev.mask and len(ev.mask) == 4:
        return rect_quad(ev.mask)
    q = ev.ensure_geom()
    if q is not None:
        return [[float(p[0]), float(p[1])] for p in q]
    if not (ev.mask and len(ev.mask) == 4):
        ev.ensure_mask()
    return rect_quad(ev.mask)


def event_covered(ev, placement="erase", pipe_erase=True):
    """该字幕的原字区域是否叠加遮罩。手动改过的听 mask_on; 自动的只在[遮罩覆盖]+04 勾选时对有文字的生效。"""
    if getattr(ev, "mask_custom", False):
        return bool(ev.mask_on)
    return (placement == "cover" and bool(pipe_erase)
            and bool(ev.text.strip() or ev.translated.strip()))


def prepare_cover_masks(events, placement="cover", pipe_erase=True):
    """[04 原字处理] 遮罩覆盖方式: 给导出时会被盖住的字幕补齐遮罩几何, 不改任何开关。返回会被盖住的条数。"""
    n = 0
    for ev in events:
        if not event_covered(ev, placement, pipe_erase):
            continue
        if not (ev.mask and len(ev.mask) == 4):
            ev.ensure_mask()
        n += 1
    return n


def _build(cls, data):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict(data).items() if k in names})


def stabilize_tracks(events, max_jitter_px=6.0, lock_all=False):
    """把几何轨迹收敛为固定框(取中位框), 消除成品字幕的移位/大小抖动。
    lock_all=True 时**强制锁定所有轨迹**(限定字幕不要移动, 不论幅度)——同一句字幕本就是同一段文字,
    框的变动基本都是检测噪声/特效漂移, 锁定后位置大小完全一致; 关掉则只锁抖动≤max_jitter_px 的,
    保留真正在动的字幕的轨迹。返回固定的条数。"""
    n = 0
    for ev in events:
        track = getattr(ev, "geom_track", None) or []
        pts = [s[1] for s in track if len(s) >= 2 and s[1] is not None and len(s[1]) == 4]
        if len(pts) < 3:
            continue
        jitter = max(max(p[k][i] for p in pts) - min(p[k][i] for p in pts)
                     for k in range(4) for i in (0, 1))
        if not lock_all and jitter > max_jitter_px:
            continue
        ev.geom = [[float(statistics.median(p[k][i] for p in pts)) for i in (0, 1)] for k in range(4)]
        ev.geom_anchored = True
        ev.geom_track = []
        n += 1
    return n


def flag_growing_tracks(events, growth_ratio=0.25):
    """轨迹框尺寸随时间变化过大的字幕标为待复核(让人工介入核对定位框):
    ①渐进变大(常见于 RGB 分色/特效字幕导致的定位漂移) ②合并/跳变导致尺寸突变。
    返回标记的条数。"""
    n = 0
    for ev in events:
        track = getattr(ev, "geom_track", None) or []
        dims = []
        for s in track:
            if len(s) >= 2 and s[1] is not None and len(s[1]) == 4:
                p = s[1]
                w = math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1])
                h = math.hypot(p[3][0] - p[0][0], p[3][1] - p[0][1])
                dims.append(float(w) if ev.vertical else float(h))
        if len(dims) < 4:
            continue
        med = statistics.median(dims)
        if med <= 0:
            continue
        third = max(1, len(dims) // 3)
        first = statistics.median(dims[:third])
        last = statistics.median(dims[-third:])
        grew = last > first * (1 + growth_ratio)              # 渐进变大
        span = (max(dims) - min(dims)) > med * growth_ratio   # 尺寸跳变/忽大忽小
        if grew or span:
            ev.review_reason = "字幕框大小随时间变化明显(疑似特效字幕定位漂移), 请核对定位框"
            n += 1
    return n


def clone_state(events, masks):
    return {"e": [asdict(e) for e in events], "m": [asdict(m) for m in masks]}


def restore_state(state):
    evs = [_build(SubtitleEvent, d) for d in state.get("e", [])]
    mks = [_build(MaskEvent, d) for d in state.get("m", [])]
    return evs, mks
