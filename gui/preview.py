import math
import logging

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QFontDatabase, QFontMetricsF,
                           QImage, QPainter, QPainterPath, QPainterPathStroker,
                           QPen, QPixmap, QPolygonF, QTransform)
from PySide6.QtWidgets import (QApplication, QColorDialog, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGraphicsItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView,
                               QInputDialog, QMenu)

from core.geometry import quad_from_rect
from core.models import event_covered, rect_quad
from core.v6_common import qangle, qsize, source_quad, video_meta
from gui.mpv_view import MpvPlayer, MpvViewport

from core.v6_paint import event_paths

QUICK_FONTS = ["Microsoft YaHei", "SimHei", "KaiTi", "SimSun", "FangSong",
               "Source Han Sans SC", "Noto Sans CJK SC", "Arial", "Impact",
               "Comic Sans MS"]
TOOLS = ("select", "subtitle", "mask")
_MIN_BOX = 6.0          # 定位框最小边长(像素)



log = logging.getLogger("suisen.preview")


QUICK_FONTS = ["Microsoft YaHei", "SimHei", "KaiTi", "SimSun", "FangSong",
               "Source Han Sans SC", "Noto Sans CJK SC", "Arial", "Impact",
               "Comic Sans MS"]


TOOLS = ("select", "subtitle", "mask")


_MIN_BOX = 6.0          # 定位框最小边长(像素)


def _zone_cursor(ex, ey, fallback):
    if ex and ey:
        return (Qt.CursorShape.SizeFDiagCursor if ex == ey
                else Qt.CursorShape.SizeBDiagCursor)
    if ex:
        return Qt.CursorShape.SizeHorCursor
    if ey:
        return Qt.CursorShape.SizeVerCursor
    return fallback


def _label_px(text):
    """标签块的设备像素宽度估计(CJK 按 13px, 其他 7.2px)"""
    return 8.0 + sum(13.0 if ord(ch) > 0x2E80 else 7.2 for ch in text)


class GeomItem(QGraphicsItem):
    """只读几何框: 红实线=被剔除的候选(标注原因) / 绿实线=任务实时诊断ok / 红实线=诊断fail"""

    def __init__(self, quad, label="", color="#38b6ff", dashed=True):
        super().__init__()
        self.quad = [(float(p[0]), float(p[1])) for p in quad]
        self.label = label
        self.color = QColor(color)
        self.dashed = dashed
        self.setZValue(8)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def boundingRect(self):
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return QRectF(min(xs) - 6, min(ys) - 24, max(xs) - min(xs) + 160, max(ys) - min(ys) + 32)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.color, 1.6)
        pen.setCosmetic(True)
        if self.dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        path = QPainterPath(QPointF(*self.quad[0]))
        for pt in self.quad[1:]:
            path.lineTo(QPointF(*pt))
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(QBrush(self.color))
        p.drawEllipse(QPointF(*self.quad[0]), 2.5, 2.5)
        if self.label:
            x, y = self.quad[0][0], min(q[1] for q in self.quad) - 6
            p.fillRect(QRectF(x - 2, y - 13, 8 + 7.2 * len(self.label), 16), QColor(10, 12, 18, 170))
            p.setPen(QPen(self.color))
            p.drawText(QPointF(x + 2, y), self.label)


class GeomEditItem(QGraphicsItem):
    """可编辑定位框: kind='s' 候选段(绿实线, 识别前) / kind='e' 字幕(蓝虚线, 识别/翻译后)。
    定位框决定译文的位置/字号基准与擦除/遮罩范围, 任何阶段都能改。
    形状只含边框带与标签块: 框内点击直接落到下层的译文/遮罩项, 不抢它们的手势。
    手势: 拖边=改该边 · 拖角=改两边 · 拖标签=整体移动; 斜排框沿自身轴伸缩, 角度不变。"""

    def __init__(self, kind, obj, view):
        super().__init__()
        self.kind, self.obj, self.view = kind, obj, view
        self._drag = None
        self._rot = None
        self._q = None
        self.setZValue(11 if kind == "e" else 9)      # 压在译文/遮罩之上, 边框带才总能抓到
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._sync()

    # ---- 几何 ----
    def _scale(self):
        return max(0.05, self.view.transform().m11())

    def _margin(self):
        return 7.0 / self._scale()

    def _read_quad(self):
        if self.kind == "s":
            return np.asarray(self.obj.quad(), np.float32)
        return source_quad(self.obj, self.view.cur_time)

    def _color(self):
        if self.isSelected():
            return QColor("#ffd200")
        return QColor("#38b6ff" if self.kind == "e" else "#3fbf6f")

    def _label(self):
        if self.kind == "s":
            return f"候选 {getattr(self.obj, 'score', 0.0):.2f}"
        return ""                       # 字幕定位框不再画文字标签(避免遮挡字幕); 拖动把手仍在(透明, 悬停显示手型光标)

    def _sync(self):
        self.prepareGeometryChange()
        q = self._read_quad()
        if q is None:
            self._q, self._shape, self._bounds = None, QPainterPath(), QRectF()
            return
        self._q = np.asarray(q, np.float32).reshape(4, 2)
        self._center = self._q.mean(axis=0)
        self._w, self._h = qsize(self._q)
        self._angle = qangle(self._q)
        rad = math.radians(self._angle)
        self._cos, self._sin = math.cos(rad), math.sin(rad)
        outline = QPainterPath()
        outline.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in self._q]))
        outline.closeSubpath()
        stroker = QPainterPathStroker()
        stroker.setWidth(2 * self._margin())
        label = QPainterPath()
        label.addRect(self._label_rect())
        self._shape = stroker.createStroke(outline).united(label)
        self._bounds = self._shape.boundingRect().adjusted(-1, -1, 1, 1)

    def _label_rect(self):
        s = self._scale()
        x = float(self._q[0][0])
        top, bottom = float(self._q[:, 1].min()), float(self._q[:, 1].max())
        y = top - 19.0 / s
        if y < 0:
            y = bottom + 3.0 / s                    # 贴着画面顶边的框: 标签放到框下方
        w = _label_px(self._label()) if self._label() else max(20.0, min(self._w, 60.0))
        return QRectF(x - 2.0 / s, y, w / s, 16.0 / s)

    def boundingRect(self):
        return self._bounds

    def shape(self):
        return self._shape

    def paint(self, p, opt, widget=None):
        if self._q is None:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self._color()
        pen = QPen(color, 1.6)
        pen.setCosmetic(True)
        if self.kind == "e":
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(float(self._q[0][0]), float(self._q[0][1])))
        for x, y in self._q[1:]:
            path.lineTo(QPointF(float(x), float(y)))
        path.closeSubpath()
        p.drawPath(path)
        s = self._scale()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        for x, y in self._q:                                   # 四角把手
            p.drawEllipse(QPointF(float(x), float(y)), 3.0 / s, 3.0 / s)
        rect = self._label_rect()
        if self._label():
            p.fillRect(rect, QColor(10, 12, 18, 170))
            font = QFont(p.font())
            font.setPixelSize(max(1, int(round(12.0 / s))))        # 标签字号不随缩放变化
            p.setFont(font)
            p.setPen(QPen(color))
            p.drawText(QPointF(rect.left() + 4.0 / s, rect.top() + 12.5 / s), self._label())

    # ---- 命中 ----
    def _local(self, pos):
        """场景坐标 → 框自身坐标(框心为原点, u 沿 p0→p1 边, v 沿 p0→p3 边)"""
        dx, dy = pos.x() - float(self._center[0]), pos.y() - float(self._center[1])
        return dx * self._cos + dy * self._sin, -dx * self._sin + dy * self._cos

    def _zone_at(self, pos):
        """('label', 0, 0) / ('edge', ex, ey) / None"""
        if self._q is None:
            return None
        if self._label_rect().contains(pos):
            return "label", 0, 0
        lx, ly = self._local(pos)
        m, hw, hh = self._margin(), self._w / 2.0, self._h / 2.0
        if abs(lx) > hw + m or abs(ly) > hh + m:
            return None
        ex = -1 if abs(lx + hw) <= m else (1 if abs(lx - hw) <= m else 0)
        ey = -1 if abs(ly + hh) <= m else (1 if abs(ly - hh) <= m else 0)
        if not ex and not ey:                       # 落在边框带上但框不是严格矩形: 取最近的一条边
            if hw - abs(lx) < hh - abs(ly):
                ex = -1 if lx < 0 else 1
            else:
                ey = -1 if ly < 0 else 1
        return "edge", ex, ey

    def hoverMoveEvent(self, e):
        zone = self._zone_at(e.pos())
        if zone is None:
            self.unsetCursor()
        elif zone[0] == "label":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(_zone_cursor(zone[1], zone[2], Qt.CursorShape.ArrowCursor))
        super().hoverMoveEvent(e)

    def hoverLeaveEvent(self, e):
        self.unsetCursor()
        super().hoverLeaveEvent(e)

    # ---- 手势 ----
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            if self.shape().contains(e.pos()):          # 右键按在边框带上: 进入旋转(框内右键仍穿透给字幕)
                self.view.editBegan.emit()
                self._rot = {"p0": e.scenePos(), "angle0": self._angle,
                             "center": self._center.copy(), "w": self._w, "h": self._h, "moved": False}
                e.accept()
                return
            e.ignore()
            return
        zone = self._zone_at(e.pos()) if e.button() == Qt.MouseButton.LeftButton else None
        if zone is None:
            e.ignore()                                 # 右键等交给下层项(字幕右键=旋转/菜单)
            return
        self.view.editBegan.emit()
        self._drag = {"zone": zone, "p0": e.scenePos(), "center": self._center.copy(),
                      "w": self._w, "h": self._h}
        if self.kind == "e":
            self.scene().clearSelection()
            self.setSelected(True)
            self.view.notify_clicked(self.obj)         # 同步选中表格行
        e.accept()

    def mouseMoveEvent(self, e):
        if self._rot is not None:
            r = self._rot
            center = QPointF(float(r["center"][0]), float(r["center"][1]))
            a = r["p0"] - center
            b = e.scenePos() - center
            if not r["moved"] and (b - a).manhattanLength() < 1e-6 and (e.scenePos() - r["p0"]).manhattanLength() < 6:
                return
            r["moved"] = True
            delta = math.degrees(math.atan2(b.y(), b.x()) - math.atan2(a.y(), a.x()))
            c = r["center"]
            self._apply(quad_from_rect(float(c[0]), float(c[1]), r["w"], r["h"],
                                       r["angle0"] + delta))
            e.accept()
            return
        if self._drag is None:
            e.ignore()
            return
        d = e.scenePos() - self._drag["p0"]
        kind, ex, ey = self._drag["zone"]
        c0, w0, h0 = self._drag["center"], self._drag["w"], self._drag["h"]
        if kind == "label":
            cx, cy, w, h = float(c0[0]) + d.x(), float(c0[1]) + d.y(), w0, h0
        else:
            du = d.x() * self._cos + d.y() * self._sin          # 鼠标位移投到框自身的两条轴
            dv = -d.x() * self._sin + d.y() * self._cos
            w, h, su, sv = w0, h0, 0.0, 0.0
            if ex > 0:
                w = max(_MIN_BOX, w0 + du)
                su = (w - w0) / 2.0
            elif ex < 0:
                w = max(_MIN_BOX, w0 - du)
                su = -(w - w0) / 2.0
            if ey > 0:
                h = max(_MIN_BOX, h0 + dv)
                sv = (h - h0) / 2.0
            elif ey < 0:
                h = max(_MIN_BOX, h0 - dv)
                sv = -(h - h0) / 2.0
            cx = float(c0[0]) + su * self._cos - sv * self._sin  # 对边固定, 框心随之移动一半
            cy = float(c0[1]) + su * self._sin + sv * self._cos
        if self.view.vw and self.view.vh:                        # 框心留在画面内, 不会被拖到够不着的地方
            cx = min(max(cx, 0.0), float(self.view.vw))
            cy = min(max(cy, 0.0), float(self.view.vh))
        self._apply(quad_from_rect(cx, cy, w, h, self._angle))
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._rot is not None and e.button() == Qt.MouseButton.RightButton:
            self._rot = None
            e.accept()
            return
        if self._drag is not None and e.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def _apply(self, quad):
        self.obj.set_quad(quad)
        self._sync()
        if self.kind == "s":
            self.view.segmentChanged.emit(self.obj)
        else:
            self.view.update_event_item(self.obj)         # 译文按新框重新排布/随框心平移
            self.view.update_event_mask(self.obj)         # 遮罩覆盖方式下的自动遮罩随框
            self.view.notify_changed(self.obj)
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.update()
        return super().itemChange(change, value)


class SrtOverlayItem(QGraphicsItem):
    """SRT 格式的所见即所得: SRT 不带位置/字号/颜色/遮罩, 播放器只能按默认样式显示——
    底部居中、白字黑边, 同时出现的多条自下而上叠放, 双语时原文另起一行(与 generate_srt 逐字一致)。只读。"""

    def __init__(self, events, view):
        super().__init__()
        self.setZValue(10)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        w, h = max(1, int(view.vw)), max(1, int(view.vh))
        fs = max(12.0, h * 0.052)
        font = QFont(view.default_font)
        font.setPixelSize(int(round(fs)))
        m = QFontMetricsF(font)
        fill = QPainterPath()
        y = h - h * 0.04
        for ev in sorted(events, key=lambda e: (e.start, e.y)):      # 先出现的在最下面
            text = ev.display_text()
            if view.bilingual and ev.translated.strip() and ev.text.strip():
                text += "\n" + ev.text.strip()
            for line in reversed(text.split("\n")):
                y -= m.height()
                fill.addText((w - m.horizontalAdvance(line)) / 2.0, y + m.ascent(), font, line)
            y -= m.height() * 0.2
        stroker = QPainterPathStroker()
        stroker.setWidth(max(2.0, fs / 14.0) * 2)
        stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._fill = fill
        self._edge = stroker.createStroke(fill).united(fill)
        self._bounds = self._edge.boundingRect().adjusted(-2, -2, 2, 2)

    def boundingRect(self):
        return self._bounds

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillPath(self._edge, QColor("#000000"))
        p.fillPath(self._fill, QColor("#FFFFFF"))


class SubtitleItem(QGraphicsItem):
    """字幕项。有文字时渲染文字路径; 文字为空(手动框选待识别)时渲染可编辑的虚线框。
    手势: 内部左拖=移动 · 边缘左拖=调整长宽 · 滚轮=缩放 · 右拖=旋转。"""

    def __init__(self, ev, view):
        super().__init__()
        self.ev, self.view = ev, view
        self._syncing = False
        self._right = None
        self._moved = False
        self._resize = None
        self._region = None
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(10)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.refresh_from_event()

    def refresh_from_event(self):
        self._syncing = True
        self.prepareGeometryChange()
        fill, edge = event_paths(self.ev, self.view.default_font, self.view.cur_time,
                                 getattr(self.view, "placement", "erase"),
                                 getattr(self.view, "bilingual", False),
                                 getattr(self.view, "align", "center"))
        transform = QTransform.fromTranslate(-self.ev.x, -self.ev.y)
        self._fill = transform.map(fill)
        self._edge = transform.map(edge)
        self._bounds = self._edge.boundingRect().adjusted(-2, -2, 2, 2)
        self._region = None
        if not self.ev.display_text():
            src = None
            if self.ev.mask and len(self.ev.mask) == 4:
                src = list(self.ev.mask)
            elif self.ev.bbox:
                src = list(self.ev.bbox)
            if src:
                x1, y1, x2, y2 = [float(v) for v in src]
                self._region = QRectF(min(x1, x2) - self.ev.x, min(y1, y2) - self.ev.y,
                                      max(2.0, abs(x2 - x1)), max(2.0, abs(y2 - y1)))
                self._bounds = self._region.adjusted(-3, -3, 3, 3)
        self.setPos(self.ev.x, self.ev.y)
        self._syncing = False
        self.update()

    def boundingRect(self):
        return getattr(self, "_bounds", QRectF())

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._region is not None:
            pen = QPen(QColor("#ffd200") if self.isSelected() else QColor("#4f7cff"), 0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(79, 124, 255, 28)))
            painter.drawRect(self._region)
            painter.setPen(QPen(QColor("#9fb4ff")))
            painter.drawText(self._region.adjusted(4, 2, -2, -2),
                             Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "待识别")
            return
        if self.view._composited:
            pen = QPen(QColor("#ffd200") if self.isSelected() else QColor(79, 124, 255, 150),
                       0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())
            return
        painter.fillPath(self._edge, QColor(self.ev.outline_color))
        painter.fillPath(self._fill, QColor(self.ev.color))
        if self.isSelected():
            painter.setPen(QPen(QColor("#5b8cff"), 0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self._syncing:
            nx, ny = self.pos().x(), self.pos().y()
            dx, dy = nx - self.ev.x, ny - self.ev.y
            self.ev.x, self.ev.y = nx, ny
            if self._region is not None and (dx or dy):
                if self.ev.mask and len(self.ev.mask) == 4:
                    m = self.ev.mask
                    self.ev.mask = [m[0] + dx, m[1] + dy, m[2] + dx, m[3] + dy]
                if self.ev.bbox:
                    x1, y1, x2, y2 = self.ev.bbox
                    self.ev.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                self.ev.geom = rect_quad(self.ev.bbox) if self.ev.bbox else None   # 框 = 唯一几何, 随之平移
                self.ev.geom_track = []
            self.view.notify_changed(self.ev)
        return super().itemChange(change, value)

    def _apply_region(self, x1, y1, x2, y2):
        ev = self.ev
        ev.mask = [x1, y1, x2, y2]
        ev.bbox = (x1, y1, x2, y2)
        ev.geom = rect_quad((x1, y1, x2, y2))      # 原来置 None; 现在框就是几何, 识别后字号按它自适应
        ev.geom_track = []
        ev.x, ev.y = (x1 + x2) / 2, (y1 + y2) / 2
        ev.vertical = (y2 - y1) > (x2 - x1)
        unit = (x2 - x1) if ev.vertical else (y2 - y1)
        ev.font_size = max(8, min(300, int(unit * 0.85)))
        self.refresh_from_event()
        self.view.notify_changed(ev)

    def _zone_at(self, pos):
        rect = self._region if self._region is not None else getattr(self, "_bounds", QRectF())
        if rect.isEmpty():
            return 0, 0
        s = max(0.05, self.view.transform().m11())
        m = 9.0 / s
        ex = -1 if abs(pos.x() - rect.left()) <= m else (1 if abs(pos.x() - rect.right()) <= m else 0)
        ey = -1 if abs(pos.y() - rect.top()) <= m else (1 if abs(pos.y() - rect.bottom()) <= m else 0)
        if not rect.adjusted(-m, -m, m, m).contains(pos):
            return 0, 0
        return ex, ey

    def hoverMoveEvent(self, event):
        ex, ey = self._zone_at(event.pos())
        self.setCursor(_zone_cursor(ex, ey, Qt.CursorShape.OpenHandCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._right = (event.scenePos(), self.ev.rotation)
            self._moved = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            zone = self._zone_at(event.pos())
            if zone != (0, 0):
                self.view.editBegan.emit()
                rect = self._region if self._region is not None else self.boundingRect()
                self._resize = {"zone": zone, "p0": event.scenePos(),
                                "rect": QRectF(rect.translated(self.ev.x, self.ev.y)),
                                "fs": self.ev.font_size, "h": max(4.0, rect.height()),
                                "w": max(4.0, rect.width())}
                event.accept()
                return
        self.view.editBegan.emit()
        self.view.notify_clicked(self.ev)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize is not None:
            d = event.scenePos() - self._resize["p0"]
            ex, ey = self._resize["zone"]
            if self._region is not None:
                r0 = self._resize["rect"]
                left, top, right, bottom = r0.left(), r0.top(), r0.right(), r0.bottom()
                if ex < 0:
                    left = min(left + d.x(), right - 8)
                elif ex > 0:
                    right = max(right + d.x(), left + 8)
                if ey < 0:
                    top = min(top + d.y(), bottom - 8)
                elif ey > 0:
                    bottom = max(bottom + d.y(), top + 8)
                self._apply_region(left, top, right, bottom)
            else:
                base = self._resize["h"] if ey else self._resize["w"]
                delta = d.y() * ey if ey else d.x() * ex
                factor = max(0.2, min(5.0, (base + delta) / base))
                self.ev.fit_mode = "manual"
                self.ev.font_size = max(8, min(300, int(round(self._resize["fs"] * factor))))
                self.refresh_from_event()
                self.view.notify_changed(self.ev)
            event.accept()
            return
        if self._right is not None:
            p0, angle0 = self._right
            if not self._moved:
                if (event.scenePos() - p0).manhattanLength() < 6:
                    return
                self._moved = True
                self.view.editBegan.emit()
            center = QPointF(self.ev.x, self.ev.y)
            a = p0 - center
            b = event.scenePos() - center
            delta = math.degrees(math.atan2(b.y(), b.x()) - math.atan2(a.y(), a.x()))
            self.ev.rotation = (angle0 + delta + 180) % 360 - 180
            self.refresh_from_event()
            self.view.notify_changed(self.ev)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize is not None and event.button() == Qt.MouseButton.LeftButton:
            self._resize = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._right:
            if not self._moved:
                self.view.show_item_menu(self.ev, event.screenPos())
            self._right = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.view.editBegan.emit()
        if self._region is not None:
            k = 1.06 if event.delta() > 0 else 1 / 1.06
            src = self.ev.mask if (self.ev.mask and len(self.ev.mask) == 4) else list(self.ev.bbox)
            x1, y1, x2, y2 = [float(v) for v in src]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w = max(8.0, (x2 - x1) * k)
            h = max(8.0, (y2 - y1) * k)
            self._apply_region(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            event.accept()
            return
        self.ev.fit_mode = "manual"
        delta = 2 if event.delta() > 0 else -2
        self.ev.font_size = max(8, min(300, self.ev.font_size + delta))
        self.refresh_from_event()
        self.view.notify_changed(self.ev)
        event.accept()


class MaskItem(QGraphicsRectItem):
    """遮罩块: kind='m' 右键/工具条拖出的遮挡块(任何方式下都按遮罩样式叠加) / kind='e' 字幕自身遮罩。
    手势: 内部拖=移动 · 边缘拖=调整长宽 · 滚轮=缩放(Shift 只调高 / Ctrl 只调宽)。
    拖动字幕遮罩只改遮罩, 不再带着译文一起移动。"""

    def __init__(self, kind, obj, view):
        self.kind, self.obj, self.view = kind, obj, view
        self._rs = None
        x1, y1, x2, y2 = self._rect_of()
        super().__init__(0, 0, max(2.0, x2 - x1), max(2.0, y2 - y1))
        self._syncing = True
        self.setPos(x1, y1)
        self._syncing = False
        self.setZValue(5)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self._restyle()

    def _rect_of(self):
        if self.kind == "m":
            return self.obj.x1, self.obj.y1, self.obj.x2, self.obj.y2
        return tuple(self.obj.mask)

    def sync_from_obj(self):
        """对象几何被别处改了(如定位框拖改同步了自动遮罩): 按对象重设矩形, 不回写。"""
        x1, y1, x2, y2 = self._rect_of()
        self._syncing = True
        self.setRect(0, 0, max(2.0, x2 - x1), max(2.0, y2 - y1))
        self.setPos(x1, y1)
        self._syncing = False

    def _color(self):
        return self.obj.color if self.kind == "m" else self.obj.mask_color

    def _restyle(self):
        if self.view._composited:
            self.setBrush(Qt.BrushStyle.NoBrush)
            self.setPen(QPen(QColor("#ffd200") if self.isSelected() else QColor(255, 210, 0, 130),
                             1, Qt.PenStyle.DashLine))
            return
        style = getattr(self.view, "mask_style", "solid")
        if style == "blur":
            c = QColor(255, 255, 255, 60)
        elif style == "bar":
            c = QColor(0, 0, 0, 130)
        else:
            c = QColor(self._color())
        self.setBrush(QBrush(c))
        pen = QPen(QColor("#ffd200") if self.isSelected() else QColor(0, 0, 0, 60), 1)
        pen.setStyle(Qt.PenStyle.DashLine if self.isSelected() else Qt.PenStyle.SolidLine)
        self.setPen(pen)

    def _write_back(self):
        r = self.rect()
        x1, y1 = self.pos().x(), self.pos().y()
        x2, y2 = x1 + r.width(), y1 + r.height()
        if self.kind == "m":
            self.obj.x1, self.obj.y1, self.obj.x2, self.obj.y2 = x1, y1, x2, y2
        else:
            self.obj.mask = [x1, y1, x2, y2]
            self.obj.mask_custom = True      # 用户动过 → 之后完全听 mask_on
            self.obj.mask_on = True
            self.view.update_event_item(self.obj)
            self.view.notify_changed(self.obj)
        self.view.maskChanged.emit(self)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self._syncing:
            self._write_back()
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._restyle()
        return super().itemChange(change, value)

    def _zone_at(self, pos):
        rect = self.rect()
        s = max(0.05, self.view.transform().m11())
        m = 9.0 / s
        ex = -1 if abs(pos.x() - rect.left()) <= m else (1 if abs(pos.x() - rect.right()) <= m else 0)
        ey = -1 if abs(pos.y() - rect.top()) <= m else (1 if abs(pos.y() - rect.bottom()) <= m else 0)
        if not rect.adjusted(-m, -m, m, m).contains(pos):
            return 0, 0
        return ex, ey

    def hoverMoveEvent(self, e):
        ex, ey = self._zone_at(e.pos())
        self.setCursor(_zone_cursor(ex, ey, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.view.show_mask_menu(self, e.screenPos())
            e.accept()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            zone = self._zone_at(e.pos())
            if zone != (0, 0):
                self.view.editBegan.emit()
                self._rs = {"zone": zone, "p0": e.scenePos(), "rect": QRectF(self.rect()),
                            "pos": QPointF(self.pos())}
                e.accept()
                return
        self.view.editBegan.emit()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._rs is not None:
            d = e.scenePos() - self._rs["p0"]
            r0, p0 = self._rs["rect"], self._rs["pos"]
            left, top = p0.x(), p0.y()
            right, bottom = left + r0.width(), top + r0.height()
            ex, ey = self._rs["zone"]
            if ex < 0:
                left = min(left + d.x(), right - 4)
            elif ex > 0:
                right = max(right + d.x(), left + 4)
            if ey < 0:
                top = min(top + d.y(), bottom - 4)
            elif ey > 0:
                bottom = max(bottom + d.y(), top + 4)
            self._syncing = True
            self.setRect(0, 0, right - left, bottom - top)
            self.setPos(left, top)
            self._syncing = False
            self._write_back()
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._rs is not None and e.button() == Qt.MouseButton.LeftButton:
            self._rs = None
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        self.view.editBegan.emit()
        k = 1.06 if e.delta() > 0 else 1 / 1.06
        r = self.rect()
        w, h = r.width(), r.height()
        mods = e.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        nw = w if shift else w * k
        nh = h * k if (shift or not ctrl) else h
        nw, nh = max(4.0, nw), max(4.0, nh)
        cx, cy = self.pos().x() + w / 2, self.pos().y() + h / 2
        self._syncing = True
        self.setRect(0, 0, nw, nh)
        self.setPos(cx - nw / 2, cy - nh / 2)
        self._syncing = False
        self._write_back()
        e.accept()


class MaskEditDialog(QDialog):
    def __init__(self, item: MaskItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑遮罩")
        self.item = item
        f = QFormLayout(self)
        r = item.rect()
        self.sx = QDoubleSpinBox(); self.sx.setRange(-9999, 99999); self.sx.setValue(item.pos().x())
        self.sy = QDoubleSpinBox(); self.sy.setRange(-9999, 99999); self.sy.setValue(item.pos().y())
        self.sw = QDoubleSpinBox(); self.sw.setRange(4, 99999); self.sw.setValue(r.width())
        self.sh = QDoubleSpinBox(); self.sh.setRange(4, 99999); self.sh.setValue(r.height())
        f.addRow("X:", self.sx); f.addRow("Y:", self.sy)
        f.addRow("宽:", self.sw); f.addRow("高:", self.sh)
        self.st = self.se = None
        if item.kind == "m":
            self.st = QDoubleSpinBox(); self.st.setRange(0, 999999); self.st.setDecimals(2)
            self.se = QDoubleSpinBox(); self.se.setRange(0, 999999); self.se.setDecimals(2)
            self.st.setValue(item.obj.start); self.se.setValue(item.obj.end)
            f.addRow("开始(s):", self.st); f.addRow("结束(s):", self.se)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def apply(self):
        it = self.item
        it._syncing = True
        it.setRect(0, 0, self.sw.value(), self.sh.value())
        it.setPos(self.sx.value(), self.sy.value())
        it._syncing = False
        if it.kind == "m" and self.st is not None:
            it.obj.start = round(self.st.value(), 2)
            it.obj.end = max(round(self.se.value(), 2), it.obj.start + 0.1)
        it._write_back()


class _AudioShim:
    """兼容旧接口(main_window 只读 .available): 伴音由 mpv 自带。"""
    available = True


class VideoPreview(QGraphicsView):
    timeChanged = Signal(float)
    playStateChanged = Signal(bool)
    subtitleChanged = Signal(object)
    subtitleClicked = Signal(object)
    subtitleDeleteRequested = Signal(object)
    retranslateRequested = Signal(object)
    maskChanged = Signal(object)
    maskDeleteRequested = Signal(object)
    segmentChanged = Signal(object)
    editBegan = Signal()
    createSubtitleRequested = Signal(object, float, float)
    createMaskRequested = Signal(object, float, float)
    addSubtitleAt = Signal(float, float)
    toolFinished = Signal()
    toolRequested = Signal(str)

    TASK_FRAME_MAX_W = 1920          # 任务实时帧只用于观看, 4K 降到 1080p 再上屏(叠加层仍按视频坐标)

    def __init__(self, parent=None, hwdec=True):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # ---- 画面来源: mpv 画在 viewport 的 GL 帧缓冲上; pix_item 只在任务帧/进度帧/烘焙帧时可见 ----
        self.player = MpvPlayer(hwdec=hwdec)
        self._gl = MpvViewport(self.player)
        self.setViewport(self._gl)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform
                            | QPainter.RenderHint.TextAntialiasing)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setZValue(0)
        self.pix_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.pix_item.setVisible(False)
        self._scene.addItem(self.pix_item)
        self._img_keep = None
        # ---- 状态(对外契约与旧版一致) ----
        self.default_font = "Microsoft YaHei"
        self.fps = 25.0
        self.duration = 0.0
        self.cur_time = 0.0
        self.vw, self.vh = 0, 0
        self.loaded = False
        self.playing = False
        self.creation_enabled = True
        self.tool = "select"
        self.mask_style = "bar"
        self.placement = "erase"
        self.pipe_erase = True
        self.bilingual = False
        self.subtitle_format = "ass"
        self.align = "center"
        self.source_path = ""
        self.composite_hook = None       # 所见即所得: (frame_bgr, t) → 导出口栅格结果(main_window._paint_wysiwyg)
        self.frame_override = None       # 任务进行中: t → 已处理进度内的预览帧 / None
        self.show_geom = True
        self.audio = _AudioShim()
        self._baked = False              # 任务帧已含嵌字: 不建叠加项
        self._composited = False         # pix_item 显示的是 composite_hook 结果: 叠加项只画把手
        self._task_frame_shown = False   # pix_item 正显示任务实时帧(mpv 位置与 cur_time 可能不一致)
        self._file_ready = False
        self._pending_seek = None
        self._seeking = False
        self._raw_frame, self._raw_time = None, -1.0
        self._events, self._masks, self._segments = [], [], []
        self._sub_items, self._mask_items, self._geom_items, self._diag_items = [], [], [], []
        self._srt_item = None
        self._last_key = None
        self._rub = self._rub_item = None
        self._bake_timer = QTimer(self)
        self._bake_timer.setSingleShot(True)
        self._bake_timer.setInterval(40)
        self._bake_timer.timeout.connect(self._bake_now)
        br = self.player.bridge
        br.frameReady.connect(self._on_frame_ready)
        br.timePos.connect(self._on_time_pos)
        br.pauseChanged.connect(self._on_pause_changed)
        br.fileLoaded.connect(self._on_file_loaded)
        br.seekDone.connect(self._on_seek_done)

    # ================= GL 画面 / 布局 =================
    def drawBackground(self, painter, rect):
        painter.beginNativePainting()
        try:
            drawn = self._gl.render_frame()
        finally:
            painter.endNativePainting()
        if not drawn:
            painter.fillRect(rect, QColor("#0c0b19"))

    def _fit(self):
        """视图缩放 = mpv 的 keepaspect 布局(整数向下取整, 居中), 使场景坐标与 mpv 画面逐像素对齐。"""
        if not self.vw or not self.vh:
            return
        vp = self.viewport()
        dpr = vp.devicePixelRatioF()
        wd, hd = int(round(vp.width() * dpr)), int(round(vp.height() * dpr))
        if wd < 2 or hd < 2:
            return
        s = min(wd / self.vw, hd / self.vh)
        dw = max(1, int(self.vw * s + 1e-6))
        dh = max(1, int(self.vh * s + 1e-6))
        self.setTransform(QTransform.fromScale(dw / self.vw / dpr, dh / self.vh / dpr))

    def _on_frame_ready(self):
        self._gl.ack()
        self.viewport().update()

    def set_default_font(self, name):
        self.default_font = name or "Microsoft YaHei"
        self.refresh(force=True)
        self.rebake()

    def set_tool(self, tool):
        self.tool = tool if tool in TOOLS else "select"
        self.setCursor(Qt.CursorShape.CrossCursor if self.tool != "select" else Qt.CursorShape.ArrowCursor)

    # ================= 装载 / 释放 =================
    def load(self, path, audio_path=None, start=None):
        """时长/帧率/尺寸用 PyAV(与流水线同一时间源, 同步返回); 画面与声音交给 mpv(异步装载)。
        audio_path: 画面来自擦除中间视频(无音轨)时, 伴音取原视频。start: 装载完成后跳到的时刻。"""
        self.release()
        try:
            meta = video_meta(path)
        except Exception as exc:
            log.warning("无法读取视频信息 %s: %s", path, exc)
            return None
        self.vw, self.vh = int(meta["w"]), int(meta["h"])
        self.fps = float(meta["fps"] or 25.0)
        self.duration = float(meta["duration"] or 0.0)
        self._scene.setSceneRect(0, 0, self.vw, self.vh)
        self.source_path = path
        self.loaded = True
        self.cur_time = max(0.0, float(start or 0.0))
        self._pending_seek = self.cur_time if start else None
        self.player.open(path, audio_path if (audio_path and audio_path != path) else None)
        self._fit()
        return {"w": self.vw, "h": self.vh, "duration": self.duration, "fps": self.fps,
                "audio": bool(meta.get("audio") or audio_path)}

    def release(self):
        self.clear_diag()
        self._bake_timer.stop()
        self._baked = self._composited = self._task_frame_shown = False
        self.playing = False
        self.player.stop()
        self._clear_overlays()
        self._events, self._masks, self._segments = [], [], []
        self._last_key = None
        self.source_path = ""
        self.loaded = self._file_ready = self._seeking = False
        self._pending_seek = None
        self._raw_frame, self._img_keep = None, None
        self.pix_item.setVisible(False)

    def shutdown(self):
        """窗口关闭: 先释放 render context(需 GL 上下文当前), 再终止 mpv。"""
        self.release()
        try:
            self._gl.makeCurrent()
            self._gl._free_ctx()
        except Exception:
            pass
        finally:
            try:
                self._gl.doneCurrent()
            except Exception:
                pass
        self.player.terminate()

    # ================= 播放控制 =================
    def seek(self, sec, exact=True):
        """exact=False: 关键帧寻址(拖进度条时即时响应); True: 精确到帧(松手/方向键/点表格行)。"""
        if not self.loaded:
            return
        self._baked = self._task_frame_shown = False
        sec = float(sec)
        sec = max(0.0, min(sec, max(0.0, self.duration - 0.03))) if self.duration > 0 else max(0.0, sec)
        self.cur_time = sec
        self._raw_frame = None
        self._bake_timer.stop()
        if not self._apply_override(sec):
            self._show_video()
        if not self._file_ready:
            self._pending_seek = sec
        else:
            self._seeking = True
            self.player.seek(sec, exact)
        self.clear_diag()
        self.refresh(force=True)
        self.timeChanged.emit(sec)

    def play(self):
        if not self.loaded or self.playing:
            return
        if self._task_frame_shown:                 # 刚才在看任务帧: 先把 mpv 对到同一时刻
            self.seek(self.cur_time)
        self._baked = False
        self._bake_timer.stop()
        self._show_video()
        self.playing = True
        self.player.set_pause(False)
        self.playStateChanged.emit(True)

    def pause(self):
        if self.playing:
            self.playing = False
            self.player.set_pause(True)         # 停稳后由 pause 观察器触发 WYSIWYG 烘焙

    def set_muted(self, muted):
        self.player.set_mute(muted)

    def set_hwdec(self, on):
        self.player.set_hwdec(on)

    # ================= mpv 回调(已在 GUI 线程) =================
    def _on_file_loaded(self):
        self._file_ready = True
        if self._pending_seek is not None:
            t, self._pending_seek = self._pending_seek, None
            self.seek(t)
        else:
            self._raw_frame = None
            self._schedule_bake()
        self.viewport().update()

    def _on_seek_done(self):
        self._seeking = False
        self._raw_frame = None
        t = self.player.time_pos()
        if t is not None and not self.playing and abs(t - self.cur_time) > 0.002:
            self.cur_time = t                   # 关键帧寻址落点与请求不同: 叠加层/进度条对齐实际帧
            self.refresh(force=True)
            self.timeChanged.emit(t)
        if not self.playing:
            self._schedule_bake()
        self.viewport().update()

    def _on_pause_changed(self, paused):
        if paused and self.playing:             # 播到结尾(keep-open)或外部暂停
            self.playing = False
            self.playStateChanged.emit(False)
        if paused:
            self._schedule_bake()

    def _on_time_pos(self, t):
        if not self.loaded or self._baked or self._task_frame_shown or self._seeking:
            return
        if not self.playing and abs(t - self.cur_time) < 1e-4:
            return
        self.cur_time = t
        if self.playing:
            if not self._apply_override(t) and self.pix_item.isVisible():
                self._show_video()
            self.refresh()
        self.timeChanged.emit(t)

    # ================= 画面图层(pix_item) =================
    def _show_image(self, frame, max_w=None):
        h, w = frame.shape[:2]
        if max_w and w > max_w:
            frame = cv2.resize(frame, (max_w, max(2, int(round(h * max_w / w)))), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        frame = np.ascontiguousarray(frame)
        self._img_keep = frame                                     # QImage 不拷贝, 持有缓冲
        img = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
        self.pix_item.setPixmap(QPixmap.fromImage(img))
        self.pix_item.setScale(self.vw / float(w) if (self.vw and w) else 1.0)
        self.pix_item.setVisible(True)
        self.viewport().update()

    def _show_video(self):
        if self.pix_item.isVisible():
            self.pix_item.setVisible(False)
        self._set_composited(False)
        self.viewport().update()

    def _set_composited(self, flag):
        if flag == self._composited:
            return
        self._composited = flag
        for it in self._mask_items:
            it._restyle()
        for it in self._sub_items:
            it.update()

    def _apply_override(self, t):
        """任务进行中且未跟随: 进度内的时刻显示已处理画面(与旧 _show(override) 语义一致, 含 composite_hook)。"""
        fn = self.frame_override
        if fn is None:
            return False
        try:
            frame = fn(float(t))
        except Exception:
            frame = None
        if frame is None:
            return False
        composited = False
        if self.composite_hook is not None:
            try:
                frame = self.composite_hook(frame, float(t))
                composited = True
            except Exception:
                pass
        self._set_composited(composited)
        self._show_image(frame)
        return True

    def _raw_current(self):
        """当前时刻的原始画面(未叠加): 优先任务进度帧, 其次缓存, 最后向 mpv 抓取。"""
        fn = self.frame_override
        if fn is not None:
            try:
                f = fn(self.cur_time)
            except Exception:
                f = None
            if f is not None:
                return f
        if self._raw_frame is not None and abs(self._raw_time - self.cur_time) < 1e-6:
            return self._raw_frame
        if not self._file_ready or self._seeking:
            return None
        f = self.player.grab_raw()
        if f is not None:
            self._raw_frame, self._raw_time = f, self.cur_time
        return f

    def get_frame(self):
        return self._raw_current()

    # ---- WYSIWYG 烘焙: 停下来时用导出口的 paint_frame 把当前帧栅格出来, 像素级一致 ----
    def _schedule_bake(self):
        if (self.composite_hook is None or self.playing or self._baked
                or self._task_frame_shown or not self.loaded):
            return
        self._bake_timer.start()

    def _bake_now(self):
        if self.composite_hook is None or self.playing or self._baked or self._task_frame_shown:
            return
        frame = self._raw_current()
        if frame is None:
            return
        try:
            out = self.composite_hook(frame, self.cur_time)
        except Exception:
            out = frame
        self._set_composited(True)
        self._show_image(out)

    def rebake(self):
        """字幕/遮罩被编辑后: 不重新寻址, 用缓存的原始帧重烘(拖动时 40ms 合并一次)。"""
        if self.composite_hook is not None and not self.playing:
            self._schedule_bake()

    # ---- 任务实时画面(v6_live) ----
    def show_task_frame(self, frame, t, baked):
        if self.playing:
            self.playing = False
            self.player.set_pause(True)
            self.playStateChanged.emit(False)
        self._bake_timer.stop()
        self._task_frame_shown = True
        self._baked = bool(baked)
        self._set_composited(False)
        self.cur_time = float(t)
        h, w = frame.shape[:2]
        if (self.vw, self.vh) != (w, h):
            self.vw, self.vh = w, h
            self._scene.setSceneRect(0, 0, w, h)
            self._fit()
        self._show_image(frame, max_w=self.TASK_FRAME_MAX_W)
        self.clear_diag()
        self.refresh()
        self.timeChanged.emit(self.cur_time)

    def leave_task_view(self):
        """取消[跟随任务画面]: 任务帧让位, mpv 对到当前时刻。"""
        if not self._task_frame_shown:
            return
        self._task_frame_shown = self._baked = False
        if self.loaded:
            self.seek(self.cur_time)
        else:
            self._show_video()

    # ================= 叠加层 =================
    def _after_data_change(self):
        self.refresh(force=True)
        self.rebake()
    # set_events / set_masks / _clear_overlays / set_segments / refresh / set_show_geom / set_diag /
    # clear_diag / update_event_item / update_event_mask / select_event / select_mask / notify_changed /
    # notify_clicked —— 与原版相同

    # ================= 空白处框选创建 / 菜单 / 视图 =================
    # _near_any_item / mousePressEvent(仅 self.cap→self.loaded) / mouseMoveEvent / wheelEvent /
    # mouseReleaseEvent / _blank_menu / keyPressEvent / show_item_menu / show_mask_menu / resizeEvent —— 与原版相同


    def set_events(self, events):
        self._events = events
        self._after_data_change()


    def set_masks(self, masks):
        self._masks = masks
        self._after_data_change()


    def _clear_overlays(self):
        items = self._sub_items + self._mask_items + self._geom_items
        if self._srt_item is not None:
            items.append(self._srt_item)
        for it in items:
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._sub_items, self._mask_items, self._geom_items = [], [], []
        self._srt_item = None


    def set_segments(self, segs):
        self._segments = list(segs or [])
        self.refresh(force=True)


    def refresh(self, force=False):
        if (self._scene.mouseGrabberItem() is not None
                and QApplication.mouseButtons() != Qt.MouseButton.NoButton):
            self._last_key = None       # 正在拖动叠加项: 此刻重建会打断手势; 标记脏, 下次刷新再重建
            return
        t = self.cur_time
        act_e = [e for e in self._events if e.start <= t < e.end]
        act_m = [m for m in self._masks if m.start <= t < m.end]
        if self._baked:
            act_e, act_m = [], []
        if any(getattr(e, "geom_track", None) for e in act_e):
            force = True
        act_s = []
        if self.show_geom and self._segments:
            used = {e.source_seg_id for e in self._events if e.source_seg_id}   # 已生成字幕的候选段由字幕框代表
            act_s = [s for s in self._segments
                     if s.start - 0.05 <= t <= s.end + 0.05 and s.id not in used][:12]
        srt = self.subtitle_format == "srt"
        key = (tuple(sorted(e.id for e in act_e)), tuple(sorted(m.id for m in act_m)),
               tuple(s.id for s in act_s), self.show_geom, self.placement, self.pipe_erase, self._baked, srt)
        if not force and key == self._last_key:
            return
        self._last_key = key
        self._clear_overlays()
        for m in act_m:
            it = MaskItem("m", m, self)
            self._scene.addItem(it)
            self._mask_items.append(it)
        for ev in act_e:
            if event_covered(ev, self.placement, self.pipe_erase):
                if not (ev.mask and len(ev.mask) == 4):
                    ev.ensure_mask()
                it = MaskItem("e", ev, self)
                self._scene.addItem(it)
                self._mask_items.append(it)
        for s in act_s:
            if s.verdict:                       # 被剔除的候选: 只读红框标出原因
                gi = GeomItem(s.quad(), f"候选 {getattr(s, 'score', 0.0):.2f} ✗{s.verdict[:10]}",
                              "#e05c6e", dashed=False)
            else:                               # 待识别的候选: 可拖改的绿框
                gi = GeomEditItem("s", s, self)
            self._scene.addItem(gi)
            self._geom_items.append(gi)
        if self.show_geom:
            for ev in act_e:
                if ev.has_text() and source_quad(ev, t) is not None:
                    gi = GeomEditItem("e", ev, self)
                    self._scene.addItem(gi)
                    self._geom_items.append(gi)
        srt_events = []
        for ev in act_e:
            if srt and ev.display_text():
                srt_events.append(ev)           # SRT: 文字层统一由 SrtOverlayItem 按播放器默认样式显示
                continue
            it = SubtitleItem(ev, self)
            self._scene.addItem(it)
            self._sub_items.append(it)
            it.refresh_from_event()
        if srt_events:
            self._srt_item = SrtOverlayItem(srt_events, self)
            self._scene.addItem(self._srt_item)


    def set_show_geom(self, v):
        self.show_geom = bool(v)
        self.refresh(force=True)


    def set_diag(self, items):
        self.clear_diag()
        for d in items:
            gi = GeomItem(d["quad"], d.get("label", ""), "#3fbf6f" if d.get("ok") else "#e05c6e", dashed=False)
            gi.setZValue(32)
            self._scene.addItem(gi)
            self._diag_items.append(gi)


    def clear_diag(self):
        for it in self._diag_items:
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._diag_items = []


    def update_event_item(self, ev):
        for it in self._sub_items:
            if it.ev is ev:
                it.refresh_from_event()


    def update_event_mask(self, ev):
        """字幕的定位框改了: 遮罩覆盖方式下该字幕的自动遮罩项随之同步(用户改过的遮罩不受影响, 见 set_quad)。"""
        for it in self._mask_items:
            if it.kind == "e" and it.obj is ev:
                it.sync_from_obj()


    def select_event(self, ev):
        self._scene.clearSelection()
        for it in self._sub_items:
            if it.ev is ev:
                it.setSelected(True)
        for it in self._geom_items:
            if isinstance(it, GeomEditItem) and it.kind == "e" and it.obj is ev:
                it.setSelected(True)


    def select_mask(self, obj):
        self._scene.clearSelection()
        for it in self._mask_items:
            if it.obj is obj:
                it.setSelected(True)


    def notify_changed(self, ev):
        self.subtitleChanged.emit(ev)


    def notify_clicked(self, ev):
        self.subtitleClicked.emit(ev)


    def _near_any_item(self, sp):
        s = max(0.05, self.transform().m11())
        m = 14.0 / s
        for it in self._sub_items + self._mask_items:
            if it.sceneBoundingRect().adjusted(-m, -m, m, m).contains(sp):
                return True
        return False


    def mousePressEvent(self, e):
        if self.loaded and self.creation_enabled and self._rub is None:
            sp = self.mapToScene(e.position().toPoint())
            kind = None
            if e.button() == Qt.MouseButton.LeftButton and self.tool in ("subtitle", "mask"):
                kind = self.tool
            elif e.button() == Qt.MouseButton.RightButton:
                kind = "mask"          # 右键拖框=遮挡块(快捷方式); 右键单击=添加菜单
            if kind and self._scene.sceneRect().contains(sp) and not self._near_any_item(sp):
                self.pause()
                self.playStateChanged.emit(False)
                self._rub = {"btn": e.button(), "p0": sp, "t0": self.cur_time, "kind": kind}
                self._rub_item = QGraphicsRectItem(QRectF(sp, sp))
                color = QColor("#4f7cff") if kind == "subtitle" else QColor("#ffd200")
                self._rub_item.setPen(QPen(color, 0, Qt.PenStyle.DashLine))
                if kind == "mask":
                    c = QColor(color)
                    c.setAlpha(50)
                    self._rub_item.setBrush(QBrush(c))
                self._rub_item.setZValue(30)
                self._scene.addItem(self._rub_item)
                e.accept()
                return
        super().mousePressEvent(e)


    def mouseMoveEvent(self, e):
        if self._rub:
            sp = self.mapToScene(e.position().toPoint())
            self._rub_item.setRect(QRectF(self._rub["p0"], sp).normalized())
            e.accept()
            return
        super().mouseMoveEvent(e)


    def wheelEvent(self, e):
        if self._rub:  # 框选中滚轮 = 视频前后移动决定时长
            step = 0.4 if e.angleDelta().y() > 0 else -0.4
            if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                step *= 4
            self.seek(self.cur_time + step)
            e.accept()
            return
        super().wheelEvent(e)


    def mouseReleaseEvent(self, e):
        if self._rub and e.button() == self._rub["btn"]:
            rect = self._rub_item.rect().intersected(self._scene.sceneRect())
            btn, t0, kind = self._rub["btn"], self._rub["t0"], self._rub["kind"]
            p0 = self._rub["p0"]
            self._scene.removeItem(self._rub_item)
            self._rub, self._rub_item = None, None
            if rect.width() >= 8 and rect.height() >= 8:
                ta, tb = min(t0, self.cur_time), max(t0, self.cur_time)
                if tb - ta < 0.25:
                    tb = ta + 2.0
                if kind == "subtitle":
                    self.createSubtitleRequested.emit(rect, ta, tb)
                else:
                    self.createMaskRequested.emit(rect, ta, tb)
                self.toolFinished.emit()
            elif btn == Qt.MouseButton.RightButton:
                self._blank_menu(e.globalPosition().toPoint(), p0)
            e.accept()
            return
        super().mouseReleaseEvent(e)


    def _blank_menu(self, screen_pos, sp):
        menu = QMenu(self)
        a_sub = menu.addAction("框选添加字幕(切换到该工具)")
        a_mask = menu.addAction("框选添加遮挡块(切换到该工具)")
        a_here = menu.addAction("在此处添加一条字幕")
        act = menu.exec(screen_pos)
        if act is a_sub:
            self.toolRequested.emit("subtitle")
        elif act is a_mask:
            self.toolRequested.emit("mask")
        elif act is a_here:
            self.addSubtitleAt.emit(float(sp.x()), float(sp.y()))


    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete:
            subs = [it.ev for it in self._sub_items if it.isSelected()]
            masks = [it for it in self._mask_items if it.isSelected()]
            for ev in subs:
                self.subtitleDeleteRequested.emit(ev)
            for it in masks:
                self.maskDeleteRequested.emit(it)
            if subs or masks:
                e.accept()
                return
        super().keyPressEvent(e)


    def show_item_menu(self, ev, screen_pos):
        menu = QMenu(self)
        a_retr = menu.addAction("翻译此条")
        a_color = menu.addAction("更改颜色…")
        a_ocolor = menu.addAction("描边颜色…")
        a_font = menu.addAction("选择字体…(全部)")
        fmenu = menu.addMenu("常用字体")
        fams = set(QFontDatabase.families())
        font_acts = {fmenu.addAction("(跟随默认字体)"): ""}
        for name in QUICK_FONTS:
            if name in fams:
                font_acts[fmenu.addAction(name)] = name
        a_vert = menu.addAction("切换横排/竖排")
        a_auto = menu.addAction("恢复自动排布(按原字框缩放)")
        a_auto.setEnabled(ev.fit_mode == "manual" and source_quad(ev, None) is not None)
        covered = event_covered(ev, self.placement, self.pipe_erase)
        a_mask = menu.addAction(("[开] " if covered else "[关] ") + "遮罩(按设置里的遮罩样式)")
        a_mcolor = menu.addAction("遮罩颜色…(仅纯色样式)")
        a_reset = menu.addAction("旋转归零")
        menu.addSeparator()
        a_big = menu.addAction("放大字号 (+4)  [滚轮]")
        a_small = menu.addAction("缩小字号 (−4)  [滚轮]")
        menu.addSeparator()
        a_del = menu.addAction("删除此字幕  [Del]")
        act = menu.exec(screen_pos)
        if act is None:
            return
        if act is a_del:
            self.subtitleDeleteRequested.emit(ev)
            return
        if act is a_retr:
            self.retranslateRequested.emit(ev)
            return
        self.editBegan.emit()
        if act is a_color:
            c = QColorDialog.getColor(QColor(ev.color), self, "选择字幕颜色")
            if c.isValid():
                ev.color = c.name().upper()
        elif act is a_ocolor:
            c = QColorDialog.getColor(QColor(ev.outline_color), self, "选择描边颜色")
            if c.isValid():
                ev.outline_color = c.name().upper()
        elif act is a_font:
            families = sorted(QFontDatabase.families())
            items = ["(跟随默认字体)"] + families
            cur = items.index(ev.font) if ev.font in families else 0
            text, ok = QInputDialog.getItem(self, "选择字体", "字体(可输入筛选):", items, cur, True)
            if not ok:
                return
            ev.font = "" if text == "(跟随默认字体)" else text.strip()
        elif act in font_acts:
            ev.font = font_acts[act]
        elif act is a_vert:
            ev.vertical = not ev.vertical
        elif act is a_auto:
            ev.fit_mode = "contain"
        elif act is a_mask:
            ev.mask_custom = True
            ev.mask_on = not covered
            if ev.mask_on:
                ev.ensure_mask()
        elif act is a_mcolor:
            c = QColorDialog.getColor(QColor(ev.mask_color), self, "选择遮罩颜色")
            if c.isValid():
                ev.mask_color = c.name().upper()
        elif act is a_reset:
            ev.rotation = 0.0
        elif act is a_big:
            ev.fit_mode = "manual"
            ev.font_size = min(300, ev.font_size + 4)
        elif act is a_small:
            ev.fit_mode = "manual"
            ev.font_size = max(8, ev.font_size - 4)
        self.refresh(force=True)
        self.subtitleChanged.emit(ev)


    def show_mask_menu(self, item: MaskItem, screen_pos):
        menu = QMenu(self)
        a_color = menu.addAction("更改遮罩颜色…(仅纯色样式)")
        a_pick = menu.addAction("取当前画面底色")
        a_edit = menu.addAction("编辑位置/尺寸…")
        menu.addSeparator()
        a_del = menu.addAction("删除遮罩  [Del]")
        act = menu.exec(screen_pos)
        if act is None:
            return
        if act is a_del:
            self.maskDeleteRequested.emit(item)
            return
        self.editBegan.emit()
        if act is a_color:
            c = QColorDialog.getColor(QColor(item._color()), self, "遮罩颜色")
            if c.isValid():
                if item.kind == "m":
                    item.obj.color = c.name().upper()
                else:
                    item.obj.mask_color = c.name().upper()
                item._restyle()
        elif act is a_pick:
            if self.cur_frame is not None:
                from core.enhance import estimate_bg_color
                r = item.rect()
                x1, y1 = item.pos().x(), item.pos().y()
                col = estimate_bg_color(self.cur_frame, x1, y1, x1 + r.width(), y1 + r.height())
                if item.kind == "m":
                    item.obj.color = col
                else:
                    item.obj.mask_color = col
                item._restyle()
        elif act is a_edit:
            dlg = MaskEditDialog(item, self)
            if dlg.exec():
                dlg.apply()
        self.maskChanged.emit(item)


    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()
