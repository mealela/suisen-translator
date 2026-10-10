"""首次使用引导: 半透明遮罩高亮目标控件, 一步步指出位置。只出现一次。"""
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from core.i18n import tr
from gui.theme import palette


class TutorialOverlay(QWidget):
    def __init__(self, win, steps, on_done):
        super().__init__(win)
        self.win, self.steps, self.on_done = win, steps, on_done
        self.idx = 0
        self._pal = palette(getattr(getattr(win, "cfg", None), "ui_theme", "dark"))
        pal = self._pal
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setGeometry(win.rect())
        self.card = QFrame(self)
        self.card.setStyleSheet(
            f"QFrame{{background:{pal['panel']};border:1px solid {pal['border2']};"
            "border-radius:12px;}")
        v = QVBoxLayout(self.card)
        v.setContentsMargins(16, 12, 16, 12)
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(f"font-size:15px;font-weight:bold;"
                                     f"color:{pal['accent2']};background:transparent;")
        self.lbl_text = QLabel("")
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setStyleSheet("background:transparent;")
        self.lbl_text.setMinimumWidth(340)
        self.lbl_text.setMaximumWidth(380)
        row = QVBoxLayout()
        self.btn_next = QPushButton(tr("下一步 →"))
        self.btn_next.setProperty("accent", True)
        self.btn_skip = QPushButton(tr("跳过教程"))
        v.addWidget(self.lbl_title)
        v.addWidget(self.lbl_text)
        v.addWidget(self.btn_next)
        v.addWidget(self.btn_skip)
        self.btn_next.clicked.connect(self._next)
        self.btn_skip.clicked.connect(self._finish)
        self._apply()

    def _target_rect(self):
        widget = self.steps[self.idx][0]
        if widget is None or not widget.isVisible():
            return None
        pos = widget.mapTo(self.win, widget.rect().topLeft())
        return QRect(pos, widget.size()).adjusted(-6, -6, 6, 6)

    def _apply(self):
        _, title, text = self.steps[self.idx]
        step = f"({self.idx + 1}/{len(self.steps)}) "
        self.lbl_title.setText(step + title)
        self.lbl_text.setText(text)
        self.btn_next.setText(tr("完成 ✔") if self.idx == len(self.steps) - 1
                              else tr("下一步 →"))
        self.card.adjustSize()
        r = self._target_rect()
        if r is None:
            self.card.move((self.width() - self.card.width()) // 2,
                           (self.height() - self.card.height()) // 2)
        else:
            x = min(max(8, r.left()), self.width() - self.card.width() - 8)
            y = r.bottom() + 10
            if y + self.card.height() > self.height() - 8:
                y = max(8, r.top() - self.card.height() - 10)
            self.card.move(x, y)
        self.update()

    def _next(self):
        if self.idx >= len(self.steps) - 1:
            self._finish()
            return
        self.idx += 1
        self._apply()

    def _finish(self):
        try:
            self.on_done()
        finally:
            self.hide()
            self.deleteLater()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        r = self._target_rect()
        if r is not None:
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(r), 10, 10)
            path = path.subtracted(hole)
            pen_path = QPainterPath()
            pen_path.addRoundedRect(QRectF(r), 10, 10)
            p.setPen(QColor(self._pal["accent"]))
            p.drawPath(pen_path)
        p.fillPath(path, QColor(6, 8, 14, 170))
        p.end()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply()

    def mousePressEvent(self, e):
        e.accept()   # 挡住底下控件, 引导期间只能点卡片按钮
