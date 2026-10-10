"""通用小控件: 可展开/收起的高级设置区。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleBox(QWidget):
    def __init__(self, title, expanded=False, parent=None):
        super().__init__(parent)
        self._btn = QToolButton(text=title)
        self._btn.setCheckable(True)
        self._btn.setChecked(bool(expanded))
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setStyleSheet(
            "QToolButton{border:none;background:transparent;font-weight:bold;"
            "text-align:left;padding:4px 2px;}")
        self._body = QFrame()
        self._body.setObjectName("CollapsibleBody")
        self._body.setVisible(bool(expanded))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(self._btn)
        lay.addWidget(self._body)
        self._btn.toggled.connect(self._on_toggle)
        self._on_toggle(bool(expanded))

    def _on_toggle(self, on):
        self._body.setVisible(on)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def set_content_layout(self, layout):
        layout.setContentsMargins(10, 8, 10, 8)
        self._body.setLayout(layout)

    def set_expanded(self, on):
        self._btn.setChecked(bool(on))

    def is_expanded(self):
        return self._btn.isChecked()
