"""小对话框: 批量修改样式。"""
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QWidget)


def _color_button(color):
    btn = QPushButton()
    btn.setFixedSize(56, 26)
    btn.setProperty("value", color)
    btn.setStyleSheet(f"background:{color}; border:1px solid #555; border-radius:6px;")

    def pick():
        c = QColorDialog.getColor(QColor(btn.property("value")), btn, "选择颜色")
        if c.isValid():
            btn.setProperty("value", c.name().upper())
            btn.setStyleSheet(f"background:{c.name()}; border:1px solid #555; border-radius:6px;")
    btn.clicked.connect(pick)
    return btn


class BatchStyleDialog(QDialog):
    """对选中的多条字幕统一修改样式: 勾选哪一项就只改哪一项。"""

    def __init__(self, n, sample, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"批量修改样式 · {n} 条")
        form = QFormLayout(self)
        form.addRow(QLabel(f"将对选中的 {n} 条字幕应用勾选的项目(未勾选的保持原样)。"))

        def row(label, widget):
            chk = QCheckBox()
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(chk)
            h.addWidget(widget, 1)
            form.addRow(label, box)
            return chk

        self.cmb_font = QComboBox()
        self.cmb_font.addItem("(跟随默认字体)")
        self.cmb_font.addItems(sorted(QFontDatabase.families()))
        if sample.font:
            idx = self.cmb_font.findText(sample.font)
            self.cmb_font.setCurrentIndex(idx if idx >= 0 else 0)
        self.chk_font = row("字体", self.cmb_font)
        self.sp_fs = QSpinBox()
        self.sp_fs.setRange(8, 300)
        self.sp_fs.setValue(int(sample.font_size))
        self.chk_fs = row("字号(并切为手动排布)", self.sp_fs)
        self.cmb_fit = QComboBox()
        self.cmb_fit.addItem("自动: 保持字形比例", "contain")
        self.cmb_fit.addItem("自动: 拉伸填满原字幕框", "stretch")
        self.cmb_fit.addItem("手动字号", "manual")
        self.chk_fit = row("排布方式", self.cmb_fit)
        self.btn_color = _color_button(sample.color)
        self.chk_color = row("文字颜色", self.btn_color)
        self.btn_ocolor = _color_button(sample.outline_color)
        self.chk_ocolor = row("描边颜色", self.btn_ocolor)
        self.sp_ow = QDoubleSpinBox()
        self.sp_ow.setRange(0.0, 30.0)
        self.sp_ow.setDecimals(1)
        self.sp_ow.setSpecialValueText("自动")
        self.sp_ow.setValue(float(sample.outline_width))
        self.chk_ow = row("描边宽", self.sp_ow)
        self.chk_bold_v = QCheckBox("粗体")
        self.chk_bold_v.setChecked(bool(sample.bold))
        self.chk_bold = row("粗体", self.chk_bold_v)
        self.chk_vert_v = QCheckBox("竖排")
        self.chk_vert_v.setChecked(bool(sample.vertical))
        self.chk_vert = row("竖排", self.chk_vert_v)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def apply(self, events):
        n = 0
        for ev in events:
            if self.chk_font.isChecked():
                ev.font = "" if self.cmb_font.currentIndex() == 0 else self.cmb_font.currentText()
            if self.chk_fs.isChecked():
                ev.font_size = self.sp_fs.value()
                ev.fit_mode = "manual"
            if self.chk_fit.isChecked():
                ev.fit_mode = self.cmb_fit.currentData()
            if self.chk_color.isChecked():
                ev.color = str(self.btn_color.property("value"))
            if self.chk_ocolor.isChecked():
                ev.outline_color = str(self.btn_ocolor.property("value"))
            if self.chk_ow.isChecked():
                ev.outline_width = round(self.sp_ow.value(), 1)
            if self.chk_bold.isChecked():
                ev.bold = self.chk_bold_v.isChecked()
            if self.chk_vert.isChecked():
                ev.vertical = self.chk_vert_v.isChecked()
            n += 1
        return n
