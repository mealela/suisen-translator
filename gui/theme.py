"""主题: 暗色(#11101f 系, 默认) / 深蓝(更蓝更亮一档) / 亮色(#e4dfdd 系)。
运行时可切换, 调用 apply_theme(app, mode)。"""
from string import Template

from PySide6.QtGui import QColor, QPalette

DARK_BASE = "#11101f"
LIGHT_BASE = "#e4dfdd"

PALETTES = {
    # 色阶同色相单调递亮: base < bg < alt < panel < border < panel2 < sel < border2
    "dark": dict(
        base="#0c0b19", bg=DARK_BASE, alt="#17152b", border="#262347",
        panel="#1c1a35", panel2="#272449", sel="#2b2e64", border2="#3a3773",
        text="#bcc0dc", hint="#7e83a8", muted="#545879",
        accent="#6673d8", accent2="#7683e2", accent_fg="#0b0c1f",
        danger_bg="#2a1626", danger_bd="#4a2542", danger_fg="#e28aa3",
        sel_fg="#e3e6fa", tip_bg="#1c1a35", scroll="#33305f",
        ok="#6dbd7c", warn="#d2a45c",
    ),
    "blue": dict(
        base="#0b1020", bg="#101626", alt="#161d31", border="#26304d",
        panel="#1b2439", panel2="#27324f", sel="#2b3a6a", border2="#38467c",
        text="#bcc8e0", hint="#7f8fb2", muted="#55638a",
        accent="#5581d4", accent2="#6690dd", accent_fg="#071021",
        danger_bg="#2a1626", danger_bd="#4a2542", danger_fg="#e28aa3",
        sel_fg="#e3eafa", tip_bg="#1b2439", scroll="#334270",
        ok="#6bbf80", warn="#d1a45a",
    ),
    "light": dict(
        bg=LIGHT_BASE, panel="#f1edeb", panel2="#faf8f7", base="#ffffff",
        alt="#f4f0ee", border="#c9c1bd", border2="#a89e99",
        text="#221d33", hint="#6b6373", muted="#a29ba7",
        accent="#4b3fb5", accent2="#5d51c9", accent_fg="#ffffff",
        danger_bg="#f7dde4", danger_bd="#d99aaf", danger_fg="#a12a4a",
        sel="#d3ccf4", sel_fg="#1a1533", ok="#2e8b57", warn="#b26a00",
        tip_bg="#fffaf5", scroll="#b8b0ac",
    ),
}

_QSS = Template("""
QMainWindow, QDialog, QWidget { background-color: $bg; color: $text;
  font-family: 'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif;
  font-size: 13px; }
QLabel { background-color: transparent; }
QLabel[hint="true"] { color: $hint; font-size: 12px; }
QToolTip { background: $tip_bg; color: $text; border: 1px solid $border;
  padding: 6px 8px; border-radius: 6px; }
QScrollArea, QStackedWidget { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background-color: $bg; }

QPushButton, QToolButton { background-color: $panel; border: 1px solid $border;
  border-radius: 9px; padding: 6px 14px; min-height: 22px; color: $text; }
QPushButton:hover, QToolButton:hover { background-color: $panel2; border-color: $border2; }
QPushButton:pressed, QToolButton:pressed { background-color: $base; }
QPushButton:disabled, QToolButton:disabled { color: $muted; background-color: $alt;
  border-color: $border; }
QPushButton:checked { background-color: $sel; border-color: $accent; color: $sel_fg; }
QPushButton[accent="true"] { background-color: $accent; border: none; color: $accent_fg;
  font-weight: bold; }
QPushButton[accent="true"]:hover { background-color: $accent2; }
QPushButton[accent="true"]:disabled { background-color: $alt; color: $muted; }
QPushButton[danger="true"] { background-color: $danger_bg; border-color: $danger_bd;
  color: $danger_fg; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox, QPlainTextEdit {
  background-color: $base; color: $text; border: 1px solid $border; border-radius: 8px;
  padding: 5px 10px; min-height: 22px;
  selection-background-color: $sel; selection-color: $sel_fg; }
QPlainTextEdit { padding: 8px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus { border-color: $accent; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background-color: $panel; color: $text; border: 1px solid $border;
  border-radius: 8px; padding: 4px; selection-background-color: $sel; }
QComboBox QAbstractItemView::item { min-height: 26px; padding: 4px 8px; }

QAbstractItemView { background-color: $base; color: $text; }
QTableWidget { background-color: $base; alternate-background-color: $alt;
  border: 1px solid $border; border-radius: 10px; gridline-color: $alt;
  selection-background-color: $sel; selection-color: $sel_fg; }
QTableWidget::item { padding: 5px 8px; }
QHeaderView::section { background-color: $panel; color: $hint; border: none;
  border-bottom: 1px solid $border; padding: 8px 10px; font-weight: bold; }
QTableCornerButton::section { background-color: $panel; border: none; }
QListWidget { background-color: $base; border: 1px solid $border; border-radius: 10px;
  padding: 4px; }
QListWidget::item { border-radius: 8px; margin: 2px; padding: 3px; }
QListWidget::item:selected { background: $sel; color: $sel_fg; }
QListWidget::item:hover { background: $alt; }

QMenu { background: $panel; border: 1px solid $border; border-radius: 10px; padding: 6px; }
QMenu::item { padding: 7px 26px; border-radius: 6px; }
QMenu::item:selected { background: $sel; color: $sel_fg; }
QMenu::separator { height: 1px; background: $border; margin: 5px 10px; }

QGroupBox { background-color: $alt; border: 1px solid $border; border-radius: 10px;
  margin-top: 16px; padding: 10px 6px 6px 6px; font-weight: bold; color: $hint; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: $accent; }
QFrame#CollapsibleBody { background-color: $alt; border: 1px solid $border;
  border-radius: 8px; }

QSlider::groove:horizontal { height: 5px; background: $border; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; background: $accent;
  border-radius: 8px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QProgressBar { background-color: $base; border: 1px solid $border; border-radius: 7px;
  text-align: center; color: $text; min-height: 18px; font-weight: bold; }
QProgressBar::chunk { background-color: $accent; border-radius: 6px; }
QStatusBar { background: $panel; border-top: 1px solid $border; }
QGraphicsView { border: 1px solid $border; border-radius: 10px; background: $base; }
QSplitter::handle { background: transparent; width: 6px; }
QSplitter::handle:hover { background: $sel; }
QCheckBox { spacing: 7px; background: transparent; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px;
  border: 1px solid $border; background: $base; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: $scroll; min-height: 36px; border-radius: 5px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: $scroll; min-width: 36px; border-radius: 5px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QMessageBox { background-color: $panel; }

QFrame#TopBar { background: $panel; border: 1px solid $border; border-radius: 14px; }
#TopBar QLabel#AppTitle { font-size: 18px; font-weight: 700; color: $text; }
/* 顶栏本身就是 panel, 上面的按钮再抬一级, 否则和卡片同色只剩一圈边 */
#TopBar QPushButton, #TopBar QToolButton { min-height: 0px; padding: 0px 12px;
  border-radius: 8px; background-color: $panel2; border-color: $border2; }
#TopBar QPushButton:hover, #TopBar QToolButton:hover { background-color: $sel;
  border-color: $accent; }
QFrame#PipeStep { background: $alt; border: 1px solid $border; border-radius: 9px; }
QFrame#PipeStep[included="true"] { background: $base; border-color: $accent; }
QFrame#PipeStep QPushButton { min-width: 0px; min-height: 0px; padding: 0px 6px;
  border: none; background: transparent; color: $text; }
QFrame#PipeStep QPushButton:hover { background: $sel; color: $sel_fg; }
QFrame#PipeStep QPushButton:disabled { background: transparent; color: $muted; }
QFrame#PipeStep QCheckBox { margin: 0px; spacing: 0px; }
QFrame#LogoBox { background: $panel; border: 1px solid $border; border-radius: 16px; }
""")


def palette(mode):
    return PALETTES[mode if mode in PALETTES else "dark"]


def build_qss(mode="dark"):
    return _QSS.substitute(**palette(mode))


def apply_theme(app, mode="dark"):
    p = palette(mode)
    pal = QPalette()
    R = QPalette.ColorRole
    for role, key in ((R.Window, "bg"), (R.WindowText, "text"), (R.Base, "base"),
                      (R.AlternateBase, "alt"), (R.Text, "text"), (R.Button, "panel"),
                      (R.ButtonText, "text"), (R.Highlight, "sel"),
                      (R.HighlightedText, "sel_fg"), (R.ToolTipBase, "tip_bg"),
                      (R.ToolTipText, "text"), (R.PlaceholderText, "hint")):
        pal.setColor(role, QColor(p[key]))
    app.setPalette(pal)
    app.setStyleSheet(build_qss(mode))


# 兼容旧引用: from gui.theme import QSS
QSS = build_qss("dark")
