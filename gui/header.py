from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMenu,
                               QPushButton, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from core.i18n import tr

# (属性名, 显示文字, 配置键)。顶部勾选=加入队列自动流程; 点按钮=单独执行该步。
PIPE_STEPS = [
    ("btn_locate", "01  定位", "pipe_locate"),
    ("btn_read", "02  识别", "pipe_read"),
    ("btn_trans", "03  翻译", "pipe_translate"),
    ("btn_erase", "04  原字处理", "pipe_erase"),
    ("btn_render", "05  导出", "pipe_render"),
]

STEP_TIPS = {
    "pipe_locate": "找出画面中字幕出现的位置与时间段(工作区内, 不生成文件)",
    "pipe_read": "读出定位到的文字(工作区内, 不生成文件)",
    "pipe_translate": "把原文翻译成目标语言; 漏译的会自动补译一次",
    "pipe_erase": "勾选=按[设置→原字处理]选的方式处理原字幕: AI 无痕擦除 / 遮罩覆盖 / 保留原字。\n"
                  "单独点按钮=只在工作区内处理, 不生成文件; 擦除结果缓存起来, 导出时直接复用, "
                  "字幕改动后只重擦有改动的时间段",
    "pipe_render": "输出成品视频到输出目录(可在设置里勾选同时生成字幕文件, 格式 ASS/SRT 在设置中选); 这是唯一会写文件的步骤",
}


class ElidedLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__("", parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self.setFixedHeight(17)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._update_text()

    def _update_text(self):
        super().setText(self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, max(1, self.contentsRect().width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_text()


def build_header(win):
    header = QWidget()
    outer = QHBoxLayout(header)
    outer.setContentsMargins(10, 8, 10, 8)
    outer.setSpacing(10)
    outer.addWidget(win._logo_box(), 0, Qt.AlignmentFlag.AlignTop)

    panel = QFrame()
    panel.setObjectName("TopBar")
    panel.setFixedHeight(115)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(4)

    def button(text):
        obj = QPushButton(text)
        obj.setFixedHeight(34)
        return obj

    actions = QHBoxLayout()
    actions.setSpacing(8)
    title = QLabel(tr("水仙翻译器"))
    title.setObjectName("AppTitle")
    actions.addWidget(title)
    actions.addSpacing(10)
    win.btn_open = button(tr("打开视频"))
    win.btn_recent = QToolButton()
    win.btn_recent.setText(tr("最近"))
    win.btn_recent.setFixedHeight(34)
    win.btn_recent.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    win._recent_menu = QMenu(win.btn_recent)
    win.btn_recent.setMenu(win._recent_menu)
    win._rebuild_recent()
    actions.addWidget(win.btn_open)
    actions.addWidget(win.btn_recent)
    actions.addStretch(1)
    win.btn_models = button(tr("模型库"))
    win.btn_settings = button(tr("设置"))
    win.btn_models.setMinimumWidth(96)
    win.btn_settings.setMinimumWidth(84)
    actions.addWidget(win.btn_models)
    actions.addWidget(win.btn_settings)
    layout.addLayout(actions)

    steps = QHBoxLayout()
    steps.setSpacing(6)
    for attr, text, key in PIPE_STEPS:
        frame = QFrame()
        frame.setObjectName("PipeStep")
        frame.setFixedHeight(40)
        included = bool(getattr(win.cfg, key))
        frame.setProperty("included", included)
        row = QHBoxLayout(frame)
        row.setContentsMargins(7, 2, 2, 2)
        row.setSpacing(3)
        check = QCheckBox()
        check.setChecked(included)
        tip = STEP_TIPS.get(key, "")
        check.setToolTip("是否加入自动队列流程" + ("\n" + tip if tip else ""))
        action = button(tr(text))
        action.setToolTip("单独执行此步骤(表格里选中了字幕 → 只对选中的字幕重做; 没选中 → 整段视频)"
                          + ("\n" + tip if tip else ""))
        action.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        setattr(win, attr, action)
        win._pipe_chks[key] = check

        def toggle(value, k=key, box=frame):
            win._set_pipe(k, value)
            box.setProperty("included", bool(value))
            box.style().unpolish(box)
            box.style().polish(box)
            box.update()

        check.toggled.connect(toggle)
        row.addWidget(check)
        row.addWidget(action, 1)
        steps.addWidget(frame, 1)
    layout.addLayout(steps)

    win.lbl_engine = ElidedLabel()
    win.lbl_engine.setProperty("hint", True)
    layout.addWidget(win.lbl_engine)
    outer.addWidget(panel, 1)
    return header
