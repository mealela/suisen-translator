"""首次配置向导: 场景 → 翻译引擎 → 读字引擎, 三步、每步一个动作。
本地推荐模型一键填好, 勾选[完成后自动下载]则向导结束直接排队下载(不再多问一次); 云端 API 需测试通过。"""
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from config import DEFAULT_MT_TAG, DEFAULT_VLM_TAG
from core import ai
from core import model_manager as mm

PRESETS = [
    ("日语动画/AI视频 → 中文", {"target_lang": "中文(简体)", "ocr_lang": "japan", "vertical_support": True}),
    ("英语视频 → 中文", {"target_lang": "中文(简体)", "ocr_lang": "en", "vertical_support": False}),
    ("韩语视频 → 中文", {"target_lang": "中文(简体)", "ocr_lang": "korean", "vertical_support": False}),
    ("中文视频 → English", {"target_lang": "English", "ocr_lang": "ch", "vertical_support": False}),
    ("自定义(跳过预设)", {}),
]


def _hint(t):
    lb = QLabel(t)
    lb.setProperty("hint", True)
    lb.setWordWrap(True)
    return lb


class _ApiPage(QWidget):
    def __init__(self, title, ep, vision, local_tag, parent=None):
        super().__init__(parent)
        from gui.endpoint_v6 import EndpointEditor
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b style='font-size:15px'>{title}</b>"))
        item = mm.find_item(tag=local_tag)
        size = f", 约 {float(item.get('size_gb', 0)):.1f} GB" if item else ""
        installed = bool(item and mm.is_installed(item))
        btn = QPushButton(f"使用推荐本地模型 {local_tag}(免费{size}{', 已安装' if installed else ''})")
        btn.setProperty("accent", True)
        layout.addWidget(btn)
        layout.addWidget(_hint("免 API Key, 数据不出本机。模型未下载时, 向导结束后可自动排队下载。"
                               "也可以在下面改用云端 API(需填 Key 并测试通过)。"))
        self.editor = EndpointEditor(ep, vision, self)
        layout.addWidget(self.editor)
        btn.clicked.connect(lambda: self.editor.set_values(ai.local_endpoint(local_tag)))

    @property
    def page_ok(self):
        if self.editor.tested_ok:
            return True
        try:
            return mm.is_managed(self.editor.value())
        except ValueError:
            return False

    def value(self):
        return self.editor.value()


class FirstRunWizard(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.download_now = False           # 主窗口据此决定向导结束后是否直接排队下载本地模型
        self.setWindowTitle("欢迎使用 · 首次配置")
        self.setMinimumSize(640, 600)
        v = QVBoxLayout(self)
        v.setSpacing(10)
        self.lbl_step = QLabel("")
        self.lbl_step.setProperty("hint", True)
        v.addWidget(self.lbl_step)
        self.stack = QStackedWidget()

        p1 = QWidget()
        v1 = QVBoxLayout(p1)
        v1.setSpacing(10)
        v1.addWidget(QLabel("<b style='font-size:15px'>选择使用场景</b>(自动配好识别语言/竖排等)"))
        self.cmb_scene = QComboBox()
        for name, _ in PRESETS:
            self.cmb_scene.addItem(name)
        v1.addWidget(self.cmb_scene)
        v1.addWidget(_hint("流程: 找到字幕位置 → 读出文字 → AI 翻译 → 处理原字幕并贴上译文 → 导出。全程可在[设置]调整。"))
        v1.addStretch(1)
        self.stack.addWidget(p1)

        mt_tag = cfg.ep_translate.get("ollama_model") or DEFAULT_MT_TAG
        vlm_tag = cfg.ep_vlm.get("ollama_model") or DEFAULT_VLM_TAG
        self.pg_trans = _ApiPage("翻译引擎", cfg.ep_translate, False, mt_tag)
        self.pg_vlm = _ApiPage("读字引擎(看画面认字)", cfg.ep_vlm, True, vlm_tag)
        self.chk_download = QCheckBox("完成后自动下载所选的本地模型(需联网; 云端 API 配置不受影响)")
        self.chk_download.setChecked(True)
        self.pg_vlm.layout().addWidget(self.chk_download)
        self.stack.addWidget(self.pg_trans)
        self.stack.addWidget(self.pg_vlm)
        v.addWidget(self.stack, 1)

        hb = QHBoxLayout()
        self.btn_skip = QPushButton("跳过(稍后在设置中配置)")
        self.btn_prev = QPushButton("← 上一步")
        self.btn_next = QPushButton("下一步 →")
        self.btn_next.setProperty("accent", True)
        hb.addWidget(self.btn_skip)
        hb.addStretch(1)
        hb.addWidget(self.btn_prev)
        hb.addWidget(self.btn_next)
        v.addLayout(hb)
        self.btn_skip.clicked.connect(self.reject)
        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)
        self._sync()

    def _sync(self):
        i = self.stack.currentIndex()
        self.lbl_step.setText(f"第 {i + 1} / 3 步")
        self.btn_prev.setEnabled(i > 0)
        self.btn_next.setText("完成 ✔" if i == 2 else "下一步 →")

    def _prev(self):
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._sync()

    def _next(self):
        i = self.stack.currentIndex()
        if i == 1 and not self.pg_trans.page_ok:
            QMessageBox.warning(self, "还差一步", "请点[使用推荐本地模型], 或配置 API 并测试通过。")
            return
        if i == 2:
            if not self.pg_vlm.page_ok:
                QMessageBox.warning(self, "还差一步", "请点[使用推荐本地模型], 或配置 API 并测试通过。")
                return
            self._apply()
            self.accept()
            return
        self.stack.setCurrentIndex(i + 1)
        self._sync()

    def _apply(self):
        c = self.cfg
        _, preset = PRESETS[self.cmb_scene.currentIndex()]
        for k, v in preset.items():
            setattr(c, k, v)
        c.ep_translate = self.pg_trans.value()
        c.ep_vlm = self.pg_vlm.value()
        c.ocr_engine = "vlm"                 # 与默认一致: 质量优先, 全 VLM 读字
        c.save()
        self.download_now = self.chk_download.isChecked() and (
            mm.is_managed(c.ep_translate) or mm.is_managed(c.ep_vlm))

    def done(self, result):
        running = [t for t in self.findChildren(QThread) if t.isRunning()]
        if running:
            for t in running:
                if hasattr(t, "stop"):
                    t.stop()
                t.requestInterruption()
            QMessageBox.information(self, "请稍候", "正在等待网络请求结束…")
            return
        super().done(result)
