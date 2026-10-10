from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from core import ai


class Job(QThread):
    result = Signal(bool, object)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function
        self.finished.connect(self.deleteLater)

    def run(self):
        try:
            self.result.emit(True, self.function(self.isInterruptionRequested))
        except Exception as exc:
            self.result.emit(False, str(exc))


def _row_visible(form, field, visible):
    label = form.labelForField(field)
    if label is not None:
        label.setVisible(visible)
    field.setVisible(visible)


class EndpointEditor(QWidget):
    """端点编辑器: 只显示所选接入协议相关的字段(选 Ollama 的人不会再看到 API Key 一栏);
    供应商预设来自 core.ai.api_presets(与设置页/向导同一数据源)。"""

    def __init__(self, ep, vision=False, parent=None, timeout=120):
        super().__init__(parent)
        self.vision = vision
        self.timeout = int(timeout)
        self._original = deepcopy(ep)
        self._job = None
        self._passed = None

        root = QVBoxLayout(self)
        self.form = QFormLayout()
        self.provider = QComboBox()
        providers = [("OpenAI 兼容 API(云端或自建)", "api"), ("Ollama(模型库托管 / 自装)", "ollama")]
        if not vision:
            providers.append(("本地文本 GGUF(需要 llama-cpp-python)", "local"))
        for title, value in providers:
            self.provider.addItem(title, value)
        self.provider.setCurrentIndex(max(0, self.provider.findData(ep.get("provider", "api"))))

        self.supplier = QComboBox()
        self.supplier.addItem("(选择供应商预设, 自动填地址与模型名)")
        self._presets = ai.api_presets(vision)
        for name, _base, _model in self._presets:
            self.supplier.addItem(name)
        self.base = QLineEdit(ep.get("api_base", ""))
        self.key = QLineEdit(ep.get("api_key", ""))
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("可填 env:DEEPSEEK_API_KEY; 明文 Key 保存时会自动移入系统凭据库")
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setCurrentText(ep.get("api_model", ""))
        self.ourl = QLineEdit(ep.get("ollama_url", "http://localhost:11434"))
        self.omodel = QLineEdit(ep.get("ollama_model", ""))
        self.local = QLineEdit(ep.get("local_model", ""))
        browse = QPushButton("选择 GGUF")
        browse.clicked.connect(self._pick)
        self.local_box = QWidget()
        row = QHBoxLayout(self.local_box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.local)
        row.addWidget(browse)
        self.extra = QPlainTextEdit(json.dumps(ep.get("api_extra") or {}, ensure_ascii=False, indent=2))
        self.extra.setMaximumHeight(90)
        for title, widget in (("接入协议", self.provider), ("供应商预设", self.supplier),
                              ("API Base URL", self.base), ("API Key / env 引用", self.key),
                              ("API 模型", self.model), ("Ollama 地址", self.ourl),
                              ("Ollama 模型", self.omodel), ("本地文本模型", self.local_box),
                              ("API 附加 JSON", self.extra)):
            self.form.addRow(title, widget)
        root.addLayout(self.form)

        buttons = QHBoxLayout()
        self.fetch = QPushButton("获取模型列表")
        self.test = QPushButton("测试实际能力")
        self.status = QLabel("未测试")
        self.status.setWordWrap(True)
        buttons.addWidget(self.fetch)
        buttons.addWidget(self.test)
        root.addLayout(buttons)
        root.addWidget(self.status)
        self.fetch.clicked.connect(self._fetch)
        self.test.clicked.connect(self._test)
        self.supplier.activated.connect(self._preset)
        for edit in (self.base, self.key, self.ourl, self.omodel, self.local):
            edit.textChanged.connect(self._invalidate)
        self.model.currentTextChanged.connect(self._invalidate)
        self.provider.currentIndexChanged.connect(self._invalidate)
        self.provider.currentIndexChanged.connect(self._apply_visibility)
        self.extra.textChanged.connect(self._invalidate)
        self._apply_visibility()

    def _apply_visibility(self, *_):
        p = self.provider.currentData()
        for w in (self.supplier, self.base, self.key, self.model, self.extra):
            _row_visible(self.form, w, p == "api")
        for w in (self.ourl, self.omodel):
            _row_visible(self.form, w, p == "ollama")
        _row_visible(self.form, self.local_box, p == "local")
        self.fetch.setVisible(p in ("api", "ollama"))

    @property
    def tested_ok(self):
        try:
            return self._passed == self._signature()
        except ValueError:
            return False

    def _signature(self):
        value = json.dumps(self.value(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def _invalidate(self, *args):
        self._passed = None
        self.status.setText("配置已变化, 需要重新测试")

    def _preset(self, index):
        if index <= 0:
            return
        _, base, model = self._presets[index - 1]
        if self.base.text().strip().rstrip("/") != base.rstrip("/"):
            self.key.clear()                 # 跨供应商时不把旧 Key 带到新供应商
        self.provider.setCurrentIndex(self.provider.findData("api"))
        self.base.setText(base)
        self.model.setCurrentText(model)
        self._invalidate()

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择文本 GGUF", "", "GGUF (*.gguf)")
        if path:
            self.local.setText(path)

    def value(self):
        value = deepcopy(self._original)
        extra = json.loads(self.extra.toPlainText().strip() or "{}")
        if not isinstance(extra, dict):
            raise ValueError("API 附加参数必须为 JSON 对象")
        value.update(provider=self.provider.currentData(), api_base=self.base.text().strip(),
                     api_key=self.key.text().strip(), api_model=self.model.currentText().strip(),
                     ollama_url=self.ourl.text().strip(), ollama_model=self.omodel.text().strip(),
                     local_model=self.local.text().strip(), api_extra=extra)
        return value

    def set_values(self, ep):
        self.provider.setCurrentIndex(max(0, self.provider.findData(ep.get("provider", "api"))))
        if "api_base" in ep:
            self.base.setText(ep.get("api_base", ""))
        if "api_key" in ep:
            self.key.setText(ep.get("api_key", ""))
        if "api_model" in ep:
            self.model.setCurrentText(ep.get("api_model", ""))
        if "ollama_url" in ep:
            self.ourl.setText(ep.get("ollama_url", self.ourl.text()))
        if "ollama_model" in ep:
            self.omodel.setText(ep.get("ollama_model", ""))
        if "local_model" in ep:
            self.local.setText(ep.get("local_model", ""))
        self._apply_visibility()
        self._invalidate()

    def _start(self, function, callback):
        if self._job is not None:
            return
        self.fetch.setEnabled(False)
        self.test.setEnabled(False)
        self.status.setText("处理中…")
        self._job = Job(function, self)
        self._job.result.connect(callback)
        self._job.finished.connect(self._finished)
        self._job.start()

    def _finished(self):
        self._job = None
        self.fetch.setEnabled(True)
        self.test.setEnabled(True)

    def _test(self):
        try:
            ep = self.value()
            signature = self._signature()
        except ValueError as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return

        def result(ok, message):
            self.status.setText(str(message))
            if ok and signature == self._signature():
                self._passed = signature

        self._start(lambda abort: ai.test_endpoint(ep, self.vision, abort=abort, timeout=self.timeout), result)

    def _fetch(self):
        try:
            ep = self.value()
        except ValueError as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return

        def result(ok, values):
            if not ok:
                self.status.setText(str(values))
                return
            from gui.settings_dialog import ListPickDialog
            current = self.omodel.text() if ep["provider"] == "ollama" else self.model.currentText()
            dialog = ListPickDialog("选择模型", values, current, self)
            if dialog.exec() and dialog.value():
                if ep["provider"] == "ollama":
                    self.omodel.setText(dialog.value())
                else:
                    self.model.setCurrentText(dialog.value())
            self.status.setText("模型列表获取完成; 列表不代表具备视觉能力, 请再[测试实际能力]")

        self._start(lambda abort: ai.list_models(ep), result)
