from copy import deepcopy

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPlainTextEdit, QPushButton, QRadioButton,
                               QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from core import ai, hardware, inpaint, prompts, workspace
from core import model_manager as mm
from core.workers import HardwareProbeWorker
from gui.endpoint_v6 import EndpointEditor
from gui.locator_settings import LocatorSettingsWidget
from gui.theme import apply_theme
from gui.widgets import CollapsibleBox

LANGS = ["中文(简体)", "中文(繁体)", "English", "日本語", "한국어",
         "Français", "Deutsch", "Español", "Русский"]
OCR_LANGS = [("中英混合", "ch"), ("英文", "en"), ("日文", "japan"), ("韩文", "korean")]
OCR_ENGINES = [("全部用 AI 视觉模型读字(推荐, 质量最好)", "vlm"),
               ("智能: 本地先认, 看不清交给 AI(省调用)", "auto"),
               ("只用本地识别(完全免费, 质量一般)", "local")]

_WS = QFontDatabase.WritingSystem
LANG_WS = {"中文(简体)": _WS.SimplifiedChinese, "中文(繁体)": _WS.TraditionalChinese,
           "日本語": _WS.Japanese, "한국어": _WS.Korean}


def font_covers(family, target_lang):
    ws = LANG_WS.get(target_lang)
    if ws is None:
        return True
    try:
        return ws in QFontDatabase.writingSystems(family)
    except Exception:
        return True


def _hint(text):
    lb = QLabel(text)
    lb.setProperty("hint", True)
    lb.setWordWrap(True)
    return lb


def _scrolled(page):
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    sa.setWidget(page)
    return sa


def _set_data(combo, value):
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return


def _fmt_bytes(n):
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    return f"{n / (1 << 20):.0f} MB"


def _presets(kind, vision):
    """预设的唯一数据源: 模型库里的本地模型 + core.ai 的 API 预设。"""
    out = []
    for it in mm.load_registry():
        if it.get("kind") == kind and it.get("tag"):
            state = "已安装" if mm.is_installed(it) else "未安装, 到[模型库]下载"
            out.append((f"本地 · {it['name']}({state})", ai.local_endpoint(it["tag"])))
    for name, base, model in ai.api_presets(vision):
        tail = "" if "127.0.0.1" in base or "localhost" in base else "(需填 Key)"
        out.append((f"云端 · {name}{tail}", {"provider": "api", "api_base": base, "api_model": model}))
    return out


class ListPickDialog(QDialog):
    def __init__(self, title, items, current="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(440, 500)
        v = QVBoxLayout(self)
        self.ed = QLineEdit()
        self.ed.setPlaceholderText("输入关键字筛选...")
        v.addWidget(self.ed)
        self.lst = QListWidget()
        v.addWidget(self.lst, 1)
        self.lbl = QLabel("")
        self.lbl.setProperty("hint", True)
        v.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._all = list(items)
        self._current = current
        self.ed.textChanged.connect(self._fill)
        self.lst.itemDoubleClicked.connect(lambda *_: self.accept())
        self._fill()

    def _fill(self):
        kw = self.ed.text().strip().lower()
        self.lst.clear()
        for it in self._all:
            if not kw or kw in it.lower():
                self.lst.addItem(it)
        self.lbl.setText(f"{self.lst.count()} / {len(self._all)} 项")
        hits = self.lst.findItems(self._current, Qt.MatchFlag.MatchExactly) if self._current else []
        if hits:
            self.lst.setCurrentItem(hits[0])
        elif self.lst.count():
            self.lst.setCurrentRow(0)

    def value(self):
        it = self.lst.currentItem()
        return it.text() if it else ""


class FontPickDialog(ListPickDialog):
    def __init__(self, target_lang, current="", parent=None):
        self._fams = sorted(QFontDatabase.families())
        self._lang = target_lang
        super().__init__("选择字幕字体", self._fams, current, parent)
        self.chk = QCheckBox(f"只显示支持 [{target_lang}] 的字体")
        self.chk.setChecked(True)
        self.layout().insertWidget(1, self.chk)
        self.chk.toggled.connect(self._refilter)
        self._refilter()

    def _refilter(self):
        self._all = [f for f in self._fams if font_covers(f, self._lang)] \
            if self.chk.isChecked() else list(self._fams)
        self._fill()


def _advanced_box(editor, ep):
    box = CollapsibleBox("高级: 接口地址 / 模型名 / API Key(选本地预设无需修改)",
                         expanded=ep.get("provider") == "api")
    lay = QVBoxLayout()
    lay.addWidget(editor)
    box.set_content_layout(lay)
    return box


def _preset_row(presets, editor, box):
    row = QHBoxLayout()
    cmb = QComboBox()
    for name, _ in presets:
        cmb.addItem(name)
    cmb.view().setMinimumWidth(460)
    btn = QPushButton("应用预设")
    row.addWidget(QLabel("快速预设:"))
    row.addWidget(cmb, 1)
    row.addWidget(btn)

    def apply():
        _, ep = presets[cmb.currentIndex()]
        editor.set_values(ep)
        box.set_expanded(ep.get("provider") == "api")

    btn.clicked.connect(apply)
    return row


class SettingsDialog(QDialog):
    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._original_cfg = cfg
        self.cfg = deepcopy(cfg)
        self._hw_pending = None
        self._hw_worker = None
        self._release_cache = getattr(parent, "release_erased_preview", None)
        self.setWindowTitle("设置 · 水仙翻译器")
        self.resize(1000, 760)
        self.setMinimumSize(760, 520)
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        navigation = QListWidget()
        navigation.setFixedWidth(150)
        pages = QStackedWidget()
        self.loc_settings = LocatorSettingsWidget(self.cfg)
        content = [
            ("翻译引擎", self._tab_translate()),
            ("读字 OCR", self._tab_ocr()),
            ("定位与时间", self.loc_settings),
            ("原字处理 / 遮罩", self._tab_erase()),
            ("输出与导出", self._tab_output()),
            ("提示词", self._tab_prompts()),
            ("杂项", self._tab_misc()),
        ]
        for title, widget in content:
            navigation.addItem(title)
            pages.addWidget(_scrolled(widget))
        navigation.currentRowChanged.connect(pages.setCurrentIndex)
        navigation.setCurrentRow(0)
        body.addWidget(navigation)
        body.addWidget(pages, 1)
        root.addLayout(body, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---------- 翻译 ----------
    def _tab_translate(self):
        cfg = self.cfg
        page = QWidget()
        v = QVBoxLayout(page)
        self.ep_t = EndpointEditor(cfg.ep_translate, vision=False, timeout=min(180, cfg.request_timeout))
        self.box_t = _advanced_box(self.ep_t, cfg.ep_translate)
        v.addLayout(_preset_row(_presets("ollama_llm", False), self.ep_t, self.box_t))
        f = QFormLayout()
        self.cmb_lang = QComboBox()
        self.cmb_lang.setEditable(True)
        self.cmb_lang.addItems(LANGS)
        self.cmb_lang.setCurrentText(cfg.target_lang)
        f.addRow("目标语言:", self.cmb_lang)
        self.sp_batch = QSpinBox()
        self.sp_batch.setRange(1, 30)
        self.sp_batch.setValue(cfg.batch_size)
        self.sp_batch.setToolTip("一次交给模型翻译几句。本地 CPU 跑得慢时调小。")
        f.addRow("批量条数:", self.sp_batch)
        self.chk_ctx = QCheckBox("翻译时参考前两句(对话更连贯)")
        self.chk_ctx.setChecked(cfg.ctx_translate)
        f.addRow("", self.chk_ctx)
        self.sp_timeout = QSpinBox()
        self.sp_timeout.setRange(30, 3600)
        self.sp_timeout.setSingleStep(30)
        self.sp_timeout.setSuffix(" 秒")
        self.sp_timeout.setValue(int(cfg.request_timeout))
        self.sp_timeout.setToolTip("单次 AI 请求等待上限(翻译与读字共用)。本地模型纯 CPU 跑一批可能要几分钟, 请放宽; 随时可取消。")
        f.addRow("单次请求超时:", self.sp_timeout)
        v.addLayout(f)
        v.addWidget(_hint("翻译结束后会自动检查漏译的条目并逐条补译一次。"))
        v.addWidget(self.box_t)
        v.addStretch(1)
        return page

    # ---------- 读字 ----------
    def _tab_ocr(self):
        cfg = self.cfg
        page = QWidget()
        v = QVBoxLayout(page)
        f = QFormLayout()
        self.cmb_ocr1 = QComboBox()
        for name, code in OCR_ENGINES:
            self.cmb_ocr1.addItem(name, code)
        _set_data(self.cmb_ocr1, cfg.ocr_engine)
        self.cmb_ocr1.view().setMinimumWidth(420)
        f.addRow("读字方式:", self.cmb_ocr1)
        self.cmb_ocrlang = QComboBox()
        for name, code in OCR_LANGS:
            self.cmb_ocrlang.addItem(name, code)
        _set_data(self.cmb_ocrlang, cfg.ocr_lang)
        self.cmb_ocrlang.setToolTip("本地识别用的语言模型; 选错会认成乱码, 比如日语视频必须选日文。")
        f.addRow("视频原文语言:", self.cmb_ocrlang)
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.10, 0.95)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(cfg.conf_threshold)
        f.addRow("本地读不准时求助 AI 的门槛:", self.sp_conf)
        self.chk_vert = QCheckBox("自动定位阶段支持竖排字幕(手动框选的竖框不受此项影响)")
        self.chk_vert.setChecked(cfg.vertical_support)
        f.addRow("", self.chk_vert)
        self.v6_budget = QSpinBox()
        self.v6_budget.setRange(1, 100000)
        self.v6_budget.setValue(cfg.ocr_request_budget)
        f.addRow("每视频 AI 读字次数上限:", self.v6_budget)
        self.v6_fallback = QCheckBox("AI 读字不可用/失败时改用本地识别")
        self.v6_fallback.setChecked(cfg.ocr_local_fallback)
        f.addRow("", self.v6_fallback)
        v.addLayout(f)
        g = QGroupBox("AI 读字引擎")
        gv = QVBoxLayout(g)
        self.ep_v = EndpointEditor(cfg.ep_vlm, vision=True, timeout=min(180, cfg.request_timeout))
        self.box_v = _advanced_box(self.ep_v, cfg.ep_vlm)
        gv.addLayout(_preset_row(_presets("ollama_vlm", True), self.ep_v, self.box_v))
        fw = QFormLayout()
        self.sp_vw = QSpinBox()
        self.sp_vw.setRange(480, 1600)
        self.sp_vw.setSingleStep(80)
        self.sp_vw.setValue(cfg.vlm_detail_width)
        fw.addRow("发给 AI 的图片宽度(px):", self.sp_vw)
        gv.addLayout(fw)
        gv.addWidget(self.box_v)
        v.addWidget(g)
        v.addStretch(1)
        return page

    # ---------- 原字处理 / 遮罩 ----------
    def _tab_erase(self):
        cfg = self.cfg
        page = QWidget()
        v = QVBoxLayout(page)
        g0 = QGroupBox("原字幕处理方式(顶部 04 勾选时生效)")
        g0.setToolTip("单独点[04]按钮=只在工作区内处理, 不生成文件; 所有文件都在[05 导出]时产出。")
        v0 = QVBoxLayout(g0)
        self.rb_erase = QRadioButton("无痕擦除: AI 抹掉原字并补全背景, 译文放在原位(推荐)")
        self.rb_cover = QRadioButton("遮罩覆盖: 按下方遮罩样式盖住原字, 译文放在原位")
        self.rb_below = QRadioButton("保留原字幕: 不处理原字, 译文放在原字幕下方")
        {"erase": self.rb_erase, "cover": self.rb_cover, "below": self.rb_below}[cfg.placement].setChecked(True)
        for rb in (self.rb_erase, self.rb_cover, self.rb_below):
            v0.addWidget(rb)
            rb.toggled.connect(self._placement_changed)
        v.addWidget(g0)

        self.g_engine = QGroupBox("智能擦除(处理方式为[无痕擦除]时使用)")
        f = QFormLayout(self.g_engine)
        self.cmb_erase = QComboBox()
        for name, code in inpaint.BASIC_ENGINES:
            self.cmb_erase.addItem(name, code)
        _set_data(self.cmb_erase, cfg.erase_engine)
        f.addRow("擦除引擎:", self.cmb_erase)
        self.sp_expand = QSpinBox()
        self.sp_expand.setRange(0, 30)
        self.sp_expand.setValue(int(round(cfg.erase_expand_ratio * 100)))
        self.sp_expand.setToolTip("擦除范围在定位框基础上绕框心按比例外扩, 防止原字边缘被漏擦。")
        f.addRow("擦除范围外扩(%):", self.sp_expand)
        self.sp_dilate = QSpinBox()
        self.sp_dilate.setRange(0, 40)
        self.sp_dilate.setValue(cfg.erase_dilate)
        self.sp_dilate.setToolTip("擦除区域向外多扩几个像素, 盖住描边/阴影残留。")
        f.addRow("擦除范围外扩(像素):", self.sp_dilate)
        self.sp_lworkers = QSpinBox()
        self.sp_lworkers.setRange(1, 4)
        self.sp_lworkers.setValue(int(getattr(cfg, "erase_lama_workers", 1) or 1))
        self.sp_lworkers.setToolTip("LAMA 并行子进程数: 每个进程约多占 1.5~2GB 显存; GPU 有余量时 2 个约提速 1.8~2 倍。默认 1。")
        f.addRow("LAMA 并行进程数:", self.sp_lworkers)
        self.sp_lbatch = QSpinBox()
        self.sp_lbatch.setRange(1, 8)
        self.sp_lbatch.setValue(int(getattr(cfg, "erase_lama_batch", 1) or 1))
        self.sp_lbatch.setToolTip("LAMA 批量推理: 同尺寸帧按此数量合成一个张量一次推理。GPU 被其他程序占用时收益不稳定, 默认 1(逐帧)。")
        f.addRow("LAMA 每批帧数:", self.sp_lbatch)
        self.sp_lama_side = QSpinBox()
        self.sp_lama_side.setRange(0, 4096)
        self.sp_lama_side.setSingleStep(128)
        self.sp_lama_side.setSpecialValueText("不限")
        self.sp_lama_side.setValue(int(getattr(cfg, "erase_lama_max_side", 1920) or 0))
        self.sp_lama_side.setToolTip("LAMA 输入 ROI 的长边上限: 4K 字幕条降到 1080p 推理再回贴, 约 3~4 倍提速; 0=不限。")
        f.addRow("LAMA 处理分辨率上限(长边 px, 0=不限):", self.sp_lama_side)
        btn_chk = QPushButton("检查擦除引擎是否可用")

        def do_chk():
            self._collect_erase()
            ok, msg = inpaint.check_engine(self.cfg)
            (QMessageBox.information if ok else QMessageBox.warning)(self, "擦除检查", msg)
        btn_chk.clicked.connect(do_chk)
        f.addRow("", btn_chk)
        adv = CollapsibleBox("高级: 视频级引擎(需自备 repo/权重) / 自建服务 / 独立 Python")
        fa = QFormLayout()
        self.cmb_erase_adv = QComboBox()
        self.cmb_erase_adv.addItem("(不使用高级引擎)", "")
        for name, code in inpaint.ADVANCED_ENGINES:
            self.cmb_erase_adv.addItem(name, code)
        if cfg.erase_engine in {c for _, c in inpaint.ADVANCED_ENGINES}:
            _set_data(self.cmb_erase_adv, cfg.erase_engine)
            adv.set_expanded(True)
        fa.addRow("高级引擎(选中后覆盖上面的选择):", self.cmb_erase_adv)
        self.sp_chunk = QSpinBox()
        self.sp_chunk.setRange(16, 200)
        self.sp_chunk.setValue(cfg.erase_chunk)
        fa.addRow("每段帧数:", self.sp_chunk)
        row = QHBoxLayout()
        self.ed_epython = QLineEdit(cfg.erase_python)
        self.ed_epython.setPlaceholderText("视频级引擎专用, 一般留空")
        bp = QPushButton("浏览...")
        bp.clicked.connect(lambda: self._pick_file(self.ed_epython))
        row.addWidget(self.ed_epython)
        row.addWidget(bp)
        fa.addRow("引擎 Python 路径:", row)
        self.ed_eapi = QLineEdit(cfg.erase_api_url)
        self.ed_eapi.setPlaceholderText("IOPaint 兼容, 如 http://127.0.0.1:8080/api/v1/inpaint")
        fa.addRow("自建服务地址:", self.ed_eapi)
        adv.set_content_layout(fa)
        f.addRow(adv)
        v.addWidget(self.g_engine)

        g2 = QGroupBox("遮罩样式(遮罩覆盖 与 画面上拖出的遮挡块 共用)")
        f2 = QFormLayout(g2)
        self.cmb_mask = QComboBox()
        self.cmb_mask.addItem("半透明渐变字幕条(推荐)", "bar")
        self.cmb_mask.addItem("智能底色(按周围背景色填满, 完全盖住)", "bg")
        self.cmb_mask.addItem("局部模糊 + 轻微压暗(柔和, 嵌字可读)", "blur")
        self.cmb_mask.addItem("纯色填充", "solid")
        _set_data(self.cmb_mask, cfg.mask_style)
        f2.addRow("遮罩样式:", self.cmb_mask)
        self.sp_alpha = QSpinBox()
        self.sp_alpha.setRange(40, 70)
        self.sp_alpha.setSuffix(" %")
        self.sp_alpha.setValue(int(cfg.mask_bar_alpha))
        f2.addRow("字幕条深浅:", self.sp_alpha)
        self.sp_blur = QSpinBox()
        self.sp_blur.setRange(15, 45)
        self.sp_blur.setValue(int(cfg.mask_blur))
        f2.addRow("模糊强度:", self.sp_blur)
        v.addWidget(g2)

        g3 = QGroupBox("擦除缓存(工作区 .work/erased/)")
        f3 = QFormLayout(g3)
        self.lbl_cache = QLabel(f"当前占用 {_fmt_bytes(workspace.erased_usage())}")
        btn_clear = QPushButton("清空擦除缓存")
        btn_clear.clicked.connect(self._clear_cache)
        rowc = QHBoxLayout()
        rowc.addWidget(self.lbl_cache, 1)
        rowc.addWidget(btn_clear)
        f3.addRow("占用:", rowc)
        self.sp_cache_days = QSpinBox()
        self.sp_cache_days.setRange(1, 30)
        self.sp_cache_days.setSuffix(" 天")
        self.sp_cache_days.setValue(int(cfg.erase_cache_days))
        f3.addRow("自动清理早于:", self.sp_cache_days)
        self.chk_direct = QCheckBox("队列导出时一次完成擦除+嵌字(单次编码, 更快更省磁盘; 不生成可预览的擦除缓存)")
        self.chk_direct.setChecked(bool(cfg.export_direct))
        f3.addRow("", self.chk_direct)
        v.addWidget(g3)
        v.addStretch(1)
        self._placement_changed()
        return page

    def _placement_changed(self, *_):
        self.g_engine.setEnabled(self.rb_erase.isChecked())

    def _clear_cache(self):
        if QMessageBox.question(self, "清空擦除缓存",
                                "删除所有视频在工作区里的擦除画面? 下次导出会重新擦除。") \
                != QMessageBox.StandardButton.Yes:
            return
        if callable(self._release_cache):
            self._release_cache()
        n = workspace.clear_erased()
        self.lbl_cache.setText(f"已删除 {n} 个文件 · 当前占用 {_fmt_bytes(workspace.erased_usage())}")

    def _pick_file(self, edit):
        p, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if p:
            edit.setText(p)

    def _collect_erase(self):
        c = self.cfg
        c.placement = "erase" if self.rb_erase.isChecked() else ("cover" if self.rb_cover.isChecked() else "below")
        c.erase_engine = self.cmb_erase_adv.currentData() or self.cmb_erase.currentData()
        c.erase_dilate = self.sp_dilate.value()
        c.erase_expand_ratio = self.sp_expand.value() / 100.0
        c.erase_chunk = self.sp_chunk.value()
        c.erase_lama_workers = self.sp_lworkers.value()
        c.erase_lama_batch = self.sp_lbatch.value()
        c.erase_lama_max_side = self.sp_lama_side.value()
        c.erase_python = self.ed_epython.text().strip()
        c.erase_api_url = self.ed_eapi.text().strip()
        c.mask_style = self.cmb_mask.currentData()
        c.mask_bar_alpha = self.sp_alpha.value()
        c.mask_blur = self.sp_blur.value()
        c.erase_cache_days = self.sp_cache_days.value()
        c.export_direct = self.chk_direct.isChecked()

    # ---------- 提示词 ----------
    def _tab_prompts(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_hint("给 AI 的指令模板, 一般不需要改。清空即恢复默认。"))
        self._pkeys = list(prompts.DEFAULTS.keys())
        self._pvals = {k: ((self.cfg.prompts or {}).get(k) or prompts.DEFAULTS[k]) for k in self._pkeys}
        self._pcur = self._pkeys[0]
        self.cmb_pk = QComboBox()
        for k in self._pkeys:
            self.cmb_pk.addItem(prompts.TITLES.get(k, k), k)
        v.addWidget(self.cmb_pk)
        self.lbl_ph = _hint(prompts.HINTS.get(self._pcur, ""))
        v.addWidget(self.lbl_ph)
        self.ed_prompt = QPlainTextEdit(self._pvals[self._pcur])
        self.ed_prompt.setMinimumHeight(240)
        v.addWidget(self.ed_prompt, 1)
        b_reset = QPushButton("恢复此条默认")
        b_reset.clicked.connect(lambda: self.ed_prompt.setPlainText(prompts.DEFAULTS[self._pcur]))
        v.addWidget(b_reset)
        self.cmb_pk.currentIndexChanged.connect(self._prompt_switch)
        return page

    def _prompt_switch(self):
        self._pvals[self._pcur] = self.ed_prompt.toPlainText()
        self._pcur = self.cmb_pk.currentData()
        self.ed_prompt.setPlainText(self._pvals[self._pcur])
        self.lbl_ph.setText(prompts.HINTS.get(self._pcur, ""))

    # ---------- 输出 ----------
    def _tab_output(self):
        cfg = self.cfg
        page = QWidget()
        v = QVBoxLayout(page)
        g = QGroupBox("输出文件(只在 05 导出 时产出)")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.ed_outdir = QLineEdit(cfg.out_dir)
        self.ed_outdir.setPlaceholderText("留空 = 和原视频放一起")
        bo = QPushButton("浏览...")

        def pick_dir():
            d = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if d:
                self.ed_outdir.setText(d)
        bo.clicked.connect(pick_dir)
        row.addWidget(self.ed_outdir)
        row.addWidget(bo)
        f.addRow("输出目录:", row)
        self.ed_suffix = QLineEdit(cfg.out_suffix)
        f.addRow("文件名后缀:", self.ed_suffix)
        self.cmb_overwrite = QComboBox()
        self.cmb_overwrite.addItem("自动改名, 如 xxx (2).mp4(推荐)", "rename")
        self.cmb_overwrite.addItem("直接覆盖", "overwrite")
        _set_data(self.cmb_overwrite, cfg.out_overwrite)
        f.addRow("输出同名文件时:", self.cmb_overwrite)
        self.chk_subs = QCheckBox("导出成品时同时生成字幕文件(与成品同名)")
        self.chk_subs.setChecked(cfg.pipe_subs)
        f.addRow("", self.chk_subs)
        self.cmb_subfmt = QComboBox()
        self.cmb_subfmt.addItem("ASS(推荐: 位置/字号/颜色/遮罩与画面一致)", "ass")
        self.cmb_subfmt.addItem("SRT(纯文本: 只有时间与文字, 播放器按默认样式显示)", "srt")
        _set_data(self.cmb_subfmt, cfg.subtitle_format)
        self.cmb_subfmt.view().setMinimumWidth(420)
        f.addRow("字幕文件格式:", self.cmb_subfmt)
        f.addRow("", _hint("预览区按所选格式显示字幕文件的实际效果: ASS 与画面排布完全一致; "
                           "SRT 不带位置与样式, 预览改为播放器默认样式(底部居中白字黑边, 同时出现的多条自下而上叠放)。"
                           "成品 MP4 内嵌的译文不受此项影响, 始终按画面排布。"))
        self.chk_bi = QCheckBox("双语字幕(译文 + 小号原文)")
        self.chk_bi.setChecked(cfg.out_bilingual)
        f.addRow("", self.chk_bi)
        self.chk_audio = QCheckBox("保留原视频声音(推荐勾选; 源为 AAC 时直接拷贝不重编)")
        self.chk_audio.setChecked(cfg.keep_audio)
        f.addRow("", self.chk_audio)
        self.sp_crf = QSpinBox()
        self.sp_crf.setRange(10, 28)
        self.sp_crf.setValue(cfg.erase_crf)
        self.sp_crf.setToolTip("x264 CRF: 数字越小越清晰、文件越大。16 接近无损观感, 20 适合分享。")
        f.addRow("成品画质(CRF, 越小越清晰):", self.sp_crf)
        self.cmb_encoder = QComboBox()
        self.cmb_encoder.addItem("自动(有 N 卡用 NVENC, 否则 libx264)", "auto")
        self.cmb_encoder.addItem("libx264(CPU, 兼容性最好)", "libx264")
        self.cmb_encoder.addItem("h264_nvenc(NVIDIA 显卡, 快数倍)", "h264_nvenc")
        self.cmb_encoder.addItem("h264_qsv(Intel 核显)", "h264_qsv")
        self.cmb_encoder.addItem("h264_amf(AMD 显卡)", "h264_amf")
        _set_data(self.cmb_encoder, getattr(cfg, "video_encoder", "auto"))
        f.addRow("成品编码器:", self.cmb_encoder)
        v.addWidget(g)

        g2 = QGroupBox("字幕样式")
        f2 = QFormLayout(g2)
        row_f = QHBoxLayout()
        self.ed_font = QLineEdit(cfg.subtitle_font)
        self.ed_font.setReadOnly(True)
        b_font = QPushButton("选择字体...")

        def pick_font():
            lang = self.cmb_lang.currentText().strip() or self.cfg.target_lang
            dlg = FontPickDialog(lang, self.ed_font.text(), self)
            if dlg.exec() and dlg.value():
                self.ed_font.setText(dlg.value())
                self._font_check()
        b_font.clicked.connect(pick_font)
        row_f.addWidget(self.ed_font)
        row_f.addWidget(b_font)
        f2.addRow("字幕字体:", row_f)
        self.lbl_fwarn = _hint("")
        f2.addRow("", self.lbl_fwarn)
        self.v6_fit = QComboBox()
        self.v6_fit.addItem("字号按框高自适应(竖排: 框宽), 宽度不限(推荐)", "contain")
        self.v6_fit.addItem("拉伸填满定位框(宽高都贴合)", "stretch")
        _set_data(self.v6_fit, cfg.layout_fit)
        f2.addRow("自动嵌字方式:", self.v6_fit)
        self.cmb_align = QComboBox()
        self.cmb_align.addItem("居中", "center")
        self.cmb_align.addItem("靠左(竖排: 靠上)", "left")
        self.cmb_align.addItem("靠右(竖排: 靠下)", "right")
        _set_data(self.cmb_align, cfg.subtitle_align)
        f2.addRow("译文在框内对齐:", self.cmb_align)
        f2.addRow("", _hint("定位框 = 字幕本体: 字号刚好占满框高、不越界、不缩小; 宽度不受限, 按这里的对齐方式放。"
                            "预览、成品 MP4、ASS 三者同一套排布; 拖动框即移动字幕。"))
        self.chk_autocolor = QCheckBox("嵌字颜色自动匹配原字幕颜色(准确度一般; 取色失败自动用安全色)")
        self.chk_autocolor.setChecked(bool(cfg.auto_text_color))
        f2.addRow("", self.chk_autocolor)
        self.chk_lockpos = QCheckBox("锁定字幕位置(强制收敛为中位框, 防止字幕乱动; 关掉可保留移动字幕的轨迹)")
        self.chk_lockpos.setChecked(bool(getattr(cfg, "subtitle_lock_position", True)))
        f2.addRow("", self.chk_lockpos)
        v.addWidget(g2)
        v.addStretch(1)
        self._font_check()
        return page

    def _font_check(self):
        fam = self.ed_font.text().strip()
        lang = self.cmb_lang.currentText().strip() if hasattr(self, "cmb_lang") else self.cfg.target_lang
        if not font_covers(fam, lang):
            self.lbl_fwarn.setText(f"[{fam}] 不支持 {lang}, 成品可能出现方块。推荐 Microsoft YaHei / 思源黑体")
            self.lbl_fwarn.setStyleSheet("color:#ffb454;")
        else:
            self.lbl_fwarn.setText("字体支持目标语言")
            self.lbl_fwarn.setStyleSheet("color:#7dd87d;")

    # ---------- 杂项 ----------
    def _tab_misc(self):
        cfg = self.cfg
        page = QWidget()
        v = QVBoxLayout(page)
        g = QGroupBox("界面与日志")
        f = QFormLayout(g)
        self.cmb_theme = QComboBox()
        self.cmb_theme.addItem("暗色(近黑深蓝, 默认)", "dark")
        self.cmb_theme.addItem("深蓝(更亮更蓝一档)", "blue")
        self.cmb_theme.addItem("亮色", "light")
        _set_data(self.cmb_theme, cfg.ui_theme)
        f.addRow("界面主题(保存后立即生效):", self.cmb_theme)
        self.chk_debug = QCheckBox("调试模式(生成诊断文件, 排查问题时再开)")
        self.chk_debug.setChecked(cfg.debug_mode)
        f.addRow("", self.chk_debug)
        self.chk_hwdec = QCheckBox("硬件解码(预览播放器 + 悬停截图取帧)")
        self.chk_hwdec.setChecked(bool(getattr(cfg, "hw_decode", True)))
        f.addRow("", self.chk_hwdec)
        self.sp_logmb = QSpinBox()
        self.sp_logmb.setRange(100, 4096)
        self.sp_logmb.setSuffix(" MB")
        self.sp_logmb.setValue(int(cfg.logs_max_mb))
        f.addRow("日志/临时文件占用上限:", self.sp_logmb)
        rowf = QHBoxLayout()
        self.ed_ffmpeg = QLineEdit(cfg.ffmpeg_path)
        self.ed_ffmpeg.setPlaceholderText("留空 = 自动(models/bin → 程序目录 → PATH; 没有则首次导出时自动下载)")
        bf = QPushButton("浏览...")
        bf.clicked.connect(lambda: self._pick_file(self.ed_ffmpeg))
        rowf.addWidget(self.ed_ffmpeg)
        rowf.addWidget(bf)
        f.addRow("FFmpeg 路径:", rowf)
        rowm = QHBoxLayout()
        self.ed_libmpv = QLineEdit(cfg.libmpv_path)
        self.ed_libmpv.setPlaceholderText("留空 = 自动(models/bin → 程序目录 → PATH; 没有则首次启动时自动下载)")
        bm = QPushButton("浏览...")
        bm.clicked.connect(lambda: self._pick_file(self.ed_libmpv))
        rowm.addWidget(self.ed_libmpv)
        rowm.addWidget(bm)
        f.addRow("libmpv 路径:", rowm)
        v.addWidget(g)

        g_hw = QGroupBox("硬件分级预设")
        fh = QFormLayout(g_hw)
        self.cmb_hw = QComboBox()
        for code, name in (("auto", "自动检测"), ("high", "高配"), ("mid", "中配"), ("low", "低配")):
            self.cmb_hw.addItem(name, code)
        _set_data(self.cmb_hw, cfg.hw_tier)
        fh.addRow("等级:", self.cmb_hw)
        self.btn_hw = QPushButton("检测硬件并套用预设(保存后生效)")
        self.btn_hw.clicked.connect(self._detect_hw)
        fh.addRow("", self.btn_hw)
        self.lbl_hw = _hint(cfg.hw_summary or "尚未检测")
        fh.addRow("", self.lbl_hw)
        fh.addRow("", _hint("预设会一次性改写 定位/读字/翻译/擦除/导出 的相关参数与本地模型型号, 之后仍可手动微调。"
                            "AMD/Intel 显卡不能给 LAMA 擦除加速, 按无独显分级; 觉得档位偏低可手动指定。"))
        v.addWidget(g_hw)
        v.addStretch(1)
        return page

    def _detect_hw(self):
        if self._hw_worker is not None:
            return
        self.btn_hw.setEnabled(False)
        self.lbl_hw.setText("正在探测硬件…")
        self._hw_worker = HardwareProbeWorker(self)
        self._hw_worker.probed.connect(self._hw_probed)
        self._hw_worker.start()

    def _hw_probed(self, info):
        self._hw_worker = None
        self.btn_hw.setEnabled(True)
        if not info:
            self.lbl_hw.setText("硬件探测失败")
            return
        sel = self.cmb_hw.currentData()
        tier = sel if sel in hardware.PRESETS else hardware.classify(info)
        self._hw_pending = (tier, hardware.summary(info, tier))
        self.lbl_hw.setText(self._hw_pending[1] + " · 点[保存]后套用")
    # ---------- 保存 ----------
    def _save(self):
        c = self.cfg
        self.loc_settings.apply(c)
        try:
            c.ep_translate = self.ep_t.value()
            c.ep_vlm = self.ep_v.value()
        except ValueError as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        # 翻译
        c.target_lang = self.cmb_lang.currentText().strip() or "中文(简体)"
        c.batch_size = self.sp_batch.value()
        c.ctx_translate = self.chk_ctx.isChecked()
        c.request_timeout = self.sp_timeout.value()
        # 读字
        c.ocr_engine = self.cmb_ocr1.currentData()
        c.ocr_lang = self.cmb_ocrlang.currentData()
        c.conf_threshold = round(self.sp_conf.value(), 2)
        c.vertical_support = self.chk_vert.isChecked()
        c.vlm_detail_width = self.sp_vw.value()
        c.ocr_request_budget = self.v6_budget.value()
        c.ocr_local_fallback = self.v6_fallback.isChecked()
        # 原字处理 / 遮罩 / 缓存
        self._collect_erase()
        # 提示词
        self._pvals[self._pcur] = self.ed_prompt.toPlainText()
        c.prompts = {k: v.strip() for k, v in self._pvals.items()
                     if v.strip() and v.strip() != prompts.DEFAULTS[k].strip()}
        # 输出
        c.out_dir = self.ed_outdir.text().strip()
        c.out_suffix = self.ed_suffix.text().strip() or "_translated"
        c.out_overwrite = self.cmb_overwrite.currentData()
        c.pipe_subs = self.chk_subs.isChecked()
        c.subtitle_format = self.cmb_subfmt.currentData()
        c.out_bilingual = self.chk_bi.isChecked()
        c.keep_audio = self.chk_audio.isChecked()
        c.erase_crf = self.sp_crf.value()
        c.video_encoder = self.cmb_encoder.currentData()
        c.subtitle_font = self.ed_font.text().strip() or "Microsoft YaHei"
        c.layout_fit = self.v6_fit.currentData()
        c.subtitle_align = self.cmb_align.currentData()
        c.auto_text_color = self.chk_autocolor.isChecked()
        c.subtitle_lock_position = self.chk_lockpos.isChecked()
        # 杂项
        c.ui_theme = self.cmb_theme.currentData()
        c.debug_mode = self.chk_debug.isChecked()
        c.hw_decode = self.chk_hwdec.isChecked()
        c.logs_max_mb = self.sp_logmb.value()
        c.ffmpeg_path = self.ed_ffmpeg.text().strip()
        c.libmpv_path = self.ed_libmpv.text().strip()
        c.hw_tier = self.cmb_hw.currentData()
        if self._hw_pending:                       # 最后套用, 覆盖上面收集的相关字段
            tier, text = self._hw_pending
            hardware.apply_preset(c, tier)
            c.hw_applied = True
            c.hw_summary = text
        try:
            c.save()
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self._original_cfg.__dict__.update(deepcopy(c.__dict__))
        apply_theme(QApplication.instance(), c.ui_theme)
        self.accept()

    def done(self, result):
        running = [t for t in self.findChildren(QThread) if t.isRunning()]
        if running:
            for t in running:
                if hasattr(t, "stop"):
                    t.stop()
                t.requestInterruption()
            QMessageBox.information(self, "请稍候", "正在等待任务结束…")
            return
        super().done(result)
