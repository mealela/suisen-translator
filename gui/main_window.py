import base64
import logging
import math
import os
import subprocess
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QDesktopServices, QFontDatabase, QKeySequence,
                           QPainter, QPainterPath, QPalette, QPixmap, QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QApplication,
                               QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSlider, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolButton,
                               QToolTip, QVBoxLayout, QWidget)

from config import APP_VERSION, AppConfig
from core import ai, hardware, inpaint, workspace
from core import model_manager as mm
from core.ass_generator import parse_ass
from core.enhance import estimate_bg_color
from core.layout import pick_font
from core.models import (MaskEvent, SubtitleEvent, clone_state, event_covered,
                         prepare_cover_masks, rect_quad, restore_state)
from core.paths import base_dir, resource_dir
from core.pipeline import segs_brief, segs_from_brief
from core.v6_common import (WORK, fmt_mmss, fmt_time_cs, grab_frame, load_image,
                            parse_time, qbox, source_quad)
from core.v6_paint import apply_cover_masks, ass_font_ratio, paint_frame, prewarm_glyphs
from core.v6_render import erase_signature, erase_targets_exist
from core.workers import (EraseWorker, HardwareProbeWorker, LocateWorker,
                          QueueWorker, ReadWorker, RecognizeWorker, RenderWorker,
                          TranslateWorker, output_path, resolve_font,
                          subtitle_text)
from gui.dialogs import BatchStyleDialog
from gui.preview import VideoPreview
from gui.settings_dialog import SettingsDialog, font_covers
from gui.tutorial import TutorialOverlay
from gui.v6_live import install as install_live
from gui.v6_live import sync_flags
from gui.wizard import FirstRunWizard

log = logging.getLogger("suisen.gui")

VIDEO_FILTER = "视频文件 (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.ts);;所有文件 (*)"
VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts")
BASE_DIR = base_dir()
RES_DIR = resource_dir()
LOG_PATH = os.path.join(BASE_DIR, "logs", "suisen.log")
TEXT_INPUTS = (QLineEdit, QPlainTextEdit, QAbstractSpinBox)

HELP_TEXT = """<b>快捷键</b><br>
Space 播放/暂停(任何位置) · ←/→ 快退/快进 2s · Ctrl+O 打开 · Ctrl+E 导出字幕文件 · Ctrl+R 翻译选中<br>
Ctrl+C 复制表格行 · Del 删除选中 · Ctrl+Z/Y 撤销/重做 · F1 本帮助<br><br>
<b>定位框 = 字幕本体</b><br>
勾选[显示定位框]后, 候选段(绿实线)与字幕(蓝虚线)的框都能<b>拖边/角改大小、拖标签移动</b>。
框决定译文位置与字号(字号刚好占满框高, 不越界; 宽度不限, 对齐方式见设置)、擦除范围与自动遮罩;
移动框 = 移动字幕。框改动后擦除缓存自动失效, 再点[04]只重擦有改动的时间段。<br><br>
<b>手动加字幕</b><br>
工具条选[框选加字幕], 在字幕上按住左键拖一个框(按住期间<b>滚动滚轮</b>前后翻画面, 松开处=结束时间)
→ 表格出现一行, 可直接双击改<b>时间、原文、译文</b>, 或选中后点[识别选中]/[翻译选中]。框高>宽自动按竖排。
手动框与自动定位框同理: 识别/擦除/嵌字都以框为准。<br><br>
<b>画面编辑手势(字幕与遮罩通用)</b><br>
内部拖=移动 · 拖边缘/四角=改大小 · 滚轮=缩放 · 字幕右键拖=旋转 · Del=删除<br>
空白处右键=添加菜单 · 右键拖框=加遮挡块(任何处理方式下都按[设置→原字处理]的遮罩样式叠加, 不参与擦除)<br><br>
<b>流水线与队列</b><br>
顶部 01~05: 勾选=加入自动流程, 点名字=单独执行该步。<b>表格里选中了字幕 → 只对选中的字幕重做该步</b>
(01 在其时间段内重新定位 / 02 按其框重新读字 / 03 强制重译 / 04 只重擦其时间段 / 05 只导出覆盖它们的片段);
<b>没选中 → 对整段视频重做</b>。进度条只表示当前环节自身的 0→100%。<br>
01~04 只在工作区内处理, 不生成文件; 只有 05 导出会产出文件(按设置附带字幕文件, 格式 ASS/SRT 在设置中选,
预览区按所选格式所见即所得)。<br>
<b>断点接续</b>: 每个视频做到哪一步都会记录(重启也保留)。取消/失败后再点[开始队列], 已识别的跳过 01/02,
已翻译的跳过 03, 已擦除的直接导出。队列条目右键[重置状态]可从头重来。<br><br>
<b>复核</b>: [状态]列标出读字把握低/翻译异常的条目; [原文]列悬停可看到该字幕框截取的原视频画面;
被自动剔除的候选在画面上以红框标出原因。时间条带: 灰=未译 绿=已译 红=待复核, 点击跳转"""


class TimelineStrip(QWidget):
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.events, self.duration, self.cur = [], 0.0, 0.0
        self.setToolTip("灰=未译 / 绿=已译 / 红=待复核 · 点击跳转")

    def set_data(self, events, duration):
        self.events, self.duration = list(events), max(0.001, duration)
        self.update()

    def set_time(self, t):
        self.cur = t
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self.palette().color(QPalette.ColorRole.Base))
        if self.duration <= 0 or w <= 0:
            p.end()
            return
        for e in self.events:
            x1 = int(e.start / self.duration * w)
            x2 = max(x1 + 2, int(e.end / self.duration * w))
            if e.needs_review():
                c = QColor("#e05c6e")
            elif e.translated.strip():
                c = QColor("#3fbf6f")
            else:
                c = QColor("#6a6d7c")
            p.fillRect(x1, 3, x2 - x1, h - 6, c)
        x = int(self.cur / self.duration * w)
        p.fillRect(x, 0, 2, h, QColor("#4f7cff"))
        p.end()

    def mousePressEvent(self, e):
        if self.duration > 0:
            self.seekRequested.emit(e.position().x() / max(1, self.width()) * self.duration)


class QueueItemWidget(QWidget):
    """队列条目: 名称 / 状态文字 / 进度条 分三行"""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(5)
        self.lbl_name = QLabel(name)
        self.lbl_name.setStyleSheet("font-weight:bold;")
        self.lbl_stage = QLabel("等待处理")
        self.lbl_stage.setProperty("hint", True)
        self.lbl_stage.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.setVisible(False)
        v.addWidget(self.lbl_name)
        v.addWidget(self.lbl_stage)
        v.addWidget(self.bar)

    def apply_state(self, st):
        p = st.get("p", 0)
        msg = st.get("msg", "等待处理")
        done = bool(st.get("done")) or p >= 100
        self.bar.setVisible(0 < p < 100 and not done)
        self.bar.setValue(p)
        self.lbl_stage.setText(msg)
        self.lbl_stage.setStyleSheet(st.get("style", ""))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = AppConfig.load()
        self.setWindowTitle(f"水仙翻译器 Suisen Translator v{APP_VERSION}")
        self.resize(1560, 880)
        self.setAcceptDrops(True)
        try:
            changed = mm.align_endpoints(self.cfg)
        except Exception:
            changed = []
        self.video_path = ""
        self.meta = {}
        self.events = []
        self.masks = []
        self.current_ev = None
        self.segments = []
        self._chain_read = False
        self._locate_keep = None          # 范围定位时保留的范围外候选段
        self.worker = None
        self._hw_worker = None
        self._hw_then = None
        self.queue_paths = []
        self._queue_map = []
        self._queue_current_row = -1
        self._qstate = {}
        self._undo, self._redo = [], []
        self._last_undo_key = None
        self._last_undo_t = 0.0
        self._last_locfill = 0.0
        self._last_tbl = 0.0
        self._last_qpersist = 0.0
        self._loading = False
        self._from_preview = False
        self._q_sync = False
        self._store = {}                  # _store[path] = {"events","masks","meta","segments","erased","report"}
        self._processing_path = ""
        self._pending_seek = None
        self._task_t0 = 0.0
        self._last_rect_sig = None
        self._tip_cache = OrderedDict()   # [原文]列悬停截图缓存
        self._tip_frame_cache = OrderedDict()  # 悬停截图的取帧缓存(8 条, 同一时刻多条字幕共享一次取帧)
        self._tip_lock = threading.Lock()
        self._task_preview = None         # 当前视频正在擦除/导出时: {"t": 已处理进度(秒)}; 供非跟随预览按进度展示
        self._task_video = None           # 单步任务(定位/识别/擦除)正在处理的视频: 用户切走视图后, 结果只落该视频工作区
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(self._do_debounced_seek)
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.timeout.connect(self._on_persist_timer)
        self._build_ui()
        self._build_shortcuts()
        install_live(self)
        self.cfg.subtitle_font = pick_font(self.cfg.subtitle_font, self.cfg.target_lang)
        self.preview.set_default_font(self.cfg.subtitle_font)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._elapsed_timer.start(1000)
        self._refresh_engine_label()
        self._update_buttons()
        if changed:
            self.lbl_status.setText("已自动选用已安装模型: " + "、".join(changed))
        if self.cfg.first_run:
            QTimer.singleShot(300, self._show_wizard)
        else:
            QTimer.singleShot(600, lambda: self._check_font(silent=True))
            QTimer.singleShot(900, self._startup_models)

    # ================= UI =================
    def _logo_box(self):
        box = QFrame()
        box.setObjectName("LogoBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lbl = QLabel()
        path = os.path.join(RES_DIR, "assets", "logo.png")
        if os.path.isfile(path):
            pm = QPixmap(path)
            if not pm.isNull():
                pm = pm.scaledToHeight(99, Qt.TransformationMode.SmoothTransformation)
                out = QPixmap(pm.size())
                out.fill(Qt.GlobalColor.transparent)
                p = QPainter(out)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                clip = QPainterPath()
                clip.addRoundedRect(QRectF(0, 0, pm.width(), pm.height()), 16, 16)
                p.setClipPath(clip)
                p.drawPixmap(0, 0, pm)
                p.end()
                lbl.setPixmap(out)
        if lbl.pixmap() is None or lbl.pixmap().isNull():
            lbl.setText("🌼")
            lbl.setStyleSheet("font-size:56px;")
        lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignCenter)
        box.setFixedHeight(115)
        return box

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._pipe_chks = {}

        from gui.header import build_header
        root.addWidget(build_header(self))

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- 队列 ----
        qw = QWidget()
        qv = QVBoxLayout(qw)
        qv.setContentsMargins(12, 12, 6, 12)
        lb = QLabel("批量队列 · 拖入视频文件即可添加")
        lb.setProperty("hint", True)
        qv.addWidget(lb)
        self.list_queue = QListWidget()
        self.list_queue.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_queue.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        qv.addWidget(self.list_queue, 1)
        qr = QHBoxLayout()
        self.btn_qadd = QPushButton("添加")
        self.btn_qdel = QPushButton("移除")
        qr.addWidget(self.btn_qadd)
        qr.addWidget(self.btn_qdel)
        qv.addLayout(qr)
        qr2 = QHBoxLayout()
        self.btn_qstart = QPushButton("开始队列")
        self.btn_qstart.setProperty("accent", True)
        self.btn_qpause = QPushButton("暂停")
        self.btn_qpause.setCheckable(True)
        self.btn_qpause.setVisible(False)
        qr2.addWidget(self.btn_qstart, 1)
        qr2.addWidget(self.btn_qpause)
        qv.addLayout(qr2)
        self.qbar = QProgressBar()                      # 队列级总进度(按视频时长加权)
        self.qbar.setFixedHeight(8)
        self.qbar.setTextVisible(False)
        self.qbar.setRange(0, 100)
        self.qbar.setVisible(False)
        qv.addWidget(self.qbar)
        self.lbl_eta = QLabel("")
        self.lbl_eta.setProperty("hint", True)
        self.lbl_eta.setWordWrap(True)
        qv.addWidget(self.lbl_eta)

        # ---- 预览 ----
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 10, 4, 10)
        tools = QHBoxLayout()
        tools.setSpacing(6)

        def tool(text, tip):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setFixedHeight(28)
            return b
        self.btn_tool_select = tool("选择 / 编辑", "拖动、缩放、旋转画面上的字幕与遮罩; 拖定位框的边/角改范围, 拖标签移动")
        self.btn_tool_sub = tool("框选加字幕", "在字幕上按住左键拖一个框(按住时滚轮=前后翻画面), 松开即生成一条字幕; 框即定位框")
        self.btn_tool_mask = tool("框选加遮挡块", "拖一个框盖住水印/台标等; 任何处理方式下都按遮罩样式叠加, 不参与擦除")
        self.btn_tool_select.setChecked(True)
        self.btn_mute = QToolButton()
        self.btn_mute.setCheckable(True)
        self.btn_mute.setFixedHeight(28)
        self.btn_mute.setText("🔊")
        self.btn_mute.setToolTip("静音 / 取消静音(预览伴音来自原视频)")
        for b in (self.btn_tool_select, self.btn_tool_sub, self.btn_tool_mask):
            tools.addWidget(b)
        tools.addStretch(1)
        tools.addWidget(self.btn_mute)
        lv.addLayout(tools)
        self.preview = VideoPreview(hwdec=bool(getattr(self.cfg, "hw_decode", True)))
        self.preview.frame_override = self._task_frame_override
        lv.addWidget(self.preview, 1)
        if not self.preview.audio.available:
            self.btn_mute.setEnabled(False)
            self.btn_mute.setText("🔇")
            self.btn_mute.setToolTip("当前环境没有可用的音频输出(QtMultimedia 不可用), 预览无声")
        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_play.setFixedWidth(90)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.lbl_time = QLabel("00:00 / 00:00")
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.lbl_time)
        lv.addLayout(ctrl)
        self.strip = TimelineStrip()
        lv.addWidget(self.strip)
        hint = QLabel("字幕/遮罩: 内部拖=移动 · 边缘拖=改大小 · 滚轮=缩放 · 字幕右键拖=旋转 · "
                      "定位框: 拖边/角=改范围 · 拖标签=移动(=移动字幕) · 详见 F1")
        hint.setProperty("hint", True)
        hint.setWordWrap(True)
        lv.addWidget(hint)

        # ---- 字幕页 ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 10, 10, 10)
        self.btn_add = QPushButton("添加字幕")
        self.btn_add.setToolTip("在当前时刻加一条字幕(默认在画面下方居中); 要圈住画面里的原字幕请用工具条[框选加字幕]")
        self.btn_recog = QPushButton("识别选中")
        self.btn_recog.setToolTip("按选中字幕的定位框重新读字(框高>宽时自动按竖排识别); 等同于选中后点顶部[02 识别]")
        self.btn_trans_sel = QPushButton("翻译选中")
        self.btn_trans_sel.setToolTip("翻译选中行(强制重译); 没选中则翻译全部未译")
        self.btn_style = QPushButton("批量样式")
        self.btn_style.setToolTip("对选中的多条字幕统一修改字体/颜色/描边/排布")
        self.btn_props = QPushButton("属性")
        self.btn_props.setCheckable(True)
        self.btn_del = QPushButton("删除")
        self.btn_del.setProperty("danger", True)
        self.btn_delall = QPushButton("全部删除")
        self.btn_delall.setProperty("danger", True)
        self.btn_imp = QPushButton("导入 ASS")
        self.btn_subs = QPushButton("导出字幕文件")
        self.btn_subs.setToolTip("按[设置→输出与导出]所选格式(ASS/SRT)导出字幕文件; 预览区按该格式所见即所得")
        self.btn_next_review = QPushButton("下一待复核 →")
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        for b in (self.btn_add, self.btn_recog, self.btn_trans_sel, self.btn_style, self.btn_props):
            row1.addWidget(b, 1)
        for b in (self.btn_del, self.btn_delall, self.btn_imp, self.btn_subs, self.btn_next_review):
            row2.addWidget(b, 1)
        rv.addLayout(row1)
        rv.addLayout(row2)

        chkrow = QHBoxLayout()
        self.chk_untrans = QCheckBox("只看未译")
        self.chk_review = QCheckBox("只看待复核")
        self.chk_boxes = QCheckBox("显示定位框")
        self.chk_boxes.setChecked(True)
        self.chk_boxes.setToolTip("在画面上显示候选段(绿)与字幕(蓝虚线)的定位框, 可直接拖改")
        chkrow.addWidget(self.chk_untrans)
        chkrow.addWidget(self.chk_review)
        chkrow.addWidget(self.chk_boxes)
        chkrow.addStretch(1)
        rv.addLayout(chkrow)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["开始", "结束", "原文", "译文", "状态"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.viewport().installEventFilter(self)          # [原文]列悬停截图(见 eventFilter)
        hh = self.table.horizontalHeader()
        self.table.setColumnWidth(0, 66)
        self.table.setColumnWidth(1, 66)
        self.table.setColumnWidth(4, 96)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(30)
        rv.addWidget(self.table, 1)

        self.grp = QGroupBox("字幕属性(选中一条后编辑)")
        gl = QGridLayout(self.grp)
        self.sp_start = QDoubleSpinBox(); self.sp_start.setRange(0, 999999); self.sp_start.setDecimals(2)
        self.sp_end = QDoubleSpinBox(); self.sp_end.setRange(0, 999999); self.sp_end.setDecimals(2)
        self.sp_x = QSpinBox(); self.sp_x.setRange(0, 20000)
        self.sp_y = QSpinBox(); self.sp_y.setRange(0, 20000)
        self.sp_fs = QSpinBox(); self.sp_fs.setRange(8, 300)
        self.sp_fs.setToolTip("有定位框时字号由框高自动决定, 此值仅作字形比例基准; 改动后该条切为手动字号")
        self.sp_rot = QDoubleSpinBox(); self.sp_rot.setRange(-180.0, 180.0); self.sp_rot.setDecimals(1)
        self.btn_color = QPushButton(); self.btn_color.setFixedSize(56, 26)
        self.cmb_font = QComboBox()
        self.cmb_font.addItem("(默认)")
        self.cmb_font.addItems(sorted(QFontDatabase.families()))
        self.btn_ocolor = QPushButton(); self.btn_ocolor.setFixedSize(56, 26)
        self.sp_ow = QDoubleSpinBox(); self.sp_ow.setRange(0.0, 30.0); self.sp_ow.setDecimals(1)
        self.sp_ow.setSpecialValueText("自动")
        self.chk_bold = QCheckBox("粗体")
        self.chk_vert = QCheckBox("竖排")
        self.chk_mask = QCheckBox("底色遮罩")
        self.chk_mask.setToolTip("给这条字幕的原字区域叠遮罩(样式见设置→原字处理); 手动改过后就只听这里")
        self.btn_mcolor = QPushButton(); self.btn_mcolor.setFixedSize(56, 26)
        r = 0
        gl.addWidget(QLabel("开始(s)"), r, 0); gl.addWidget(self.sp_start, r, 1)
        gl.addWidget(QLabel("结束(s)"), r, 2); gl.addWidget(self.sp_end, r, 3); r += 1
        gl.addWidget(QLabel("X"), r, 0); gl.addWidget(self.sp_x, r, 1)
        gl.addWidget(QLabel("Y"), r, 2); gl.addWidget(self.sp_y, r, 3); r += 1
        gl.addWidget(QLabel("字号"), r, 0); gl.addWidget(self.sp_fs, r, 1)
        gl.addWidget(QLabel("旋转°"), r, 2); gl.addWidget(self.sp_rot, r, 3); r += 1
        gl.addWidget(QLabel("颜色"), r, 0); gl.addWidget(self.btn_color, r, 1)
        gl.addWidget(QLabel("字体"), r, 2); gl.addWidget(self.cmb_font, r, 3); r += 1
        gl.addWidget(QLabel("描边色"), r, 0); gl.addWidget(self.btn_ocolor, r, 1)
        gl.addWidget(QLabel("描边宽"), r, 2); gl.addWidget(self.sp_ow, r, 3); r += 1
        hb2 = QHBoxLayout()
        hb2.addWidget(self.chk_bold); hb2.addWidget(self.chk_vert)
        hb2.addWidget(self.chk_mask); hb2.addStretch(1)
        gl.addLayout(hb2, r, 0, 1, 2)
        gl.addWidget(QLabel("遮罩色"), r, 2); gl.addWidget(self.btn_mcolor, r, 3)
        self.grp.setVisible(False)
        rv.addWidget(self.grp)

        splitter.addWidget(qw)
        splitter.addWidget(left)
        splitter.addWidget(right)
        qw.setMinimumWidth(210)
        qw.setMaximumWidth(320)
        right.setMinimumWidth(380)
        right.setMaximumWidth(680)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 900, 460])
        root.addWidget(splitter, 1)

        # ---- 状态栏 ----
        self.lbl_elapsed = QLabel("")
        self.lbl_elapsed.setStyleSheet("color:#7d8ada;font-weight:bold;")
        self.statusBar().addWidget(self.lbl_elapsed)
        self.lbl_status = QLabel("就绪 · 拖入视频或点[打开视频] · F1 帮助")
        self.statusBar().addWidget(self.lbl_status, 1)
        self.pbar = QProgressBar()                      # 只表示当前环节自身的 0→100%(环节名写在条上)
        self.pbar.setFixedWidth(340)
        self.pbar.setTextVisible(True)
        self.pbar.setFormat("%p%")
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setProperty("danger", True)
        self.btn_cancel.setVisible(False)
        self.statusBar().addPermanentWidget(self.pbar)
        self.statusBar().addPermanentWidget(self.btn_cancel)

        # ---- 信号 ----
        self.btn_open.clicked.connect(self.open_video)
        self.btn_locate.clicked.connect(self.run_locate)
        self.btn_read.clicked.connect(self.run_read)
        self.btn_trans.clicked.connect(self.run_translate)
        self.btn_erase.clicked.connect(self.run_erase)
        self.btn_render.clicked.connect(self.export_video)
        self.btn_models.clicked.connect(self.open_models)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_cancel.clicked.connect(self.cancel_worker)
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider.sliderPressed.connect(self._slider_pressed)
        self.slider.sliderMoved.connect(lambda v: self._slider_seek(v, final=False))
        self.slider.sliderReleased.connect(lambda: self._slider_seek(self.slider.value(), final=True))
        self.strip.seekRequested.connect(lambda t: self._preview_interactive() and self.preview.seek(t))
        self.btn_mute.toggled.connect(self._toggle_mute)
        self.btn_tool_select.clicked.connect(lambda: self._set_tool("select"))
        self.btn_tool_sub.clicked.connect(lambda: self._set_tool("subtitle"))
        self.btn_tool_mask.clicked.connect(lambda: self._set_tool("mask"))
        pv = self.preview
        pv.timeChanged.connect(self._on_time)
        pv.playStateChanged.connect(lambda p: self.btn_play.setText("暂停" if p else "播放"))
        pv.subtitleChanged.connect(self._on_sub_changed)
        pv.subtitleClicked.connect(self._on_sub_clicked)
        pv.subtitleDeleteRequested.connect(self.delete_event)
        pv.retranslateRequested.connect(self._retranslate_one)
        pv.maskDeleteRequested.connect(self._delete_mask_item)
        pv.maskChanged.connect(lambda *_: self._after_edit())
        pv.segmentChanged.connect(self._on_seg_changed)
        pv.editBegan.connect(lambda: self.push_undo("edit"))
        pv.createSubtitleRequested.connect(self._create_sub_from_rect)
        pv.createMaskRequested.connect(self._create_mask_from_rect)
        pv.addSubtitleAt.connect(lambda x, y: self.add_subtitle(x, y))
        pv.toolFinished.connect(lambda: self._set_tool("select"))
        pv.toolRequested.connect(self._set_tool)
        self.table.currentCellChanged.connect(self._on_row_changed)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.customContextMenuRequested.connect(self._table_menu)
        self.btn_add.clicked.connect(lambda: self.add_subtitle())
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_delall.clicked.connect(self.delete_all)
        self.btn_recog.clicked.connect(self.recognize_selected)
        self.btn_trans_sel.clicked.connect(self.retranslate_selected)
        self.btn_style.clicked.connect(self.batch_style)
        self.btn_imp.clicked.connect(self.import_ass)
        self.btn_subs.clicked.connect(self.export_subtitles)
        self.btn_next_review.clicked.connect(self.next_review)
        self.btn_props.toggled.connect(self.grp.setVisible)
        self.chk_untrans.toggled.connect(self._apply_row_filter)
        self.chk_review.toggled.connect(self._apply_row_filter)
        self.chk_boxes.toggled.connect(self.preview.set_show_geom)
        for w in (self.sp_start, self.sp_end, self.sp_x, self.sp_y, self.sp_fs, self.sp_rot, self.sp_ow):
            w.valueChanged.connect(self._panel_changed)
        for c in (self.chk_bold, self.chk_vert, self.chk_mask):
            c.toggled.connect(self._panel_flags_changed)
        self.cmb_font.currentIndexChanged.connect(self._panel_font_changed)
        self.btn_color.clicked.connect(lambda: self._pick_color("color"))
        self.btn_ocolor.clicked.connect(lambda: self._pick_color("outline_color"))
        self.btn_mcolor.clicked.connect(lambda: self._pick_color("mask_color"))
        self.btn_qadd.clicked.connect(self.queue_add)
        self.btn_qdel.clicked.connect(self.queue_del)
        self.btn_qstart.clicked.connect(lambda: self.queue_start())
        self.btn_qpause.toggled.connect(self._toggle_pause)
        self.list_queue.itemDoubleClicked.connect(self.queue_double_clicked)
        self.list_queue.currentRowChanged.connect(self._on_queue_row)
        self.list_queue.customContextMenuRequested.connect(self._queue_menu)
        self.list_queue.model().rowsMoved.connect(self._queue_reordered)

    def _build_shortcuts(self):
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_video)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.export_subtitles)
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.export_subtitles)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.retranslate_selected)
        QShortcut(QKeySequence("F1"), self, activated=self._show_help)
        space = QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._space_pressed)
        space.setContext(Qt.ShortcutContext.WindowShortcut)
        QShortcut(QKeySequence.StandardKey.Copy, self.table,
                  context=Qt.ShortcutContext.WidgetShortcut,
                  activated=lambda: self._copy_table_rows(self.table))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.table,
                  context=Qt.ShortcutContext.WidgetShortcut, activated=self.delete_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.preview,
                  context=Qt.ShortcutContext.WidgetShortcut, activated=lambda: self._nudge_seek(-2.0))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self.preview,
                  context=Qt.ShortcutContext.WidgetShortcut, activated=lambda: self._nudge_seek(2.0))

    # ================= 帮助/向导/模型/教程/字体/拖拽 =================
    def _show_help(self):
        box = QMessageBox(self)
        box.setWindowTitle("帮助")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(HELP_TEXT + f"<br><br>日志文件: {LOG_PATH}")
        btn_log = box.addButton("打开日志文件夹", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is btn_log:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(LOG_PATH)))

    def _show_wizard(self):
        dlg = FirstRunWizard(self.cfg, self)
        dlg.exec()
        self.cfg.first_run = False
        self.cfg.save()
        self._refresh_engine_label()
        self._check_font(silent=True)
        auto = True if dlg.download_now else None
        if self.cfg.hw_tier == "auto" and not self.cfg.hw_applied:
            self._start_hw_probe(then=lambda: self._ensure_models(auto))
        else:
            QTimer.singleShot(300, lambda: self._ensure_models(auto))

    def _startup_models(self):
        if self.cfg.hw_tier == "auto" and not self.cfg.hw_applied:
            self._start_hw_probe(then=self._ensure_models)
        else:
            self._ensure_models()

    def _start_hw_probe(self, then=None):
        if self._hw_worker is not None:
            return
        self._hw_then = then
        self.lbl_status.setText("正在检测硬件, 自动套用参数预设…")
        self._hw_worker = HardwareProbeWorker(self)
        self._hw_worker.probed.connect(self._hw_probed)
        self._hw_worker.start()

    def _hw_probed(self, info):
        self._hw_worker = None
        if info:
            try:
                hardware.detect_and_apply(self.cfg, info=info)
                self.lbl_status.setText("已按硬件套用预设: " + self.cfg.hw_summary)
            except Exception as exc:
                self.lbl_status.setText(f"硬件预设套用失败, 使用默认参数: {exc}")
        else:
            self.lbl_status.setText("硬件探测失败, 使用默认参数")
        self._refresh_engine_label()
        sync_flags(self)
        self._apply_preview_source()
        then, self._hw_then = self._hw_then, None
        if then:
            QTimer.singleShot(200, then)

    def _ensure_models(self, auto=None):
        if self.worker:
            return
        try:
            missing = mm.missing_required(self.cfg)
        except Exception:
            missing = []
        if not missing:
            self._maybe_tutorial()
            return
        names = "、".join(x["name"] for x in missing)
        size = mm.total_size_gb(missing)
        self.lbl_status.setText("本地模型尚未下载: " + names)
        if auto is None:
            auto = QMessageBox.question(
                self, "下载本地模型",
                f"当前配置需要的本地模型尚未安装:\n{names}\n\n共约 {size:.1f} GB, 请确认磁盘空间充足。\n"
                "是否现在自动排队下载? 下载完成后即可直接处理视频;\n选择[No]可稍后在[模型库]手动安装。"
            ) == QMessageBox.StandardButton.Yes
        from gui.model_dialog import ModelManagerDialog
        dlg = ModelManagerDialog(self.cfg, self, pinned_ids=[x["id"] for x in missing])
        if auto:
            QTimer.singleShot(0, dlg.install_pinned)
        dlg.exec()
        try:
            mm.align_endpoints(self.cfg)
        except Exception:
            pass
        self._refresh_engine_label()
        self._maybe_tutorial()

    def _maybe_tutorial(self):
        if not self.cfg.tutorial_done:
            QTimer.singleShot(300, self.start_tutorial)

    def start_tutorial(self):
        if self.cfg.tutorial_done:
            return
        steps = [
            (None, "欢迎使用", "三步出成品: 把视频拖进左侧队列 → 点[开始队列] → 到输出目录拿成品。"),
            (self.list_queue, "批量队列", "视频拖到这里(可多个); 每个视频做到哪一步都会记住, 取消后再开始会从没做完的阶段继续。"),
            (self.btn_trans.parentWidget(), "流水线 01~05",
             "每格的勾选=加入自动流程, 点名字=单独执行该步(表格里选中了字幕就只对选中的重做); 只有 05 导出会生成文件。"),
            (self.btn_erase.parentWidget(), "04 原字处理",
             "按[设置→原字处理]选的方式处理原字幕(AI 擦除 / 遮罩 / 保留); 单独点它只在工作区内处理, 导出时直接复用。"),
            (self.preview, "定位框与手动补字幕",
             "画面上的框就是字幕本体: 拖边/角改范围、拖标签移动, 译文字号自动贴合框高。"
             "自动没认出的: 工具条选[框选加字幕]拖框, 再选中表格行点[识别选中]/[翻译选中]。"),
            (self.table, "核对原文",
             "鼠标悬停在表格[原文]列上, 会气泡显示该字幕框截取的原视频画面(蓝框=定位框), 便于快速核对读字是否正确。"),
        ]
        self._tut = TutorialOverlay(self, steps, self._tutorial_finished)
        self._tut.show()
        self._tut.raise_()

    def _tutorial_finished(self):
        self.cfg.tutorial_done = True
        self.cfg.save()

    def _check_font(self, silent=False):
        fam = self.cfg.subtitle_font
        if not font_covers(fam, self.cfg.target_lang):
            msg = f"⚠ 字幕字体 [{fam}] 不支持 {self.cfg.target_lang}, 成品可能出现方块。请到[设置→输出与导出]更换。"
            self.lbl_status.setText(msg)
            if not silent:
                QMessageBox.warning(self, "字体检查", msg)
            return False
        return True

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        vids = [p for p in paths if p.lower().endswith(VIDEO_EXTS)]
        for p in vids:
            self._queue_add_path(p)
        if vids and not self.video_path and not self.worker:
            self.load_video_path(vids[0])
        if vids:
            self.lbl_status.setText(f"已添加 {len(vids)} 个视频到队列")

    def _rebuild_recent(self):
        self._recent_menu.clear()
        for p in self.cfg.recent:
            if os.path.isfile(p):
                act = self._recent_menu.addAction(os.path.basename(p))
                act.setToolTip(p)
                act.triggered.connect(lambda _=False, path=p: (self.worker is None and self.load_video_path(path)))
        if not self._recent_menu.actions():
            self._recent_menu.addAction("(暂无记录)").setEnabled(False)

    # ================= 撤销(字幕 + 遮挡块 + 候选段的定位框) =================
    def _snapshot(self):
        snap = clone_state(self.events, self.masks)
        snap["s"] = segs_brief(self.segments)
        return snap

    def push_undo(self, key=None):
        self._schedule_persist()
        now = time.monotonic()
        if key and key == self._last_undo_key and now - self._last_undo_t < 1.2:
            self._last_undo_t = now
            return
        snap = self._snapshot()
        if self._undo and snap == self._undo[-1]:
            self._last_undo_key, self._last_undo_t = key, now
            return
        self._undo.append(snap)
        if len(self._undo) > 60:
            self._undo.pop(0)
        self._redo.clear()
        self._last_undo_key, self._last_undo_t = key, now

    def _apply_state(self, snap):
        self.events, self.masks = restore_state(snap)
        if "s" in snap:
            self.segments = segs_from_brief(snap["s"])
        self.current_ev = None
        self._fill_table()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self._set_segment_views()
        self._commit_edit()

    def undo(self):
        if not self._undo or not self._editable():
            return
        self._redo.append(self._snapshot())
        self._apply_state(self._undo.pop())
        self._last_undo_key = None
        self.lbl_status.setText("已撤销")

    def redo(self):
        if not self._redo or not self._editable():
            return
        self._undo.append(self._snapshot())
        self._apply_state(self._redo.pop())
        self._last_undo_key = None
        self.lbl_status.setText("已重做")

    # ================= 工作区状态(断点接续) =================
    def _ensure_store(self, path):
        st = self._store.get(path)
        if st is None:
            loaded = workspace.load_state(path)
            st = ({"events": loaded["events"], "masks": loaded["masks"], "meta": loaded["meta"],
                   "segments": self._coerce_segs(loaded["segments"]), "erased": loaded["erased"],
                   "locate_done": bool(loaded.get("locate_done", False))}
                  if loaded else {})
            self._store[path] = st
        return st

    def _persist(self, path=None):
        path = path or self.video_path
        if not path:
            return
        st = self._store.get(path) or {}
        cur = path == self.video_path
        evs = self.events if cur else (st.get("events") or [])
        mks = self.masks if cur else (st.get("masks") or [])
        meta = self.meta if cur else (st.get("meta") or {})
        segs = self.segments if cur else (st.get("segments") or [])
        try:
            if evs or mks or segs or st.get("erased"):
                workspace.save_state(path, evs, mks, meta, st.get("erased"), segs,
                                     locate_done=bool(st.get("locate_done", False)))
            else:
                workspace.drop_state(path)
        except Exception as exc:
            log.warning("保存工作区状态失败 %s: %s", path, exc)

    def _on_persist_timer(self):
        self._persist()
        self._reconcile_preview_source()
        self._refresh_queue_summary(self.video_path)

    def _schedule_persist(self):
        self._persist_timer.start(1500)

    def _sync_store(self):
        if not self.video_path:
            return
        st = self._store.setdefault(self.video_path, {})
        st["events"] = self.events
        st["masks"] = self.masks
        st["segments"] = self.segments
        st["meta"] = self.meta
        self._persist(self.video_path)
        self._refresh_queue_summary(self.video_path)

    def _commit_edit(self):
        self._sync_store()
        self._reconcile_preview_source()
        self._update_buttons()

    def _state_of(self, path):
        st = self._ensure_store(path)
        if path == self.video_path:
            return {"events": self.events, "masks": self.masks, "meta": self.meta,
                    "segments": self.segments, "erased": st.get("erased"),
                    "locate_done": bool(st.get("locate_done", False))}
        return {"events": st.get("events") or [], "masks": st.get("masks") or [],
                "meta": st.get("meta") or {}, "segments": st.get("segments") or [],
                "erased": st.get("erased"), "locate_done": bool(st.get("locate_done", False))}

    def _has_events(self, path):
        if path == self.video_path:
            return bool(self.events)
        return bool((self._ensure_store(path) or {}).get("events"))

    def _erased_cache(self, path=None):
        path = path or self.video_path
        if not path:
            return None
        st = self._store.get(path) or {}
        info = st.get("erased")
        if not info:
            return None
        evs = self.events if path == self.video_path else (st.get("events") or [])
        try:
            if info.get("sig") != erase_signature(evs, self.cfg, path):
                return None
        except Exception:
            return None
        p = str(info.get("path") or "")
        return p if os.path.isfile(p) else None

    def _stage_summary(self, path):
        st = self._store.get(path) or {}
        evs = self.events if path == self.video_path else (st.get("events") or [])
        if not evs:
            segs = self.segments if path == self.video_path else (st.get("segments") or [])
            return f"已定位 {len(segs)} 段, 待识别" if segs else "等待处理"
        texted = [e for e in evs if e.text.strip()]
        done_tr = [e for e in texted if e.translated.strip()]
        parts = [f"已识别 {len(evs)} 条"]
        if texted:
            parts.append(f"已译 {len(done_tr)}/{len(texted)}")
        if self.cfg.placement == "erase" and self.cfg.pipe_erase:
            parts.append("已擦除" if self._erased_cache(path) else "未擦除")
        review = sum(1 for e in evs if e.needs_review())
        if review:
            parts.append(f"待复核 {review}")
        return " · ".join(parts)

    def _refresh_queue_summary(self, path=None):
        if self.worker is not None:
            return
        for r in range(self.list_queue.count()):
            p = self._row_path(r)
            if p and (path is None or p == path):
                self._set_qstate(r, p=0, msg=self._stage_summary(p), style="", done=False)

    def release_erased_preview(self):
        self._apply_preview_source(prefer_original=True)
        for st in self._store.values():
            if isinstance(st, dict):
                st["erased"] = None

    # ================= 状态 =================
    def _set_pipe(self, key, v):
        setattr(self.cfg, key, bool(v))
        self.cfg.save()
        self._refresh_engine_label()
        if key == "pipe_erase":
            self.preview.pipe_erase = bool(v)
            self.preview.refresh(force=True)
            self._apply_preview_source()
            self._refresh_queue_summary()

    def _nudge_seek(self, dt):
        if not self.video_path or not self._preview_interactive():
            return
        self.preview.pause()
        self.btn_play.setText("播放")
        self.preview.seek(self.preview.cur_time + dt)

    def _refresh_engine_label(self):
        c = self.cfg
        if not c.pipe_erase:
            handle = "原字:保留(04未勾选)"
        elif c.placement == "below":
            handle = "原字:保留(译文放下方)"
        elif c.placement == "erase":
            handle = f"擦除:{c.erase_engine}" + ("(导出时直出)" if c.export_direct else "")
        else:
            handle = f"遮罩:{c.mask_style}"
        export = "导出:MP4" + (f"+{c.subtitle_format.upper()}" if c.pipe_subs else "")
        align = {"left": "靠左", "center": "居中", "right": "靠右"}.get(c.subtitle_align, "居中")
        parts = [f"读字:{ai.ep_summary(c.ep('vlm'))}",
                 f"翻译:{ai.ep_summary(c.ep('translate'))}→{c.target_lang}",
                 handle, export, f"嵌字:{align}",
                 "音轨:保留" if c.keep_audio else "音轨:关",
                 f"档位:{hardware.TIER_NAMES.get(c.hw_tier, '自动')}"]
        self.lbl_engine.setText(" · ".join(parts))

    def _update_buttons(self):
        busy = self.worker is not None
        editable = self._editable()
        has_video = bool(self.video_path)
        has_events = len(self.events) > 0
        has_any = has_events or bool(self.masks)
        self.btn_open.setEnabled(not busy)
        self.btn_recent.setEnabled(not busy)
        self.btn_locate.setEnabled(not busy and has_video)
        self.btn_read.setEnabled(not busy and has_video)
        self.btn_trans.setEnabled(not busy and has_events)
        self.btn_erase.setEnabled(not busy and has_events and has_video)
        self.btn_render.setEnabled(not busy and has_video and has_any)
        self.btn_settings.setEnabled(not busy)
        self.btn_models.setEnabled(not busy)
        for check in self._pipe_chks.values():
            check.setEnabled(not busy)
        self.btn_add.setEnabled(editable and has_video)
        self.btn_del.setEnabled(editable and has_events)
        self.btn_delall.setEnabled(editable and has_any)
        self.btn_recog.setEnabled(editable and has_events and has_video)
        self.btn_trans_sel.setEnabled(not busy and has_events)
        self.btn_style.setEnabled(editable and has_events)
        self.btn_props.setEnabled(editable)
        self.btn_imp.setEnabled(editable and has_video)
        self.btn_subs.setEnabled(editable and has_video and self._can_export_subs())
        self.btn_next_review.setEnabled(editable and any(e.needs_review() for e in self.events))
        for b in (self.btn_tool_select, self.btn_tool_sub, self.btn_tool_mask):
            b.setEnabled(editable and has_video)
        self.btn_qadd.setEnabled(not busy)
        self.btn_qdel.setEnabled(True)
        self.btn_qstart.setEnabled(not busy and len(self.queue_paths) > 0)
        queue_running = isinstance(self.worker, QueueWorker)
        self.btn_qpause.setVisible(queue_running)
        self.qbar.setVisible(queue_running)
        self.list_queue.setDragDropMode(
            QAbstractItemView.DragDropMode.NoDragDrop if busy
            else QAbstractItemView.DragDropMode.InternalMove)

    def _manual_mode(self):
        return (isinstance(self.worker, QueueWorker) and bool(self._processing_path)
                and self.video_path != self._processing_path)

    def _editable(self):
        return self.worker is None or self._manual_mode()

    def _preview_interactive(self):
        follow = getattr(self, "_follow_task", None)
        return self._editable() or follow is None or not follow.isChecked()

    def _apply_editable(self):
        ed = self._editable()
        self.preview.creation_enabled = ed
        self.preview.setInteractive(ed)
        trig = ((QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
                if ed else QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setEditTriggers(trig)
        self.grp.setEnabled(ed)

    def _busy(self, on, msg="", stage=None):
        """stage: 进度条上标出的环节名(如 "02 识别"); 进度条只表示该环节自身的 0→100%。
        未显式给出时, 从 "0X 名称: ..." 形式的消息里取。"""
        self.pbar.setVisible(on)
        self.btn_cancel.setVisible(on)
        if on:
            self.pbar.setValue(0)
            self._task_t0 = time.monotonic()
            if stage is None:
                stage = msg.split(":", 1)[0].strip() if msg[:2].isdigit() else ""
            self.pbar.setFormat(f"{stage} %p%" if stage else "%p%")
        else:
            self._task_t0 = 0.0
            self.lbl_elapsed.setText("")
            self.pbar.setFormat("%p%")
        self._apply_editable()
        if msg:
            self.lbl_status.setText(msg)
        self._update_buttons()

    def _tick_elapsed(self):
        if self.worker is not None and self._task_t0:
            self.lbl_elapsed.setText("⏱ " + fmt_mmss(time.monotonic() - self._task_t0))

    def _progress(self, p, msg):
        self.pbar.setValue(max(0, min(100, p)))
        self.lbl_status.setText(msg)

    def cancel_worker(self):
        if self.worker:
            if isinstance(self.worker, QueueWorker):
                self.worker.set_paused(False)
                self.btn_qpause.setChecked(False)
            self.worker.stop()
            self.lbl_status.setText("正在取消...(已完成的阶段会保留)")

    def _toggle_pause(self, on):
        if isinstance(self.worker, QueueWorker):
            self.worker.set_paused(on)
            self.btn_qpause.setText("继续" if on else "暂停")
            self.lbl_status.setText("队列已暂停" if on else "队列继续")

    def _task_failed(self, msg):
        self.worker = None
        self._task_preview = None
        self._task_video = None
        self._chain_read = False
        self._locate_keep = None
        self._sync_store()
        self._apply_preview_source()
        self._busy(False, "失败: " + msg)

    @staticmethod
    def _ep_error(ep, vision):
        provider = ep.get("provider", "api")
        if provider == "local":
            if vision:
                return "请在[模型库]启用本地读字模型"
            if not os.path.isfile(ep.get("local_model", "")):
                return "本地模型文件不存在"
        elif provider == "api":
            if not ep.get("api_base", "").strip():
                return "服务地址为空"
            if not ep.get("api_model", "").strip():
                return "模型名为空"
        elif provider == "ollama":
            if not ep.get("ollama_model", "").strip():
                return "未选择本地模型"
        else:
            return "未知接入类型"
        return ""

    def _preflight(self, paths):
        c = self.cfg
        msgs = []
        if not any((c.pipe_locate, c.pipe_read, c.pipe_translate, c.pipe_erase, c.pipe_render)):
            return ["请至少勾选顶部一个步骤"]
        need = [p for p in paths if not self._has_events(p)]
        if need and not (c.pipe_locate or c.pipe_read):
            if len(need) == len(paths):
                msgs.append("这些视频都还没有识别结果, 请勾选顶部[01 定位]/[02 识别]"
                            "(已识别的视频才能只跑翻译/原字处理/导出)")
            else:
                msgs.append(f"(警告) {len(need)} 个视频尚未识别且未勾选[01/02], 本轮将跳过它们")
        elif need and not c.pipe_read and (c.pipe_translate or c.pipe_erase or c.pipe_render):
            msgs.append(f"(警告) {len(need)} 个视频尚未识别, 只勾了[01 定位]: 这些视频本轮只做定位")
        if need and c.pipe_read and c.ocr_engine in ("vlm", "auto"):
            e = self._ep_error(c.ep("vlm"), True)
            if e:
                msgs.append("读字引擎: " + e)
        if c.pipe_translate:
            e = self._ep_error(c.ep("translate"), False)
            if e:
                msgs.append("翻译引擎: " + e)
        try:
            missing = mm.missing_required(c)
        except Exception:
            missing = []
        if missing:
            msgs.append("本地模型尚未下载: " + "、".join(x["name"] for x in missing)
                        + f"(共约 {mm.total_size_gb(missing):.1f} GB), 请先到[模型库]安装")
        erase_needed = (c.pipe_erase and c.placement == "erase"
                        and any(self._erased_cache(p) is None for p in paths))
        if (c.pipe_render or erase_needed) and not inpaint.ffmpeg_bin(c):
            msgs.append("(警告) 勾选了[原字处理]/[导出]; 开始时会自动下载 FFmpeg(约110MB)")
        if erase_needed and not any(x["id"] == c.erase_engine for x in missing):
            ok, m = inpaint.check_engine(c)
            if not ok:
                msgs.append("智能擦除: " + m)
        if need and c.pipe_read and c.ocr_engine == "local" and c.vertical_support:
            msgs.append("(警告) 只用本地识别时, 竖排字幕靠投影切字拼行识别, 准确率明显低于 AI 视觉模型; "
                        "竖排多的视频建议改用[全部用 AI]或[智能]")
        if c.pipe_erase and c.placement == "below":
            msgs.append("(警告) 处理方式为[保留原字幕, 译文放下方]: 勾选的[04 原字处理]不会改动原字幕")
        return msgs

    def _confirm_cost(self, paths):
        cloud = (self.cfg.ep("vlm").get("provider") == "api"
                 or self.cfg.ep("translate").get("provider") == "api")
        if not cloud:
            return True
        return QMessageBox.question(
            self, "云端调用确认",
            "当前配置会把字幕截图/文字发给云端 AI 服务, 可能产生费用。\n"
            "使用[模型库]的本地模型则完全免费。\n\n继续?") == QMessageBox.StandardButton.Yes

    # ================= 队列 =================
    def _queue_add_path(self, path):
        if path in self.queue_paths:
            return
        self._ensure_store(path)
        self.queue_paths.append(path)
        self._qstate[path] = {"p": 0, "msg": self._stage_summary(path), "style": ""}
        it = QListWidgetItem()
        it.setData(Qt.ItemDataRole.UserRole, path)
        wdg = QueueItemWidget(os.path.basename(path))
        wdg.apply_state(self._qstate[path])
        sh = wdg.sizeHint()
        sh.setHeight(max(sh.height(), 84))
        it.setSizeHint(sh)
        it.setToolTip(path)
        self.list_queue.addItem(it)
        self.list_queue.setItemWidget(it, wdg)
        self._update_buttons()

    def _row_path(self, row):
        it = self.list_queue.item(row)
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _row_widget(self, row):
        it = self.list_queue.item(row)
        return self.list_queue.itemWidget(it) if it else None

    def _row_for(self, idx):
        return self._queue_map[idx] if 0 <= idx < len(self._queue_map) else -1

    def _set_qstate(self, row, **kw):
        if row < 0:
            return
        path = self._row_path(row)
        if path is None:
            return
        st = self._qstate.setdefault(path, {})
        st.update(kw)
        w = self._row_widget(row)
        if w:
            w.apply_state(st)

    def _refresh_pending_summaries(self):
        for r in range(self.list_queue.count()):
            p = self._row_path(r)
            if p and not self._qstate.get(p, {}).get("done"):
                self._set_qstate(r, msg=self._stage_summary(p))

    def _queue_reordered(self, *args):
        self.queue_paths = [self._row_path(i) for i in range(self.list_queue.count())]
        for i in range(self.list_queue.count()):
            it = self.list_queue.item(i)
            if self.list_queue.itemWidget(it) is None:
                path = it.data(Qt.ItemDataRole.UserRole)
                w = QueueItemWidget(os.path.basename(path))
                w.apply_state(self._qstate.get(path, {}))
                sh = w.sizeHint()
                sh.setHeight(max(sh.height(), 84))
                it.setSizeHint(sh)
                self.list_queue.setItemWidget(it, w)

    def queue_add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "添加视频到队列", "", VIDEO_FILTER)
        for p in paths:
            self._queue_add_path(p)

    def queue_del(self):
        rows = sorted({self.list_queue.row(i) for i in self.list_queue.selectedItems()}, reverse=True)
        removed = False
        for r in rows:
            path = self._row_path(r)
            if path and path == self._processing_path:
                self.lbl_status.setText("正在处理的视频不可移除")
                continue
            self._queue_remove_path(path, r)
            removed = True
        if removed:
            self._rebuild_queue_map()
        self._update_buttons()

    def _rebuild_queue_map(self):
        if not isinstance(self.worker, QueueWorker):
            return
        row_of = {self._row_path(r): r for r in range(self.list_queue.count())}
        self._queue_map = [row_of.get(p, -1) for p in self.worker.paths]

    def _queue_remove_path(self, path, row):
        self._q_sync = True
        self.list_queue.takeItem(row)
        self._q_sync = False
        if path in self.queue_paths:
            self.queue_paths.remove(path)
        self._qstate.pop(path, None)
        if path == self.video_path and self.worker is None:
            nxt = min(row, self.list_queue.count() - 1)
            if nxt >= 0:
                self.load_video_path(self._row_path(nxt))
            else:
                self._clear_editor()

    def _set_time_range(self, duration):
        self.slider.setRange(0, int(max(0.0, duration) * 1000))
        self.slider.setValue(0)
        self.lbl_time.setText(f"00:00 / {fmt_mmss(duration)}")

    def _clear_editor(self):
        if self.video_path:
            self._sync_store()
        self.video_path = ""
        self.meta = {}
        self._task_preview = None
        self.events, self.masks, self.segments = [], [], []
        self.current_ev = None
        self._undo.clear()
        self._redo.clear()
        self._tip_cache.clear()
        self._tip_frame_cache.clear()
        self.preview.pause()
        self.btn_play.setText("播放")
        self.preview.release()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self._set_segment_views()
        self._fill_table()
        self._set_time_range(0.0)
        self.lbl_status.setText("就绪 · 拖入视频或点[打开视频] · F1 帮助")
        self._apply_editable()
        self._update_buttons()

    def queue_double_clicked(self, item):
        if self.worker:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and path != self.video_path:
            self.load_video_path(path)

    def _on_queue_row(self, row):
        if self._q_sync or row < 0:
            return
        path = self._row_path(row)
        if not path:
            return
        if isinstance(self.worker, QueueWorker):
            self._switch_view(path)
        elif self.worker is None and path != self.video_path:
            self.load_video_path(path)

    def _switch_view(self, path):
        if path == self.video_path:
            return
        if self.video_path:
            self._store.setdefault(self.video_path, {}).update({
                "events": self.events, "masks": self.masks, "meta": self.meta, "segments": self.segments})
        meta = self.preview.load(path)
        if not meta:
            self.lbl_status.setText(f"无法打开: {os.path.basename(path)}")
            return
        self.video_path = path
        self.meta = meta
        self._task_preview = {"t": -1.0} if path == self._processing_path else None
        st = self._ensure_store(path)
        self.events = list(st.get("events") or [])
        self.masks = list(st.get("masks") or [])
        self.segments = self._coerce_segs(st.get("segments") or [])
        st["events"], st["masks"], st["segments"] = self.events, self.masks, self.segments
        self.current_ev = None
        self._undo.clear()
        self._redo.clear()
        self._last_undo_key = None
        self._apply_preview_source()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self.preview.set_segments(self.segments)
        self._fill_table()
        self._set_time_range(meta["duration"])
        follow = (path == self._processing_path)
        self.lbl_status.setText(("跟随处理: " if follow else "手动编辑: ") + os.path.basename(path))
        self._apply_editable()
        self._update_buttons()

    def _paint_wysiwyg(self, frame, t):
        """所见即所得: 画面层与[05 导出]完全相同(遮罩+嵌字)。
        字幕文件选 SRT 时, 文字层改由预览按 SRT 样式叠加(见 preview.SrtOverlayItem), 这里只画遮罩。"""
        try:
            if self.cfg.subtitle_format == "srt":
                return apply_cover_masks(frame, self.events, self.masks, t, self.cfg)
            return paint_frame(frame, self.events, self.masks, t, self.cfg)
        except Exception:
            return frame

    def _repaint_baked(self):
        self.preview.rebake()

    # ---- 任务进行中、关闭[跟随任务画面]时的预览画面源 ----
    def _task_frame_override(self, t):
        """当前时刻 ≤ 擦除/导出进度 → 返回任务预览帧(已擦除画面, 译文/遮罩由叠加层按所见即所得绘制);
        超过进度 → 返回 None(保持原视频画面)。仅作用于当前正在被处理的视频。"""
        info = self._task_preview
        if info is None or not self.video_path:
            return None
        if float(t) > float(info.get("t", -1.0)):
            return None
        return self._task_preview_frame(float(t))

    def _task_preview_dir(self):
        from core.workspace import video_key
        return WORK / "task-preview" / video_key(self.video_path)

    def _task_preview_frame(self, t):
        """按文件名直接寻址(不扫目录): 预览帧名为 时刻毫秒.jpg, 在 t 对应的帧号附近 ±2 探测。"""
        d = self._task_preview_dir()
        if not d.is_dir():
            return None
        fps = float(self.meta.get("fps", 25.0) or 25.0)
        k0 = int(t * fps + 1e-6)
        for k in (k0, k0 - 1, k0 + 1, k0 - 2, k0 + 2):
            if k < 0:
                continue
            p = d / ("%08d.jpg" % int(round(k / fps * 1000)))
            if p.is_file():
                img = cv2.imread(str(p))
                if img is not None:
                    return img
        return None

    # ---- 单步任务在别的视频上运行时, 结果只落该视频工作区, 不污染当前视图 ----
    def _task_view_ok(self):
        return not getattr(self, "_task_video", None) or self.video_path == self._task_video

    def _stash_task(self, events=None, segments=None, meta=None, locate_done=None):
        """任务结果写入其所属视频的工作区(用户已切到别的视频): 只落盘, 不动 self.events/segments/预览/表格。"""
        if not self._task_video:
            return
        st = self._store.setdefault(self._task_video, {})
        if events is not None:
            manual = [e for e in (st.get("events") or []) if not e.source_seg_id]
            st["events"] = manual + list(events)   # 手动框选的字幕在该视频工作区里也合并保留
        if segments is not None:
            st["segments"] = list(segments)
        if meta is not None:
            st["meta"] = dict(meta)
        if locate_done is not None:
            st["locate_done"] = bool(locate_done)
        self._persist(self._task_video)

    def _apply_preview_source(self, prefer_original=False):
        if not self.video_path:
            return
        target = None
        if not prefer_original and self.cfg.placement == "erase" and self.cfg.pipe_erase:
            target = self._erased_cache()
        source = target or self.video_path
        self.preview.composite_hook = self._paint_wysiwyg if target else None
        if self.preview.source_path == source:
            self.preview.refresh(force=True)
            self.preview.rebake()
            return
        t = self.preview.cur_time
        self.preview.pause()
        self.btn_play.setText("播放")
        ok = bool(self.preview.load(source, audio_path=self.video_path, start=t))
        if not ok and source != self.video_path:
            self.preview.load(self.video_path, start=t)
            self.preview.composite_hook = None
            target = None
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self.preview.set_segments(self.segments)
        self.lbl_status.setText("预览: " + ("已擦除画面(工作区状态, 译文/遮罩实时叠加)" if target else "原视频"))

    def _reconcile_preview_source(self):
        if not self.video_path or (self.worker is not None and not self._manual_mode()):
            return
        target = None
        if self.cfg.placement == "erase" and self.cfg.pipe_erase:
            target = self._erased_cache()
        want = target or self.video_path
        if self.preview.source_path == want:
            return
        was_erased = self.preview.source_path != self.video_path
        self._apply_preview_source()
        if was_erased and not target:
            self.lbl_status.setText("字幕/定位框改动使擦除缓存失效: 预览已切回原视频; "
                                    "再点[04 原字处理]只会重擦有改动的时间段")

    @staticmethod
    def _open_folder(folder):
        if not folder or not os.path.isdir(folder):
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
            if os.name == "nt":
                os.startfile(folder)  # noqa
            else:
                subprocess.Popen(["xdg-open" if os.name == "posix" else "open", folder])

    def _reset_state(self, path, row=-1):
        if path == self.video_path:
            self._apply_preview_source(prefer_original=True)
            self.events, self.masks, self.segments = [], [], []
            self.current_ev = None
            self._undo.clear()
            self._redo.clear()
            self._tip_cache.clear()
            self._tip_frame_cache.clear()
            self.preview.set_events(self.events)
            self.preview.set_masks(self.masks)
            self._set_segment_views()
            self._fill_table()
        self._store[path] = {}
        workspace.drop_state(path)
        if row >= 0:
            self._set_qstate(row, p=0, msg="等待处理", style="", done=False)
        self._update_buttons()
        self.lbl_status.setText(f"已重置: {os.path.basename(path)}")

    def _queue_menu(self, pos):
        it = self.list_queue.itemAt(pos)
        if it is None:
            menu = QMenu(self)
            a_cf = menu.addAction("清空已完成/失败条目")
            act = menu.exec(self.list_queue.mapToGlobal(pos))
            if act is a_cf:
                self.queue_clear_finished()
            return
        row = self.list_queue.row(it)
        path = self._row_path(row)
        busy = self.worker is not None
        menu = QMenu(self)
        a_run = menu.addAction("单独处理这个视频(按顶部勾选, 从未完成的阶段继续)")
        a_load = menu.addAction("载入到编辑器")
        a_dir = menu.addAction("打开所在文件夹")
        a_out = menu.addAction("打开输出文件夹")
        menu.addSeparator()
        a_reset = menu.addAction("重置状态(清除定位/识别/翻译/擦除结果, 下次从头开始)")
        a_rm = menu.addAction("移出队列")
        menu.addSeparator()
        a_cf = menu.addAction("清空已完成/失败条目")
        a_run.setEnabled(not busy)
        a_load.setEnabled(not busy)
        a_reset.setEnabled(path != self._processing_path)
        a_rm.setEnabled(path != self._processing_path)
        act = menu.exec(self.list_queue.mapToGlobal(pos))
        if act is a_run:
            self.queue_start(rows=[row])
        elif act is a_load:
            self.load_video_path(path)
        elif act is a_dir:
            self._open_folder(os.path.dirname(path))
        elif act is a_out:
            self._open_folder(self.cfg.out_dir.strip() or os.path.dirname(path))
        elif act is a_reset:
            self._reset_state(path, row)
        elif act is a_rm:
            self._queue_remove_path(path, row)
            self._rebuild_queue_map()
            self._update_buttons()
        elif act is a_cf:
            self.queue_clear_finished()

    def queue_clear_finished(self):
        removed_current = False
        n = 0
        for r in reversed(range(self.list_queue.count())):
            path = self._row_path(r)
            if not path or path == self._processing_path:
                continue
            if self._qstate.get(path, {}).get("done"):
                removed_current = removed_current or (path == self.video_path)
                self._q_sync = True
                self.list_queue.takeItem(r)
                self._q_sync = False
                if path in self.queue_paths:
                    self.queue_paths.remove(path)
                self._qstate.pop(path, None)
                n += 1
        if n:
            self._rebuild_queue_map()
        if removed_current and self.worker is None:
            if self.list_queue.count():
                self.load_video_path(self._row_path(0))
            else:
                self._clear_editor()
        else:
            self.lbl_status.setText(f"已清空 {n} 个条目")
        self._update_buttons()

    def queue_start(self, rows=None):
        if self.worker:
            return
        if rows is None:
            rows = list(range(self.list_queue.count()))
        rows = [r for r in rows if self._row_path(r)]
        if not rows:
            return
        paths = [self._row_path(r) for r in rows]
        self._sync_store()
        msgs = self._preflight(paths)
        errs = [m for m in msgs if "(警告)" not in m]
        warns = [m for m in msgs if "(警告)" in m]
        if errs:
            QMessageBox.warning(self, "还差一步", "\n".join(errs))
            if any("[模型库]" in m for m in errs):
                self._ensure_models()
            elif any(("引擎" in m or "擦除" in m) for m in errs):
                self.open_settings()
            return
        if warns and QMessageBox.question(self, "提示", "\n".join(warns) + "\n\n仍要继续吗?") \
                != QMessageBox.StandardButton.Yes:
            return
        if not self._confirm_cost(paths):
            return
        self._check_font(silent=True)
        self._apply_preview_source(prefer_original=True)
        c = self.cfg
        for p in paths:
            evs = self.events if p == self.video_path else ((self._store.get(p) or {}).get("events") or [])
            prewarm_glyphs(evs, c.subtitle_font, c.placement, c.out_bilingual, c.subtitle_align)
        self._queue_map = rows
        for r in rows:
            self._set_qstate(r, p=0, msg="等待处理", style="", done=False)
        self.worker = QueueWorker(paths, c, state_provider=self._state_of,
                                  skip_provider=lambda p: p not in self.queue_paths, parent=self)
        self.worker.async_state = True
        self.worker.requestState.connect(self._q_supply_state)
        self.worker.videoStarted.connect(self._q_started)
        self.worker.itemStage.connect(self._q_stage)
        self.worker.itemProgress.connect(self._q_item_progress)
        self.worker.itemPartial.connect(self._q_partial)
        self.worker.itemLocate.connect(self._q_locate_partial)
        self.worker.itemErased.connect(self._q_erased)
        self.worker.itemDone.connect(self._q_done_item)
        self.worker.overall.connect(self._q_overall)
        self.worker.finishedAll.connect(self._q_finished)
        self.btn_qpause.setChecked(False)
        self.btn_qpause.setText("暂停")
        self.qbar.setValue(0)
        self.lbl_eta.setText("")
        self._busy(True, "队列启动...", stage="队列")
        self.worker.start(QThread.Priority.LowPriority)

    def _q_supply_state(self, idx, path):
        if isinstance(self.worker, QueueWorker):
            self.worker.supply_state(path, self._state_of(path))

    def _q_stage(self, idx, stage):
        """换环节: 条目文字标出环节; 当前视频的进度条清零并写上环节名——每环节自己 0→100%。"""
        row = self._row_for(idx)
        self._set_qstate(row, p=0, msg=f"[{stage}] 进行中...", style="", done=False)
        if row == self._queue_current_row:
            self.pbar.setValue(0)
            self.pbar.setFormat(f"{stage} %p%")

    def _q_started(self, idx, path):
        row = self._row_for(idx)
        self._processing_path = path
        self._ensure_store(path)
        if path == self.video_path:
            self._task_preview = {"t": -1.0}        # 当前视频进入处理: 非跟随预览按进度展示
        if row >= 0:
            self._queue_current_row = row
            self._q_sync = True
            self.list_queue.setCurrentRow(row)
            self._q_sync = False
        if self.video_path and self.video_path != path:
            self._store.setdefault(self.video_path, {}).update({
                "events": self.events, "masks": self.masks, "meta": self.meta, "segments": self.segments})
        elif not self.video_path:
            self._switch_view(path)
        self._apply_editable()
        self._update_buttons()

    def _q_item_progress(self, idx, p, msg):
        row = self._row_for(idx)
        if row < 0:
            return
        self._set_qstate(row, p=max(0, p), msg=msg)
        if row == self._queue_current_row:
            self.pbar.setValue(max(0, p))
            self.lbl_status.setText(f"[{idx + 1}/{len(self._queue_map)}] {msg}")

    def _q_overall(self, p, msg):
        self.qbar.setValue(max(0, min(100, p)))
        self.lbl_eta.setText(msg)

    def _worker_path(self, idx):
        if isinstance(self.worker, QueueWorker) and 0 <= idx < len(self.worker.paths):
            return self.worker.paths[idx]
        return None

    def _q_partial(self, idx, t, events):
        p = self._worker_path(idx)
        row = self._row_for(idx)
        if p:
            self._store.setdefault(p, {})["events"] = list(events)
        if row == self._queue_current_row and not self._manual_mode():
            self._live_events(t, events)
        if p:
            now = time.monotonic()
            if now - self._last_qpersist >= 5.0:
                self._last_qpersist = now
                self._persist(p)

    def _q_locate_partial(self, idx, segs):
        p = self._worker_path(idx)
        if p:
            self._store.setdefault(p, {})["segments"] = list(segs)
        row = self._row_for(idx)
        if row >= 0 and row == self._queue_current_row and not self._manual_mode():
            self._locate_partial(segs)

    def _q_erased(self, idx, info):
        p = self._worker_path(idx)
        if p:
            self._store.setdefault(p, {})["erased"] = {
                k: info.get(k) for k in ("path", "sig", "items", "params", "file")}
            self._persist(p)

    def _q_done_item(self, idx, ok, msg, result):
        row = self._row_for(idx)
        path = self._worker_path(idx) or (self._row_path(row) if row >= 0 else None)
        if result and path:
            store = self._store.setdefault(path, {})
            for k, v in result.items():
                if k == "segments" and not v and store.get("segments"):
                    continue
                store[k] = v
            if path == self.video_path:
                self._task_preview = None
                self.events = list(store.get("events") or [])
                self.masks = list(store.get("masks") or [])
                self.segments = self._coerce_segs(store.get("segments") or [])
                store["events"], store["masks"], store["segments"] = self.events, self.masks, self.segments
                self.current_ev = None
                self._tip_cache.clear()
                self._tip_frame_cache.clear()
                self.preview.set_events(self.events)
                self.preview.set_masks(self.masks)
                self._fill_table()
                self._set_segment_views()
            self._persist(path)
        if row >= 0:
            self._set_qstate(row, p=100 if ok else self._qstate.get(self._row_path(row), {}).get("p", 0),
                             msg=msg, style="color:#7dd87d;" if ok else "color:#ff7d9c;", done=True)

    def _q_finished(self, ok, fail):
        self.worker = None
        self._task_preview = None
        self._queue_current_row = -1
        self._processing_path = ""
        self.lbl_eta.setText("")
        self.qbar.setValue(100)
        tail = " · 未完成的视频下次[开始队列]会从中断处继续" if fail else ""
        self._busy(False, f"队列结束: 成功 {ok} · 失败/跳过 {fail}{tail}")
        self._apply_preview_source()

    # ================= 实时刷新 =================
    def _live_events(self, t, events):
        if not self._task_view_ok():
            self._stash_task(events=list(events))
            return
        manual = [e for e in self.events if not e.source_seg_id]   # 手动框选/手动添加的字幕不被自动识别结果覆盖
        self.events = manual + list(events)
        self.preview.set_events(self.events)
        now = time.monotonic()
        if now - self._last_tbl >= 0.7:
            self._last_tbl = now
            self._fill_table()

    def _on_ai_updated(self, incoming):
        by_id = {e.id: e for e in incoming}
        for ev in self.events:
            newer = by_id.get(ev.id)
            if newer is not None:
                ev.__dict__.update(newer.__dict__)
        self._fill_table()
        self.preview.set_events(self.events)
        if self.current_ev:
            self._fill_panel(self.current_ev)
        self._schedule_persist()

    # ================= 视频 =================
    def open_video(self):
        if self.worker:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if path:
            self.load_video_path(path)

    def load_video_path(self, path, keep_data=False, add_queue=True):
        meta = self.preview.load(path)
        if not meta:
            QMessageBox.warning(self, "错误", "无法打开该视频文件")
            return
        if self.video_path and self.video_path != path and not keep_data:
            self._sync_store()
        if add_queue:
            self._queue_add_path(path)
        self.cfg.add_recent(path)
        self._rebuild_recent()
        self.video_path = path
        self.meta = meta
        self._tip_cache.clear()
        self._tip_frame_cache.clear()
        self._task_preview = {"t": -1.0} if path == self._processing_path else None
        if not keep_data:
            st = self._ensure_store(path)
            self.events = list(st.get("events") or [])
            self.masks = list(st.get("masks") or [])
            self.segments = self._coerce_segs(st.get("segments") or [])
            st["events"], st["masks"], st["segments"] = self.events, self.masks, self.segments
            self.current_ev = None
            self._undo.clear()
            self._redo.clear()
            self._set_segment_views()
        self._apply_preview_source()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self._fill_table()
        self._set_time_range(meta["duration"])
        note = f" · 已恢复上次结果({self._stage_summary(path)})" if (self.events or self.segments) else ""
        self.lbl_status.setText(f"已加载: {os.path.basename(path)} · {meta['w']}x{meta['h']} · "
                                f"{fmt_mmss(meta['duration'])}{note}")
        self._update_buttons()
        if self.events:
            self._prefetch_tip_frames()         # 后台预取悬停截图帧, [原文]列悬停即出

    def toggle_play(self):
        if not self.video_path or not self._preview_interactive():
            return
        if self.preview.playing:
            self.preview.pause()
            self.btn_play.setText("播放")
        else:
            self.preview.play()
            self.btn_play.setText("暂停")

    def _space_pressed(self):
        if isinstance(QApplication.focusWidget(), TEXT_INPUTS):
            return
        self.toggle_play()

    def _toggle_mute(self, on):
        self.preview.set_muted(on)
        self.btn_mute.setText("🔇" if on else "🔊")

    def _set_tool(self, tool):
        self.preview.set_tool(tool)
        for name, btn in (("select", self.btn_tool_select), ("subtitle", self.btn_tool_sub),
                          ("mask", self.btn_tool_mask)):
            btn.setChecked(name == tool)
        if tool == "subtitle":
            self.lbl_status.setText("框选加字幕: 在画面上按住左键拖框圈住字幕(这个框就是定位框); "
                                    "按住时滚轮可前后翻画面, 松开处=结束时间")
        elif tool == "mask":
            self.lbl_status.setText("框选加遮挡块: 拖一个框盖住水印/台标; 导出时按[设置→原字处理]的遮罩样式叠加")

    def _slider_pressed(self):
        if self.video_path and self._preview_interactive():
            self.preview.pause()
            self.btn_play.setText("播放")

    def _slider_seek(self, v, final=True):
        if not self.video_path or not self._preview_interactive():
            return
        sec = v / 1000.0
        self.lbl_time.setText(f"{fmt_mmss(sec)} / {fmt_mmss(self.meta.get('duration', 0))}")
        self._seek_timer.stop()
        self._pending_seek = None
        self.preview.seek(sec, exact=final)

    def _on_time(self, t):
        self.slider.blockSignals(True)
        self.slider.setValue(int(t * 1000))
        self.slider.blockSignals(False)
        self.lbl_time.setText(f"{fmt_mmss(t)} / {fmt_mmss(self.meta.get('duration', 0))}")
        self.strip.set_time(t)

    def _debounced_seek(self, sec):
        self._pending_seek = sec
        self._seek_timer.start(150)

    def _do_debounced_seek(self):
        if self._pending_seek is None:
            return
        sec, self._pending_seek = self._pending_seek, None
        self.preview.seek(sec)
        if self.current_ev is not None and self.current_ev in self.events:
            self.preview.select_event(self.current_ev)

    # ================= 选中语义: 顶部按钮对选中字幕重做 / 不选对整段视频 =================
    def _selected_events(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.events[r] for r in rows if 0 <= r < len(self.events)]

    def _selection_span(self, evs, pad=0.3):
        """选中字幕覆盖的时间段(前后各留 pad 秒), 钳在视频时长内。"""
        dur = float(self.meta.get("duration", 0.0) or 0.0)
        t0 = max(0.0, min(e.start for e in evs) - pad)
        t1 = max(e.end for e in evs) + pad
        if dur > 0:
            t1 = min(t1, dur)
        return t0, max(t1, t0 + 0.5)

    # ================= 定位/读字 =================
    def run_locate(self):
        """[01 定位]: 选中了字幕 → 只在这些字幕的时间段内重新找候选框(范围外的候选段与所有字幕保留);
        没选中 → 整段视频重新定位(现有候选段被替换; 已识别的字幕不动)。"""
        if not self.video_path or self.worker:
            return
        sel = self._selected_events()
        t0, t1 = 0.0, None
        if sel:
            t0, t1 = self._selection_span(sel)
            self._locate_keep = [s for s in self.segments if s.end <= t0 or s.start >= t1]
        else:
            if self.segments and not self._chain_read and QMessageBox.question(
                    self, "重新定位",
                    f"将替换现有 {len(self.segments)} 个候选段(已识别的字幕不受影响), 重新扫描整段视频?\n"
                    "只想重定位某几条字幕的话, 先在表格里选中它们再点[01 定位]。") \
                    != QMessageBox.StandardButton.Yes:
                self._chain_read = False
                return
            self._locate_keep = None
        self.preview.pause()
        self.btn_play.setText("播放")
        self.push_undo()
        self._ensure_store(self.video_path)["locate_done"] = False   # 新一轮定位开始: 完成前不得按"已完成"续跑
        self._task_video = self.video_path          # 标记任务所属视频: 用户切走视图后结果只落该工作区
        self.worker = LocateWorker(self.video_path, self.cfg, t0=t0, t1=t1, parent=self)
        self.worker.progress.connect(self._progress)
        self.worker.snapshot.connect(self._locate_partial)
        self.worker.done.connect(self._locate_done)
        self.worker.failed.connect(self._task_failed)
        label = (f"01 定位: 重新扫描 {fmt_mmss(t0)} → {fmt_mmss(t1)}(选中的 {len(sel)} 条字幕所在时间段)..."
                 if sel else "01 定位: 正在扫描字幕位置(工作区内, 不生成文件)...")
        self._busy(True, label, stage="01 定位")
        self.worker.start(QThread.Priority.LowPriority)

    def _locate_partial(self, segs):
        new = self._coerce_segs(segs)
        if not self._task_view_ok():
            merged = (list(self._locate_keep or []) + new)
            self._stash_task(segments=merged)
            return
        self.segments = (list(self._locate_keep) + new) if self._locate_keep is not None else new
        now = time.monotonic()
        if now - self._last_locfill >= 2.0:
            self._last_locfill = now
            self._set_segment_views()

    def _set_segment_views(self):
        self.preview.set_segments(self.segments)

    @staticmethod
    def _coerce_segs(items):
        return segs_from_brief(items)

    def _locate_done(self, payload):
        self.worker = None
        new = list(payload["segs"])
        if not self._task_view_ok():
            merged = (list(self._locate_keep or []) + new) if self._locate_keep is not None else new
            self._locate_keep = None
            self._chain_read = False                      # 视图已切走: 不链读, 结果只落工作区
            self._stash_task(segments=merged, meta=payload["meta"], locate_done=True)
            self._task_video = None
            self._busy(False, f"01 定位完成: 找到 {len(new)} 处疑似字幕(已在后台写入该视频工作区)")
            return
        self._task_video = None
        if self._locate_keep is not None:            # 范围定位: 范围外的候选段原样保留
            self.segments = sorted(self._locate_keep + new, key=lambda s: (s.start, s.bbox[1], s.bbox[0]))
            scope = f"选中时间段内找到 {len(new)} 处"
        else:
            self.segments = new
            scope = f"找到 {len(new)} 处疑似字幕"
        self._locate_keep = None
        self.meta.update(payload["meta"])
        self._ensure_store(self.video_path)["locate_done"] = True   # 定位完整完成(续跑才能直接复用候选段读字)
        self._sync_store()
        self._set_segment_views()
        dropped = sum(1 for s in new if s.verdict)
        self._busy(False, f"01 定位完成: {scope}"
                          + (f", 其中 {dropped} 处被判为非字幕(画面上红框标出原因)" if dropped else "")
                          + "; 绿框可直接拖边/角修正范围, 拖标签移动")
        if self._chain_read:
            self._chain_read = False
            if self.segments:
                self.run_read()
            else:
                self.lbl_status.setText("定位完成, 但没有发现字幕。")

    def run_read(self):
        """[02 识别]: 选中了字幕 → 按这些字幕的定位框重新读字(=识别选中); 没选中 → 整段视频按候选段重新识别
        (会替换现有字幕与译文, 先确认)。没有候选段时先自动定位再识别。"""
        if not self.video_path or self.worker:
            return
        if self._selected_events():
            self.recognize_selected()
            return
        if not self.segments:
            self._chain_read = True
            self.run_locate()
            return
        if self.events and QMessageBox.question(
                self, "重新识别",
                f"将按 {len(self.segments)} 个候选段重新读字, 现有 {len(self.events)} 条字幕及其译文会被替换。继续?\n"
                "只想重识别某几条的话, 先在表格里选中它们再点[02 识别]。") != QMessageBox.StandardButton.Yes:
            return
        self.push_undo()
        self._task_video = self.video_path          # 标记任务所属视频
        self.worker = ReadWorker(self.video_path, self.segments, self.cfg, parent=self)
        self.worker.progress.connect(self._progress)
        self.worker.snapshot.connect(lambda evs: self._live_events(-1.0, evs))
        self.worker.done.connect(self._read_done)
        self.worker.failed.connect(self._task_failed)
        self._busy(True, "02 识别: 正在读出文字(工作区内, 不生成文件)...", stage="02 识别")
        self.worker.start(QThread.Priority.LowPriority)

    def _read_done(self, payload):
        self.worker = None
        if not self._task_view_ok():
            self._stash_task(events=list(payload["events"]), segments=payload["segs"])
            self._task_video = None
            self._busy(False, f"02 识别完成: {len(payload['events'])} 条字幕(已在后台写入该视频工作区)")
            return
        self._task_video = None
        manual = [e for e in self.events if not e.source_seg_id]   # 手动框选/手动添加的字幕与自动识别结果合并保留
        self.events = manual + list(payload["events"])
        self.segments = payload["segs"]
        self.current_ev = None
        self._fill_table()
        self.preview.set_events(self.events)
        self._set_segment_views()
        self._commit_edit()
        review = sum(1 for e in self.events if e.needs_review())
        self._busy(False, f"02 识别完成: {len(self.events)} 条字幕"
                          + (f", 其中 {review} 条待复核(用[下一待复核]快速过一遍)" if review else "")
                          + "; 可以直接翻译")
        self._prefetch_tip_frames()             # 后台预取悬停截图帧, [原文]列悬停即出

    def _on_seg_changed(self, seg):
        """候选段(识别前)的定位框被拖改: 落盘; 识别时按新框读字(快照装不下新框时自动回原视频截帧)。
        撤销快照已在 editBegan 时记录(含候选段)。"""
        self._schedule_persist()
        x1, y1, x2, y2 = seg.bbox
        self.lbl_status.setText(f"已调整候选 {seg.id[:6]} 的定位框 {x2 - x1:.0f}×{y2 - y1:.0f}px · "
                                "识别时按新框读字")

    # ================= 手动识别选中框 =================
    def recognize_selected(self):
        """[识别选中] / 选中后点[02]: 对选中字幕按其定位框截帧读字; 路由/回退与自动流程同一实现。"""
        if self.worker or not self.video_path or not self.events:
            return
        evs = self._selected_events()
        if not evs:
            QMessageBox.information(self, "识别选中",
                                    "请先选中要识别的字幕行。\n(工具条选[框选加字幕]在画面上拖框, 就会生成一行, 选中它再点识别)")
            return
        jobs = []
        for ev in evs:
            q = source_quad(ev, None)
            if q is None:
                ev.ensure_mask()
                quad = rect_quad(ev.mask)
            else:
                quad = q.tolist()
            t = min(max(ev.start + 0.05, ev.start), max(ev.start, ev.end - 0.05))
            jobs.append((ev.id, t, quad, bool(ev.vertical)))
        self.push_undo()
        self.worker = RecognizeWorker(self.video_path, jobs, self.cfg, parent=self)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._recog_done)
        self.worker.failed.connect(self._task_failed)
        self._busy(True, f"02 识别: 按定位框重新读字(选中的 {len(evs)} 条)...", stage="02 识别")
        self.worker.start()

    def _recog_done(self, results):
        self.worker = None
        n = 0
        for ev in self.events:
            hit = results.get(ev.id)
            if hit:
                ev.text, ev.conf = hit[0], float(hit[1])
                ev.review_reason = ""
                n += 1
        self._fill_table()
        self.preview.set_events(self.events)
        self._commit_edit()
        self._busy(False, f"识别完成: {n} 条已读出文字, 点[翻译选中]生成译文(也可双击表格手动修改)")
        self._prefetch_tip_frames()

    # ================= 翻译 =================
    def run_translate(self):
        """[03 翻译]: 选中了字幕 → 强制重译这些; 没选中 → 翻译全部未译(全部已译时询问是否整体重译)。"""
        if self.worker or not self.events:
            return
        sel = self._selected_events()
        if sel:
            self._retranslate(sel)
            return
        e = self._ep_error(self.cfg.ep("translate"), False)
        if e:
            QMessageBox.warning(self, "还差一步", "翻译引擎: " + e)
            self.open_settings()
            return
        if not self._models_ready_for_translate():
            return
        force = False
        if not any(ev.text.strip() and not ev.translated.strip() for ev in self.events):
            if QMessageBox.question(self, "重新翻译",
                                    "全部字幕都已有译文。整段视频重新翻译(覆盖现有译文)?\n"
                                    "只想重译某几条的话, 先在表格里选中它们再点[03 翻译]。") \
                    != QMessageBox.StandardButton.Yes:
                return
            force = True
        self.push_undo()
        self.worker = TranslateWorker(self.events, self.cfg, force=force, parent=self, video=self.video_path)
        self.worker.progress.connect(self._progress)
        self.worker.updated.connect(self._on_ai_updated)
        self.worker.done.connect(self._trans_done)
        self.worker.failed.connect(self._task_failed)
        self._busy(True, "03 翻译: 启动 AI 翻译..." + ("(整段重译)" if force else ""), stage="03 翻译")
        self.worker.start()

    def _models_ready_for_translate(self):
        """托管本地翻译模型未下载 → 不静默下载, 转到模型库(带进度、可取消)。"""
        ep = self.cfg.ep("translate")
        if not mm.is_managed(ep):
            return True
        item = mm.find_item(tag=ep.get("ollama_model", ""))
        if item is None or mm.is_installed(item):
            return True
        QMessageBox.information(self, "本地模型未下载",
                                f"翻译模型 [{item['name']}] 尚未下载(约 {float(item.get('size_gb', 0)):.1f} GB)。"
                                "接下来打开模型库, 安装完成后再试。")
        self._ensure_models()
        return False

    def _trans_done(self, n):
        self.worker = None
        self._fill_table()
        self._commit_edit()
        self.preview.refresh(force=True)
        missing = sum(1 for e in self.events if e.text.strip() and not e.translated.strip())
        tail = (f" 仍有 {missing} 条无译文(已自动补译过一次), 可选中后再点[翻译选中]" if missing else "")
        self._busy(False, f"03 翻译完成, 共更新 {n} 条。{tail}")

    def retranslate_selected(self):
        """[翻译选中]: 选中了就强制重译选中行; 没选中则翻译全部未译。"""
        if self.worker or not self.events:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        evs = [self.events[r] for r in rows if 0 <= r < len(self.events)]
        if not evs:
            if any(e.text.strip() and not e.translated.strip() for e in self.events):
                self.run_translate()
            else:
                QMessageBox.information(self, "翻译选中", "全部字幕都已有译文; 选中行可强制重译。")
            return
        self._retranslate(evs)

    def _retranslate_one(self, ev):
        if not self.worker and ev in self.events:
            self._retranslate([ev])

    def _retranslate(self, evs):
        e = self._ep_error(self.cfg.ep("translate"), False)
        if e:
            QMessageBox.warning(self, "还差一步", "翻译引擎: " + e)
            return
        if not self._models_ready_for_translate():
            return
        self.push_undo()
        self.worker = TranslateWorker(evs, self.cfg, force=True, parent=self, video=self.video_path)
        self.worker.progress.connect(self._progress)
        self.worker.updated.connect(self._on_ai_updated)
        self.worker.done.connect(self._trans_done)
        self.worker.failed.connect(self._task_failed)
        self._busy(True, f"03 翻译: 重译选中的 {len(evs)} 条...", stage="03 翻译")
        self.worker.start()

    # ================= 表格 =================
    @staticmethod
    def _status_text(ev):
        if not ev.has_text():
            return "待识别"
        if ev.needs_review():
            return "待复核"
        if not ev.translated.strip():
            return "未译"
        return "✓ 已译"

    @staticmethod
    def _status_tip(ev):
        parts = []
        if ev.review_reason:
            parts.append(ev.review_reason)
        if ev.has_text():
            parts.append(f"读字把握 {ev.conf:.2f}" + (", 偏低" if ev.conf < 0.55 else ""))
        if ev.source:
            parts.append("来源: " + ev.source)
        return "\n".join(parts)

    def _style_status_item(self, it, ev):
        it.setText(self._status_text(ev))
        it.setToolTip(self._status_tip(ev))
        if ev.needs_review() or not ev.has_text():
            it.setForeground(QColor("#e05c6e"))
        else:
            it.setData(Qt.ItemDataRole.ForegroundRole, None)

    def _fill_table(self):
        """增量更新: 行数不变时只改有变化的单元格, 不再每 0.7s 重建全部项。"""
        self._loading = True
        self.events.sort(key=lambda e: (e.start, e.y))
        n = len(self.events)
        if self.table.rowCount() != n:
            self.table.setRowCount(n)
        for i, ev in enumerate(self.events):
            values = (fmt_time_cs(ev.start), fmt_time_cs(ev.end), ev.text, ev.translated)
            for c, text in enumerate(values):
                it = self.table.item(i, c)
                if it is None:
                    self.table.setItem(i, c, QTableWidgetItem(text))
                elif it.text() != text:
                    it.setText(text)
            st = self.table.item(i, 4)
            if st is None:
                st = QTableWidgetItem("")
                st.setFlags(st.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, 4, st)
            self._style_status_item(st, ev)
        self._loading = False
        self._apply_row_filter()
        self.strip.set_data(self.events, self.meta.get("duration", 0.0))

    def _refresh_status_row(self, row):
        if 0 <= row < len(self.events) and row < self.table.rowCount():
            it = self.table.item(row, 4)
            if it is not None:
                self._loading = True
                self._style_status_item(it, self.events[row])
                self._loading = False

    def _apply_row_filter(self):
        only_un = self.chk_untrans.isChecked()
        only_rv = self.chk_review.isChecked()
        for i, ev in enumerate(self.events):
            if i < self.table.rowCount():
                hide = (only_un and bool(ev.translated.strip())) or (only_rv and not ev.needs_review())
                self.table.setRowHidden(i, hide)

    def next_review(self):
        if not self.events:
            return
        cur = self.table.currentRow()
        n = len(self.events)
        for i in list(range(cur + 1, n)) + list(range(0, cur + 1)):
            if self.events[i].needs_review():
                self.table.selectRow(i)
                self.table.scrollToItem(self.table.item(i, 0))
                return
        self.lbl_status.setText("没有待复核的字幕")

    def _copy_table_rows(self, table):
        rows = sorted({i.row() for i in table.selectedIndexes()})
        if not rows:
            return
        lines = []
        for r in rows:
            cells = []
            for c in range(table.columnCount()):
                it = table.item(r, c)
                cells.append(it.text() if it else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self.lbl_status.setText(f"已复制 {len(rows)} 行")

    def _table_menu(self, pos):
        row = self.table.rowAt(pos.y())
        sel_rows = {i.row() for i in self.table.selectedIndexes()}
        if row >= 0 and row not in sel_rows:
            self.table.selectRow(row)
        editable = self._editable()
        menu = QMenu(self)
        a_jump = menu.addAction("跳转到此条")
        a_recog = menu.addAction("识别选中(读出框里文字)")
        a_retr = menu.addAction("翻译选中  [Ctrl+R]")
        a_style = menu.addAction("批量修改样式…")
        a_ok = menu.addAction("标记为已复核")
        a_copy = menu.addAction("复制  [Ctrl+C]")
        menu.addSeparator()
        a_del = menu.addAction("删除选中  [Del]")
        a_jump.setEnabled(row >= 0 and editable)
        for a in (a_recog, a_retr, a_style, a_ok, a_del):
            a.setEnabled(editable and bool(self.events))
        act = menu.exec(self.table.mapToGlobal(pos))
        if act is a_jump and row >= 0:
            self._on_row_changed(row)
        elif act is a_recog:
            self.recognize_selected()
        elif act is a_retr:
            self.retranslate_selected()
        elif act is a_style:
            self.batch_style()
        elif act is a_ok:
            self._mark_reviewed()
        elif act is a_copy:
            self._copy_table_rows(self.table)
        elif act is a_del:
            self.delete_selected()

    def _mark_reviewed(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        evs = [self.events[r] for r in rows if 0 <= r < len(self.events)]
        if not evs:
            return
        self.push_undo()
        for ev in evs:
            ev.review_reason = ""
            ev.conf = max(ev.conf, 0.9)
        self._fill_table()
        self._commit_edit()
        self.lbl_status.setText(f"已标记 {len(evs)} 条为已复核")

    def batch_style(self):
        if not self._editable() or not self.events:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        evs = [self.events[r] for r in rows if 0 <= r < len(self.events)]
        if not evs:
            QMessageBox.information(self, "批量样式", "请先在表格里选中要修改的字幕行(可多选)。")
            return
        dlg = BatchStyleDialog(len(evs), evs[0], self)
        if not dlg.exec():
            return
        self.push_undo()
        n = dlg.apply(evs)
        if self.current_ev:
            self._fill_panel(self.current_ev)
        self.preview.refresh(force=True)
        self._repaint_baked()
        self._commit_edit()
        self.lbl_status.setText(f"已更新 {n} 条字幕的样式")

    # ================= [原文]列悬停: 该字幕框截取的原视频画面 =================
    def eventFilter(self, obj, event):
        """表格[原文]列悬停: 气泡显示该字幕框在原视频里截取的画面(蓝框=定位框), 便于对照读字是否正确、框是否盖全。
        其他列/没有可截画面时走默认行为([状态]列自己的提示不受影响)。"""
        if obj is self.table.viewport() and event.type() == QEvent.Type.ToolTip:
            index = self.table.indexAt(event.pos())
            if index.isValid() and index.column() == 2 and 0 <= index.row() < len(self.events):
                html = self._tip_html(self.events[index.row()])
                if html:
                    QToolTip.showText(event.globalPos(), html, self.table.viewport(),
                                      self.table.visualRect(index))
                    return True
        return super().eventFilter(obj, event)

    def _tip_html(self, ev):
        """按(视频, 条目, 时刻, 框几何)缓存: 框被拖改后自动重截; 没有任何框几何的自由文字不出图。
        画面来源始终是原视频(不是擦除缓存), 这样才能对照原文。"""
        if not self.video_path:
            return ""
        t = ev.sample_time if ev.start <= ev.sample_time < ev.end else ev.start + 0.05
        dur = float(self.meta.get("duration", 0.0) or 0.0)
        t = max(0.0, min(float(t), dur - 0.03) if dur > 0 else max(0.0, float(t)))
        q = source_quad(ev, t)
        if q is None and ev.mask and len(ev.mask) == 4:
            q = np.float32(rect_quad(ev.mask))
        if q is None:
            return ""
        q = np.asarray(q, np.float32).reshape(4, 2)
        key = (self.video_path, ev.id, round(t, 2), tuple(np.round(q, 1).ravel().tolist()))
        with self._tip_lock:
            hit = self._tip_cache.get(key)
            if hit is not None:
                self._tip_cache.move_to_end(key)
                return hit
        html = self._tip_render(ev, t, q)
        with self._tip_lock:
            self._tip_cache[key] = html
            while len(self._tip_cache) > 64:
                self._tip_cache.popitem(last=False)
        return html

    def _tip_frame(self, t):
        """原视频 t 时刻的帧(最多按 1600 宽降采样), LRU 缓存 8 条——同一时刻的多条字幕共享一次取帧。
        返回 (frame, scale); 取帧失败返回 (None, 1.0)。"""
        key = (self.video_path, int(round(t * 1000)))
        with self._tip_lock:
            hit = self._tip_frame_cache.get(key)
            if hit is not None:
                self._tip_frame_cache.move_to_end(key)
                return hit
        frame = grab_frame(self.video_path, t)
        if frame is not None and frame.shape[1] > 1600:
            s = 1600.0 / frame.shape[1]
            frame = cv2.resize(frame, (1600, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
            scale = s
        else:
            scale = 1.0
        item = (frame, scale)
        with self._tip_lock:
            self._tip_frame_cache[key] = item
            while len(self._tip_frame_cache) > 8:
                self._tip_frame_cache.popitem(last=False)
        return item

    def _prefetch_tip_frames(self):
        """后台预取前 8 条字幕的原视频帧, 让[原文]列悬停截图即出(取帧走缓存, 自动去重)。"""
        token = self.video_path
        events = list(self.events)[:8]
        if not token or not events:
            return

        def work():
            for e in events:
                if token != self.video_path:
                    return
                t = e.sample_time if e.start <= e.sample_time < e.end else e.start + 0.05
                try:
                    self._tip_frame(max(0.0, float(t)))
                except Exception:
                    return
        threading.Thread(target=work, daemon=True).start()

    def _tip_render(self, ev, t, q):
        frame, fscale = self._tip_frame(t)
        if frame is not None:
            qq = np.asarray(q, np.float32) * fscale
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = qbox(qq)
            pad = max(12.0, 0.6 * (y2 - y1), 0.12 * (x2 - x1))          # 与定位快照相同的上下文边距
            xa, ya = max(0, int(math.floor(x1 - pad))), max(0, int(math.floor(y1 - pad)))
            xb, yb = min(w, int(math.ceil(x2 + pad))), min(h, int(math.ceil(y2 + pad)))
            if xb - xa < 4 or yb - ya < 4:
                return ""
            crop = frame[ya:yb, xa:xb]
            offset = np.float32([xa, ya])
            note = f"{fmt_mmss(t)} · 定位框 {x2 - x1:.0f}×{y2 - y1:.0f}px(蓝框) · 画面上拖框边/角可修正"
        else:
            crop = load_image(ev.sample_image) if ev.sample_image else None   # 退回定位阶段的快照
            if crop is None:
                return ""
            offset = None
            note = "定位阶段快照(原视频取帧失败)"
        ch, cw = crop.shape[:2]
        scale = min(480.0 / cw, 360.0 / ch, 2.0)                       # 最大 480×360, 小图最多放大 2 倍
        if abs(scale - 1.0) > 1e-3:
            crop = cv2.resize(crop, (max(2, int(round(cw * scale))), max(2, int(round(ch * scale)))),
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        else:
            crop = np.ascontiguousarray(crop)
        if offset is not None:
            pts = np.rint((qq - offset) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(crop, [pts], True, (255, 182, 56), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        data = base64.b64encode(buf.tobytes()).decode("ascii")
        return (f"<img src='data:image/jpeg;base64,{data}'><br/>"
                f"<span style='color:#9fb4ff'>{note}</span>")

    def _row_of_event(self, ev_id):
        for i, e in enumerate(self.events):
            if e.id == ev_id:
                return i
        return -1

    def _resort_keep(self, ev_id):
        """时间被改后重排表格并保持选中同一条字幕。"""
        self._fill_table()
        row = self._row_of_event(ev_id)
        if row >= 0:
            self._from_preview = True
            self.table.selectRow(row)
            self._from_preview = False

    def _on_item_changed(self, item):
        if self._loading:
            return
        row, col = item.row(), item.column()
        if row < 0 or row >= len(self.events) or col > 3:
            return
        ev = self.events[row]
        if col in (0, 1):
            val = parse_time(item.text())        # 支持 MM:SS(.CC) 或纯秒数
            if val is None:
                self._loading = True
                item.setText(fmt_time_cs(ev.start if col == 0 else ev.end))
                self._loading = False
                return
            self.push_undo("cell")
            if col == 0:
                ev.start = max(0.0, round(val, 2))
                if ev.end <= ev.start:
                    ev.end = round(ev.start + 0.5, 2)
            else:
                ev.end = round(val, 2)
                if ev.end <= ev.start:
                    ev.start = max(0.0, round(ev.end - 0.5, 2))
            self.preview.refresh(force=True)
            self.strip.update()
            self._repaint_baked()
            if self.current_ev is ev:
                self._fill_panel(ev)
            QTimer.singleShot(0, lambda eid=ev.id: self._resort_keep(eid))
            return
        self.push_undo("cell")
        if col == 2:
            ev.text = item.text()
        else:
            ev.translated = item.text()
        if ev.has_text():
            ev.review_reason = ""      # 手动改过原文/译文 → 视为已复核; 有译文即显示 ✓ 已译
            ev.conf = max(ev.conf, 0.9)
        self._refresh_status_row(row)
        self.preview.refresh(force=True)
        self.strip.update()
        self._repaint_baked()
        self._update_buttons()                 # [导出字幕文件]: SRT 格式要求有带文字的字幕

    def _on_row_changed(self, row, *_):
        if self._loading or not self._editable():
            return
        if row < 0 or row >= len(self.events):
            return
        self.current_ev = self.events[row]
        self._fill_panel(self.current_ev)
        if not self._from_preview:
            self.preview.pause()
            self.btn_play.setText("播放")
            self._debounced_seek(self.current_ev.start + 0.02)
            self.preview.select_event(self.current_ev)

    # ================= 属性面板 =================
    def _fill_panel(self, ev):
        widgets = (self.sp_start, self.sp_end, self.sp_x, self.sp_y, self.sp_fs, self.sp_rot,
                   self.sp_ow, self.cmb_font, self.chk_bold, self.chk_vert, self.chk_mask)
        for w in widgets:
            w.blockSignals(True)
        self.sp_start.setValue(ev.start)
        self.sp_end.setValue(ev.end)
        self.sp_x.setValue(int(ev.x))
        self.sp_y.setValue(int(ev.y))
        self.sp_fs.setValue(int(ev.font_size))
        self.sp_fs.setToolTip("当前为自动排布(按原字框缩放), 此值仅作基准; 改动后该条切为手动字号"
                              if ev.fit_mode != "manual" else "手动字号")
        self.sp_rot.setValue(float(ev.rotation))
        self.sp_ow.setValue(float(ev.outline_width))
        idx = self.cmb_font.findText(ev.font) if ev.font else 0
        self.cmb_font.setCurrentIndex(idx if idx >= 0 else 0)
        self.chk_bold.setChecked(ev.bold)
        self.chk_vert.setChecked(ev.vertical)
        self.chk_mask.setChecked(event_covered(ev, self.cfg.placement, self.cfg.pipe_erase))
        for w in widgets:
            w.blockSignals(False)
        for btn, c in ((self.btn_color, ev.color), (self.btn_ocolor, ev.outline_color),
                       (self.btn_mcolor, ev.mask_color)):
            btn.setStyleSheet(f"background:{c}; border:1px solid #555; border-radius:6px;")

    def _panel_changed(self):
        ev = self.current_ev
        if not ev:
            return
        self.push_undo("panel")
        ev.start = round(self.sp_start.value(), 2)
        ev.end = round(self.sp_end.value(), 2)
        if ev.end <= ev.start:
            ev.end = round(ev.start + 0.5, 2)
            self.sp_end.blockSignals(True)
            self.sp_end.setValue(ev.end)
            self.sp_end.blockSignals(False)
        ev.x = float(self.sp_x.value())
        ev.y = float(self.sp_y.value())
        if ev.font_size != self.sp_fs.value():
            ev.fit_mode = "manual"
        ev.font_size = self.sp_fs.value()
        ev.rotation = round(self.sp_rot.value(), 1)
        ev.outline_width = round(self.sp_ow.value(), 1)
        if ev in self.events:
            row = self.events.index(ev)
            self._loading = True
            self.table.item(row, 0).setText(f"{ev.start:.2f}")
            self.table.item(row, 1).setText(f"{ev.end:.2f}")
            self._loading = False
        self.preview.refresh(force=True)
        self.preview.select_event(ev)
        self.strip.update()
        self._repaint_baked()

    def _panel_flags_changed(self):
        ev = self.current_ev
        if not ev:
            return
        self.push_undo("panel")
        ev.bold = self.chk_bold.isChecked()
        ev.vertical = self.chk_vert.isChecked()
        want_mask = self.chk_mask.isChecked()
        if want_mask != event_covered(ev, self.cfg.placement, self.cfg.pipe_erase):
            ev.mask_custom = True        # 用户明确表态: 之后只听这里, 不再跟随处理方式自动开关
            ev.mask_on = want_mask
            if want_mask:
                ev.ensure_mask()
        self.preview.refresh(force=True)
        self.preview.select_event(ev)
        self._repaint_baked()

    def _panel_font_changed(self):
        ev = self.current_ev
        if not ev:
            return
        self.push_undo("panel")
        ev.font = "" if self.cmb_font.currentIndex() == 0 else self.cmb_font.currentText()
        self.preview.refresh(force=True)
        self._repaint_baked()

    def _pick_color(self, attr):
        ev = self.current_ev
        if not ev:
            return
        c = QColorDialog.getColor(QColor(getattr(ev, attr)), self, "选择颜色")
        if c.isValid():
            self.push_undo("panel")
            setattr(ev, attr, c.name().upper())
            self._fill_panel(ev)
            self.preview.refresh(force=True)
            self._repaint_baked()

    def _on_sub_changed(self, ev):
        """画面上对字幕的任何改动(移动/缩放/旋转/定位框拖改): 面板同步、落盘、擦除画面重画。
        定位框改动会使擦除缓存签名失效, 由 _on_persist_timer → _reconcile_preview_source 统一处理。"""
        if ev is self.current_ev:
            self._fill_panel(ev)
        self._schedule_persist()
        self._repaint_baked()

    def _after_edit(self):
        """画面上拖/缩放遮挡块等不经字幕对象的编辑: 落盘 + 重画。"""
        self._schedule_persist()
        self._repaint_baked()

    def _on_sub_clicked(self, ev):
        if ev in self.events:
            self._from_preview = True
            self.table.selectRow(self.events.index(ev))
            self._from_preview = False

    # ================= 增删/创建 =================
    def add_subtitle(self, x=None, y=None):
        if not self.video_path or not self._editable():
            return
        w = self.meta.get("w", 1280)
        h = self.meta.get("h", 720)
        t = self.preview.cur_time
        self.push_undo()
        ev = SubtitleEvent(start=round(t, 2), end=round(t + 2.0, 2), text="新字幕",
                           x=float(x) if x is not None else w / 2,
                           y=float(y) if y is not None else h * 0.85,
                           font_size=max(24, int(h * 0.05)), color="#FFFFFF")
        self.events.append(ev)
        self._fill_table()
        self.preview.set_events(self.events)
        row = self._row_of_event(ev.id)
        if row >= 0:
            self.table.selectRow(row)
        self._commit_edit()

    def _create_sub_from_rect(self, rect, t0, t1):
        """框选加字幕=人工定位: 这个框就是定位框(geom), 字号按框高自适应, 识别/擦除/嵌字都以它为准。"""
        sig = (round(rect.left(), 1), round(rect.top(), 1), round(rect.width(), 1), round(rect.height(), 1),
               round(t0, 2), round(t1, 2))
        now = time.monotonic()
        last = self._last_rect_sig
        if last and last[0] == sig and now - last[1] < 1.0:
            return
        self._last_rect_sig = (sig, now)
        self.push_undo()
        vertical = rect.height() > rect.width()
        fs = int(max(8, min(300, (rect.width() if vertical else rect.height()) * 0.85)))
        box = (rect.left(), rect.top(), rect.right(), rect.bottom())
        ev = SubtitleEvent(start=round(t0, 2), end=round(t1, 2), text="",
                           x=rect.center().x(), y=rect.center().y(), font_size=fs, vertical=vertical,
                           color="#FFFFFF", bbox=box, geom=rect_quad(box), geom_anchored=True,
                           mask=list(box), fit_mode=str(self.cfg.layout_fit or "contain"))
        self.events.append(ev)
        self._fill_table()
        self.preview.set_events(self.events)
        row = self._row_of_event(ev.id)
        if row >= 0:
            self._from_preview = True
            self.table.selectRow(row)
            self._from_preview = False
            item = self.table.item(row, 2)
            if item is not None:
                self.table.editItem(item)
        self._commit_edit()
        self.lbl_status.setText(f"已框选 {fmt_time_cs(t0)}→{fmt_time_cs(t1)} · 可直接输入原文, 或点[识别选中]自动读出; "
                                "拖边/角改框、拖标签移动, 字号随框高自动贴合")

    def _create_mask_from_rect(self, rect, t0, t1):
        self.push_undo()
        color = "#101010"
        frame = self.preview.get_frame()
        if frame is not None:
            color = estimate_bg_color(frame, rect.left(), rect.top(), rect.right(), rect.bottom())
        m = MaskEvent(start=round(t0, 2), end=round(t1, 2), x1=rect.left(), y1=rect.top(),
                      x2=rect.right(), y2=rect.bottom(), color=color)
        self.masks.append(m)
        self.preview.set_masks(self.masks)
        self._commit_edit()
        self.lbl_status.setText(f"已添加遮挡块 {fmt_time_cs(t0)}→{fmt_time_cs(t1)}(任何处理方式下, 导出时都按遮罩样式盖住这里)")

    def _drop_segments_of(self, evs):
        """删字幕时, 把已生成该字幕的候选段一并视为无效并移除:
        否则删完字幕后该候选段又会以绿框(候选 0.xx)出现在画面上(撤销快照里含候选段, Ctrl+Z 可恢复)。"""
        ids = {getattr(e, "source_seg_id", "") for e in evs}
        ids.discard("")
        if not ids:
            return
        before = len(self.segments)
        self.segments = [s for s in self.segments if s.id not in ids]
        if len(self.segments) != before:
            self._set_segment_views()

    def delete_event(self, ev):
        if ev in self.events:
            self.push_undo("del")
            self.events.remove(ev)
            self._drop_segments_of([ev])
            if self.current_ev is ev:
                self.current_ev = None
            self._fill_table()
            self.preview.set_events(self.events)
            self._commit_edit()

    def _delete_mask_item(self, item):
        self.push_undo("del")
        if item.kind == "m":
            if item.obj in self.masks:
                self.masks.remove(item.obj)
            self.preview.set_masks(self.masks)
        else:
            item.obj.mask_custom = True      # 用户明确关掉: 不再跟随处理方式自动打开
            item.obj.mask_on = False
            self.preview.refresh(force=True)
            if self.current_ev is item.obj:
                self._fill_panel(item.obj)
        self._repaint_baked()
        self._commit_edit()

    def delete_selected(self):
        if not self._editable():
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        self.push_undo()
        removed = [self.events[r] for r in rows if 0 <= r < len(self.events)]
        for r in rows:
            if 0 <= r < len(self.events):
                self.events.pop(r)
        self._drop_segments_of(removed)
        self.current_ev = None
        self._fill_table()
        self.preview.set_events(self.events)
        self._commit_edit()
        self.lbl_status.setText(f"已删除 {len(rows)} 条字幕 (Ctrl+Z 可撤销)")

    def delete_all(self):
        if not self._editable() or (not self.events and not self.masks):
            return
        if QMessageBox.question(self, "全部删除", "删除当前视频的全部字幕和遮挡块? (Ctrl+Z 可撤销)") \
                != QMessageBox.StandardButton.Yes:
            return
        self.push_undo()
        removed = list(self.events)
        self.events = []
        self.masks = []
        self._drop_segments_of(removed)
        self.current_ev = None
        self._fill_table()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self._commit_edit()
        self.lbl_status.setText("已全部删除 (Ctrl+Z 可撤销)")

    # ================= 导入/导出字幕文件 =================
    def _default_out(self, ext):
        base = os.path.splitext(os.path.basename(self.video_path))[0] + self.cfg.out_suffix + ext
        out_dir = self.cfg.out_dir.strip() or os.path.dirname(self.video_path)
        return os.path.join(out_dir, base)

    def _can_export_subs(self):
        """字幕文件是否有内容可写: ASS 有字幕或遮挡块即可(遮罩也写进 ASS); SRT 只有文字, 必须有带文字的字幕。"""
        if self.cfg.subtitle_format == "srt":
            return any(e.display_text() for e in self.events)
        return bool(self.events or self.masks)

    def _subtitle_text(self):
        """字幕文件内容: 与 05 导出附带、队列导出同一出口(workers.subtitle_text), 格式按设置。"""
        return subtitle_text(self.events, self.masks, self.meta.get("w", 1280), self.meta.get("h", 720), self.cfg)

    def import_ass(self):
        if not self._editable():
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入 ASS 字幕", "", "ASS 字幕 (*.ass)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                events, masks = parse_ass(f.read(), self.meta.get("w", 0), self.meta.get("h", 0),
                                          fs_ratio=ass_font_ratio(resolve_font(self.cfg)))
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))
            return
        if not events and not masks:
            QMessageBox.information(self, "导入 ASS", "没有读到有效字幕")
            return
        if self.events or self.masks:
            if QMessageBox.question(self, "导入 ASS", "将替换当前全部字幕, 继续吗?") != QMessageBox.StandardButton.Yes:
                return
        self.push_undo()
        self.events, self.masks = events, masks
        self.current_ev = None
        self._fill_table()
        self.preview.set_events(self.events)
        self.preview.set_masks(self.masks)
        self._commit_edit()
        self.lbl_status.setText(f"已导入 {len(events)} 条字幕")

    def export_subtitles(self):
        """[导出字幕文件]: 格式按[设置→输出与导出](默认 ASS)。内容与预览所见一致:
        ASS 与画面同源排布(字号/折行/拉伸/竖排/遮罩); SRT 纯文本(预览区文字层按播放器默认样式显示)。"""
        if not self.video_path or not self._editable() or not self._can_export_subs():
            return
        fmt = self.cfg.subtitle_format
        tag = fmt.upper()
        path, _ = QFileDialog.getSaveFileName(self, f"导出 {tag} 字幕文件", self._default_out("." + fmt),
                                              f"{tag} 字幕 (*.{fmt})")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(self._subtitle_text())
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        self.lbl_status.setText(f"{tag} 已导出: {path}")

    def _write_sidecar(self, video_out):
        """成品旁附带字幕文件(与成品同名, 格式按设置; 同名已存在时按设置改名/覆盖)。没有可写内容返回 None。"""
        text = self._subtitle_text()
        if not text:
            return None
        folder = os.path.dirname(video_out) or "."
        stem = os.path.splitext(os.path.basename(video_out))[0]
        os.makedirs(folder, exist_ok=True)
        sub_path = output_path(folder, stem + "." + self.cfg.subtitle_format, self.cfg)
        with open(sub_path, "w", encoding="utf-8-sig") as f:
            f.write(text)
        return sub_path

    # ================= 04 原字处理(工作区) 与 05 导出 =================
    def _erase_engine_ready(self):
        """擦除引擎可用性: LAMA 未安装 → 转模型库(带进度/可取消, 不在渲染里静默下载); 其他引擎 → 设置。"""
        ok, m = inpaint.check_engine(self.cfg)
        if ok:
            return True
        if self.cfg.erase_engine == "lama":
            QMessageBox.information(self, "智能擦除未安装", m + "\n接下来打开模型库, 安装完成后再试。")
            self._ensure_models()
            return False
        QMessageBox.warning(self, "智能擦除不可用",
                            m + "\n(可到[设置→原字处理]改为遮罩覆盖, 或取消顶部[04]只嵌字)")
        self.open_settings()
        return False

    def run_erase(self):
        """[04 原字处理]: 选中了字幕 → 只重擦这些字幕的时间段(其余画面沿用缓存); 没选中 → 整段:
        缓存一致时询问是否全部重擦, 否则按增量擦除。遮罩覆盖/保留原字幕方式与原来相同。"""
        if not self.video_path or self.worker or not self.events:
            return
        c = self.cfg
        if c.placement == "below":
            QMessageBox.information(self, "原字处理",
                                    "当前处理方式为[保留原字幕, 译文放下方], 不处理原字幕。\n"
                                    "要擦除或遮罩原字幕, 请到[设置→原字处理]切换方式。")
            return
        hint = "" if c.pipe_erase else "(顶部[04]未勾选: 导出时不会处理原字幕, 勾选后生效)"
        if c.placement == "cover":
            self.push_undo()
            n = prepare_cover_masks(self.events, c.placement, c.pipe_erase)
            self.preview.refresh(force=True)
            if self.current_ev:
                self._fill_panel(self.current_ev)
            self._commit_edit()
            self.lbl_status.setText(f"遮罩覆盖: 导出时 {n} 条字幕的原字区域按[{c.mask_style}]样式盖住, 不生成文件{hint}")
            return
        if not erase_targets_exist(self.events, c):
            self.lbl_status.setText("没有带文字的字幕, 无需擦除")
            return
        sel = self._selected_events()
        redo = None
        cached = self._erased_cache()
        if sel:
            pad = float(c.erase_time_pad)
            redo = [(max(0.0, e.start - pad), e.end + pad) for e in sel]
            if not cached:
                redo = None                       # 没有可沿用的缓存: 只能整段擦, 下面照常走增量/全量
        elif cached:
            if QMessageBox.question(self, "重新擦除",
                                    "工作区里已有与当前字幕一致的擦除画面(预览正在使用)。整段重新擦除?\n"
                                    "只想重擦某几条的话, 先在表格里选中它们再点[04]。") \
                    != QMessageBox.StandardButton.Yes:
                return
            redo = [(0.0, float(self.meta.get("duration", 0.0) or 1e9))]
        if not self._erase_engine_ready():
            return
        self._apply_preview_source(prefer_original=True)
        self._sync_store()
        previous = (self._store.get(self.video_path) or {}).get("erased")
        self._task_video = self.video_path          # 标记任务所属视频
        self.worker = EraseWorker(self.video_path, c, self.events, previous=previous, redo=redo, parent=self)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._erase_done)
        self.worker.failed.connect(self._task_failed)
        if sel and redo:
            label = f"04 原字处理: 只重擦选中的 {len(sel)} 条字幕的时间段, 其余画面沿用缓存..."
        else:
            label = ("04 原字处理: 正在擦除原字幕(工作区内, 不生成文件; 遮挡块不参与擦除)..."
                     + ("(增量: 只重擦有改动的时间段)" if previous and previous.get("items") is not None
                        and not redo else ""))
        self._busy(True, label, stage="04 原字处理")
        self._task_preview = {"t": -1.0}            # 擦除进行中: 非跟随预览按进度展示已擦除区
        self.worker.start()

    def _erase_done(self, info):
        self.worker = None
        self._task_preview = None
        self._task_video = None
        path = info.get("video") or self.video_path
        st = self._store.setdefault(path, {})
        st["erased"] = {k: info.get(k) for k in ("path", "sig", "items", "params", "file")}
        self._persist(path)
        self._refresh_queue_summary(path)
        if path == self.video_path:
            self._apply_preview_source()     # 擦除完成: 预览自动切到擦除画面(工作区最新状态)
        self._busy(False, "04 擦除完成(工作区): 预览已切换到擦除画面; 导出时直接复用, "
                          "之后字幕改动只重擦有改动的时间段")

    def _export_plan(self):
        """导出的画面来源: (frames_source=可复用的擦除缓存, erase=是否现场擦除); 引擎不可用返回 None。"""
        c = self.cfg
        frames_source, erase = None, False
        if c.pipe_erase and c.placement == "erase":
            frames_source = self._erased_cache()
            if frames_source is None and erase_targets_exist(self.events, c):
                if not self._erase_engine_ready():
                    return None
                erase = True
        return frames_source, erase

    def _export_label(self, frames_source, erase):
        c = self.cfg
        if frames_source:
            return "(复用已擦除画面 + 嵌字)"
        if erase:
            return "(擦除 + 嵌字, 单次编码)"
        if c.pipe_erase and c.placement == "cover":
            return "(遮罩 + 嵌字)"
        return "(仅嵌字)"

    def export_video(self):
        """[05 导出]: 选中了字幕 → 只导出覆盖这些字幕的片段(文件名带时间段, 不附带字幕文件);
        没选中 → 整段成品(按设置附带字幕文件)。有一致的擦除画面就复用, 否则现场擦除。"""
        if not self.video_path or self.worker:
            return
        if not self.events and not self.masks:
            QMessageBox.information(self, "导出", "还没有字幕或遮挡块; 先识别/翻译, 或手动添加后再导出。")
            return
        self._check_font(silent=False)
        c = self.cfg
        plan = self._export_plan()
        if plan is None:
            return
        frames_source, erase = plan
        sel = self._selected_events()
        t0, t1 = (self._selection_span(sel, pad=0.5) if sel else (0.0, None))
        default = self._default_out(f"_{int(t0)}s-{int(t1)}s.mp4" if sel else ".mp4")
        title = f"导出片段 {fmt_mmss(t0)} → {fmt_mmss(t1)}(选中的 {len(sel)} 条字幕)" if sel else "导出成品视频"
        path, _ = QFileDialog.getSaveFileName(self, title, default, "MP4 视频 (*.mp4)")
        if not path:
            return
        self._sync_store()
        extra = []
        if c.pipe_subs and not sel:
            try:
                sub = self._write_sidecar(path)
            except Exception as exc:
                QMessageBox.warning(self, "导出字幕文件失败", str(exc))
                return
            if sub:
                extra.append(sub)
        prewarm_glyphs(self.events, c.subtitle_font, c.placement, c.out_bilingual, c.subtitle_align)
        self.worker = RenderWorker(self.video_path, c, self.events, self.masks, path, t0=t0, t1=t1,
                                   frames_source=frames_source, erase=erase, parent=self)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(lambda p, extra=extra: self._render_done(p, extra))
        self.worker.failed.connect(self._task_failed)
        self._busy(True, ("05 导出: 片段 " if sel else "05 导出: 开始...") + self._export_label(frames_source, erase),
                   stage="05 导出")
        self._task_preview = {"t": -1.0}            # 导出进行中: 非跟随预览按进度展示已擦除区
        self.worker.start()

    def _render_done(self, path, extra=()):
        self.worker = None
        self._task_preview = None
        names = [os.path.basename(path)] + [os.path.basename(x) for x in extra]
        self._busy(False, "05 导出完成: " + " + ".join(names) + f"  →  {os.path.dirname(path)}")

    # ================= 其他 =================
    def open_models(self):
        from gui.model_dialog import ModelManagerDialog
        ModelManagerDialog(self.cfg, self).exec()
        try:
            mm.align_endpoints(self.cfg)
        except Exception:
            pass
        self._refresh_engine_label()
        sync_flags(self)
        self._apply_preview_source()           # 启用 LAMA 可能把处理方式切为擦除
        self._refresh_pending_summaries()

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self._refresh_engine_label()
            self.preview.set_default_font(self.cfg.subtitle_font)
            self.preview.set_hwdec(bool(self.cfg.hw_decode))
            sync_flags(self)                       # placement / 04 / 遮罩样式 / 双语 / 字幕文件格式 同步给预览
            self.preview.refresh(force=True)
            self._apply_preview_source()           # 处理方式/擦除参数变化 → 预览来源重估
            self._repaint_baked()                  # 字幕格式切换: 擦除画面上烘入的文字层也按新格式重画
            if self.current_ev:
                self._fill_panel(self.current_ev)
            self._check_font(silent=True)
            self._refresh_pending_summaries()
            self._update_buttons()                 # [导出字幕文件]可用性随格式变化(SRT 需有带文字的字幕)

    def closeEvent(self, event):
        threads = [t for t in self.findChildren(QThread) if t.isRunning()]
        if threads:
            for thread in threads:
                if hasattr(thread, "set_paused"):
                    thread.set_paused(False)
                if hasattr(thread, "stop"):
                    thread.stop()
                thread.requestInterruption()
            QMessageBox.information(self, "正在停止",
                                    "已请求取消, 当前任务收尾后请再次关闭(已完成的阶段会保留)。")
            event.ignore()
            return
        self._persist_timer.stop()
        try:
            self._persist()                        # 把最后的编辑落盘
        except Exception:
            pass
        # 移出队列的视频: 工作区数据只保留到本次运行结束; 仍在队列里的与正在编辑的保留, 供下次断点接续
        for p in list(self._store):
            if p not in self.queue_paths and p != self.video_path:
                try:
                    workspace.drop_state(p)
                except Exception:
                    pass
        mm.shutdown()
        self.preview.shutdown()
        event.accept()
