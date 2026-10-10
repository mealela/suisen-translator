# -*- coding: utf-8 -*-
"""模型库对话框: 所需模型置顶 / 行内 + 全局进度 / 排队依次下载 / 下载前给出总体积。
绝不自行开始下载——要用户点按钮、一键推荐, 或主窗口在用户确认后调用 install_pinned()。"""
import os

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from core import model_manager as mm

KINDS = {"ollama_vlm": "视觉读字", "ollama_llm": "翻译", "erase": "擦除"}


class ModelManagerDialog(QDialog):
    def __init__(self, cfg, parent=None, pinned_ids=None):
        super().__init__(parent)
        self.cfg = cfg
        self.worker = None
        self._chain = []            # 待下载队列(按加入顺序依次执行)
        self._current = None
        self._apply_after = False
        self._pinned = list(pinned_ids or [])
        self._status_cells = {}
        self.setWindowTitle("模型库 · 下载与启用")
        self.setMinimumSize(960, 640)
        self.resize(1000, 700)
        v = QVBoxLayout(self)
        v.setSpacing(10)
        tip = QLabel("模型安装到软件目录 models/; 定位用的 RapidOCR ONNX 模型首次使用时自动下载; "
                     "中日韩字体在导出时按需自动下载, 无需在此管理。可以连续点多个[下载安装], 任务自动排队; "
                     "下载支持断点续传, 中断后再点一次即从上次进度继续; 全部镜像失败时可编辑 models/registry.json 覆盖地址。")
        tip.setProperty("hint", True)
        tip.setWordWrap(True)
        v.addWidget(tip)

        rec = [x for x in mm.load_registry() if x["id"] in mm.RECOMMENDED_IDS]
        names = " + ".join(x["name"].split("(")[0].split("·")[-1].strip() for x in rec)
        self.btn_rec = QPushButton(f"一键安装推荐组合: {names}(共约 {mm.total_size_gb(rec):.0f} GB)")
        self.btn_rec.setProperty("accent", True)
        self.btn_rec.clicked.connect(self._install_recommended)
        v.addWidget(self.btn_rec)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["模型 / 说明", "用途", "体积", "状态", "操作"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(76)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, w in ((1, 96), (2, 150), (3, 130), (4, 150)):
            self.table.setColumnWidth(col, w)
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        self.table.setShowGrid(False)
        v.addWidget(self.table, 1)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.lbl = QLabel("")
        self.lbl.setProperty("hint", True)
        self.lbl.setWordWrap(True)
        v.addWidget(self.bar)
        v.addWidget(self.lbl)

        hb = QHBoxLayout()
        self.btn_dir = QPushButton("打开 models 文件夹")
        self.btn_cancel = QPushButton("取消下载")
        self.btn_cancel.setEnabled(False)
        btn_close = QPushButton("关闭")
        hb.addWidget(self.btn_dir)
        hb.addStretch(1)
        hb.addWidget(self.btn_cancel)
        hb.addWidget(btn_close)
        v.addLayout(hb)
        self.btn_dir.clicked.connect(lambda: (
            os.makedirs(mm.MODELS_DIR, exist_ok=True),
            QDesktopServices.openUrl(QUrl.fromLocalFile(mm.MODELS_DIR))))
        self.btn_cancel.clicked.connect(lambda: self.worker and self.worker.stop())
        btn_close.clicked.connect(self.accept)
        self._fill()
        if self._pinned:
            pinned = [x for x in mm.load_registry() if x["id"] in self._pinned]
            self.lbl.setText(f"检测到当前配置所需的本地模型尚未下载(已在列表顶部标为[默认], 共约 "
                             f"{mm.total_size_gb(pinned):.1f} GB); 点[一键安装推荐组合]或各行的[下载安装]即可自动排队下载。")

    # ---------- 表格 ----------
    def _ordered_items(self):
        order = list(self._pinned) + [i for i in mm.RECOMMENDED_IDS if i not in self._pinned]
        rank = {mid: n for n, mid in enumerate(order)}
        return sorted(mm.load_registry(), key=lambda x: rank.get(x["id"], 999))

    def _name_cell(self, it):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.setSpacing(2)
        title = it["name"] + ("  · 默认" if it["id"] in self._pinned else "")
        name = QLabel(f"<b>{title}</b>")
        desc = QLabel(it.get("desc", ""))
        desc.setProperty("hint", True)
        desc.setWordWrap(True)
        lay.addWidget(name)
        lay.addWidget(desc)
        lay.addStretch(1)
        return w

    def _fill(self):
        self._items = self._ordered_items()
        self._status_cells.clear()
        self.table.setRowCount(len(self._items))
        for r, it in enumerate(self._items):
            installed = mm.is_installed(it)
            self.table.setCellWidget(r, 0, self._name_cell(it))
            c1 = QTableWidgetItem(KINDS.get(it["kind"], it["kind"]))
            c2 = QTableWidgetItem(it.get("size", "-"))
            c3 = QTableWidgetItem("已安装" if installed else "未安装")
            c3.setForeground(Qt.GlobalColor.green if installed else Qt.GlobalColor.gray)
            for col, cell in ((1, c1), (2, c2), (3, c3)):
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, col, cell)
            self._status_cells[it["id"]] = c3
            btn = QPushButton("启用" if installed else "下载安装")
            if installed:
                btn.setProperty("accent", True)
            btn.clicked.connect(lambda _=False, mid=it["id"]: self._act(mid))
            wrap = QWidget()
            wl = QHBoxLayout(wrap)
            wl.setContentsMargins(8, 14, 8, 14)
            wl.addWidget(btn)
            self.table.setCellWidget(r, 4, wrap)
        self._mark_chain()

    def _set_status(self, mid, text, color=None):
        cell = self._status_cells.get(mid)
        if cell is None:
            return
        cell.setText(text)
        if color is not None:
            cell.setForeground(color)

    def _mark_chain(self):
        for it in self._chain:
            self._set_status(it["id"], "排队中", Qt.GlobalColor.yellow)
        if self._current is not None:
            self._set_status(self._current["id"], "下载中", Qt.GlobalColor.yellow)

    # ---------- 动作 ----------
    def _find(self, mid):
        return next((x for x in self._items if x["id"] == mid), None)

    def _act(self, mid):
        it = self._find(mid)
        if it is None:
            return
        if mm.is_installed(it):
            QMessageBox.information(self, "已启用", mm.apply_to_config(it, self.cfg))
            return
        msg = (f"将下载到软件目录 models/ (约 {float(it.get('size_gb', 0)):.1f} GB)。\n" + it.get("desc", ""))
        if it.get("id") == "propainter":
            msg += "\nProPainter 包含使用限制, 请先阅读官方许可证。"
        if self.worker:
            msg += "\n\n当前已有下载进行中, 该条目将加入队列, 稍后自动开始。"
        if QMessageBox.question(self, "确认下载", msg) != QMessageBox.StandardButton.Yes:
            return
        self._install([it])

    def _install_recommended(self):
        want = [x for x in self._items if x["id"] in mm.RECOMMENDED_IDS]
        self._apply_after = True
        if not self._install(want):
            self._apply_recommended()

    def install_pinned(self):
        """主窗口在用户确认后调用: 下载当前配置所需(置顶)的模型; 配置已指向它们, 无需再[启用]。"""
        items = [it for it in (self._find(mid) for mid in self._pinned) if it]
        if not self._install(items):
            self.lbl.setText("所需模型均已安装, 可以开始处理视频。")

    def _install(self, items):
        """把条目加入下载队列; 空闲则立即开始。返回是否有任务在排队/进行。"""
        queued = False
        for it in items:
            if mm.is_installed(it):
                continue
            if self._current is not None and self._current["id"] == it["id"]:
                queued = True
                continue
            if any(x["id"] == it["id"] for x in self._chain):
                queued = True
                continue
            self._chain.append(it)
            self._set_status(it["id"], "排队中", Qt.GlobalColor.yellow)
            queued = True
        if not queued:
            return False
        if self.worker is None:
            self._next()
        else:
            self.lbl.setText(f"已加入下载队列, 等待中的任务: {len(self._chain)} 个")
        return True

    def _next(self):
        it = self._chain.pop(0)
        self._current = it
        self.worker = mm.InstallWorker(it, self)
        self.worker.progress.connect(self._prog)
        self.worker.done.connect(self._step_done)
        self.worker.failed.connect(self._fail)
        self.bar.setVisible(True)
        self.bar.setValue(0)
        self.btn_cancel.setEnabled(True)
        self._set_status(it["id"], "下载中 0%", Qt.GlobalColor.yellow)
        rest = f" · 队列中还有 {len(self._chain)} 个" if self._chain else ""
        self.lbl.setText(f"开始下载: {it['name']}{rest}")
        self.worker.start()

    def _prog(self, p, msg):
        self.bar.setValue(p)
        if self._current:
            rest = f" · 队列 {len(self._chain)}" if self._chain else ""
            self.lbl.setText(f"{self._current['name']}: {msg}{rest}")
            self._set_status(self._current["id"], f"下载中 {p}%" if p else msg[:14], Qt.GlobalColor.yellow)

    def _step_done(self, mid, msg):
        self.worker = None
        if self._apply_after and self._current:
            mm.apply_to_config(self._current, self.cfg)
        self._current = None
        self._fill()
        if self._chain:
            self._next()
            return
        apply_after = self._apply_after
        self._apply_after = False
        self._finish_ui("全部下载完成。" if apply_after else "下载完成, 可以开始处理视频。")
        if apply_after:
            self._apply_recommended()

    def _fail(self, mid, msg):
        self.worker = None
        self._current = None
        dropped = len(self._chain)
        self._chain = []
        self._apply_after = False
        self._fill()
        self._set_status(mid, "失败", Qt.GlobalColor.red)
        extra = f"(已取消队列中的 {dropped} 个任务)" if dropped else ""
        self._finish_ui("安装失败: " + msg + extra)
        if msg != "已取消":
            QMessageBox.warning(self, "安装失败", msg + "\n\n可在 models/registry.json 自定义镜像后重试。")

    def _finish_ui(self, text):
        self._current = None
        self.bar.setVisible(False)
        self.btn_cancel.setEnabled(False)
        self.lbl.setText(text)

    def _apply_recommended(self):
        for mid in mm.RECOMMENDED_IDS:
            it = mm.find_item(item_id=mid)
            if it and mm.is_installed(it):
                mm.apply_to_config(it, self.cfg)
        self._fill()
        QMessageBox.information(self, "完成", "推荐组合已就绪并设为默认, 可以直接开始处理视频了。")

    def done(self, result):
        running = [t for t in self.findChildren(QThread) if t.isRunning()]
        if running:
            for t in running:
                if hasattr(t, "stop"):
                    t.stop()
                t.requestInterruption()
            QMessageBox.information(self, "任务尚未结束", "已请求停止, 请等待当前分片写完后再关闭。")
            return
        super().done(result)
