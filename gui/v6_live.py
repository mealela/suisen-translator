"""任务实时画面: 工作线程通过 core.v6_common.publish 投递帧, 这里按 100ms 轮询取走交给预览的图像层。
跟随时 mpv 暂停、画面层显示任务帧; 取消跟随时任务帧让位, 播放器对回当前时刻。"""
import os

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox

from core.v6_common import take_live


def sync_flags(win):
    win.preview.placement = win.cfg.placement
    win.preview.pipe_erase = bool(win.cfg.pipe_erase)
    win.preview.bilingual = win.cfg.out_bilingual
    win.preview.mask_style = win.cfg.mask_style
    win.preview.subtitle_format = win.cfg.subtitle_format
    win.preview.align = win.cfg.subtitle_align


def install(win):
    sync_flags(win)
    win._follow_task = QCheckBox("跟随任务画面")
    win._follow_task.setChecked(True)
    win._follow_task.setToolTip("勾选: 任务进行中预览跟着任务走(定位/识别/翻译/擦除/导出同一套逻辑);\n"
                                "取消: 进度条和播放键归你, 可以边处理边看别的位置")
    win._follow_task.toggled.connect(lambda on: on or win.preview.leave_task_view())
    win.statusBar().addPermanentWidget(win._follow_task)
    win._frame_timer = QTimer(win)
    win._frame_timer.timeout.connect(lambda: poll(win))
    win._frame_timer.start(100)


def poll(win):
    packet = take_live()
    if packet is None:
        return
    if isinstance(getattr(win, "_task_preview", None), dict) \
            and os.path.abspath(win.video_path or ".") == packet["path"]:
        win._task_preview["t"] = float(packet["t"])          # 非跟随时预览按进度展示已擦除区
    if not win._follow_task.isChecked():
        return
    path = packet["path"]
    if os.path.abspath(win.video_path or ".") != path:
        if not os.path.isfile(path):
            return
        win._switch_view(path)
        if os.path.abspath(win.video_path or ".") != path:
            return
    sync_flags(win)
    preview = win.preview
    preview.show_task_frame(packet["frame"], packet["t"], packet["baked"])
    if preview.show_geom and packet["boxes"]:
        preview.set_diag(packet["boxes"])
    win.btn_play.setText("播放")
