"""libmpv 内嵌播放。
MpvPlayer   —— 包装 mpv.MPV(解码/音频/寻址都在 mpv 内部线程), 回调经 Qt 信号排队到 GUI 线程。
MpvViewport —— QOpenGLWidget, 作为 QGraphicsView 的 viewport; mpv render API 直接画进它的 GL 帧缓冲,
               叠加项(QGraphicsItem)照常画在视频之上, 场景坐标 = 视频像素坐标(与导出同源)。"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QOpenGLContext
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from core import mpv_runtime

log = logging.getLogger("suisen.mpv")
MPV_FLIP_Y = True        # QOpenGLWidget 的 FBO 是"上下翻转"存储的, 官方 Qt 示例即用 flip_y=1; 若画面倒置改 False


class _Bridge(QObject):
    """mpv 回调线程 → GUI 线程(跨线程信号自动排队)。"""
    frameReady = Signal()
    timePos = Signal(float)
    pauseChanged = Signal(bool)
    fileLoaded = Signal()
    seekDone = Signal()


class MpvPlayer:
    def __init__(self, hwdec=True):
        m = mpv_runtime.module()
        self.bridge = _Bridge()
        self.mpv = m.MPV(
            vo="libmpv", idle=True, keep_open="yes", pause=True,
            osc=False, osd_level=0, input_default_bindings=False, input_vo_keyboard=False,
            input_cursor=False, cursor_autohide="no", ytdl=False,
            sub_auto="no", sid="no",                      # 绝不自动加载旁边的 *_translated.ass(会和叠加层重影)
            audio_file_auto="no", hr_seek="yes", video_sync="audio", loop_file="no",
            hwdec="auto-safe" if hwdec else "no",
            log_handler=self._log, loglevel="warn")
        for key in ("background-color", "background"):   # 新旧 mpv 选项名不同
            try:
                self.mpv[key] = "#0c0b19"
                break
            except Exception:
                continue
        self.mpv.observe_property("time-pos", self._on_time)
        self.mpv.observe_property("pause", self._on_pause)
        self.mpv.event_callback("file-loaded")(self._on_file_loaded)
        self.mpv.event_callback("playback-restart")(self._on_restart)

    # ---- mpv 线程回调 ----
    def _log(self, level, prefix, text):
        log.debug("mpv[%s] %s: %s", level, prefix, str(text).strip())

    def _on_time(self, _name, value):
        if value is not None:
            self.bridge.timePos.emit(float(value))

    def _on_pause(self, _name, value):
        if value is not None:
            self.bridge.pauseChanged.emit(bool(value))

    def _on_file_loaded(self, _event):
        self.bridge.fileLoaded.emit()

    def _on_restart(self, _event):
        self.bridge.seekDone.emit()

    def _on_render_update(self):
        self.bridge.frameReady.emit()

    # ---- GUI 线程调用 ----
    def open(self, path, audio_path=None):
        self.mpv.pause = True
        try:
            self.mpv["audio-files"] = [audio_path] if audio_path else []
        except Exception:
            try:
                self.mpv.command("set", "audio-files", audio_path or "")
            except Exception as exc:
                log.debug("audio-files 设置失败: %s", exc)
        self.mpv.command("loadfile", str(path), "replace")

    def seek(self, t, exact=True):
        try:
            self.mpv.command("seek", f"{max(0.0, float(t)):.3f}",
                             "absolute+exact" if exact else "absolute+keyframes")
        except Exception as exc:
            log.debug("seek 失败: %s", exc)

    def set_pause(self, paused):
        try:
            self.mpv.pause = bool(paused)
        except Exception:
            pass

    def time_pos(self):
        try:
            v = self.mpv.time_pos
            return float(v) if v is not None else None
        except Exception:
            return None

    def set_mute(self, muted):
        try:
            self.mpv.mute = bool(muted)
        except Exception:
            pass

    def set_hwdec(self, on):
        try:
            self.mpv.hwdec = "auto-safe" if on else "no"
        except Exception:
            pass

    def stop(self):
        try:
            self.mpv.command("stop")
        except Exception:
            pass

    def grab_raw(self):
        """当前显示帧的原始视频像素(BGR, 源分辨率, 不含 OSD): 用于 WYSIWYG 烘焙与取底色。"""
        try:
            res = self.mpv.command("screenshot-raw", "video")   # command 返回解码后的节点 dict(node_command 无返回值)
        except Exception as exc:
            log.debug("screenshot-raw 失败: %s", exc)
            return None
        if not isinstance(res, dict):
            return None
        try:
            w, h, stride = int(res["w"]), int(res["h"]), int(res["stride"])
            arr = np.frombuffer(res["data"], np.uint8)
            if arr.size < stride * h:
                return None
            px = arr[:stride * h].reshape(h, stride)[:, :w * 4].reshape(h, w, 4)
            fmt = str(res.get("format", "bgr0"))
            if fmt in ("rgba", "rgb0"):
                return cv2.cvtColor(px, cv2.COLOR_RGBA2BGR)
            return np.ascontiguousarray(px[:, :, :3])               # bgr0 / bgra
        except Exception as exc:
            log.debug("screenshot-raw 解析失败: %s", exc)
            return None

    def terminate(self):
        try:
            self.mpv.terminate()
        except Exception:
            pass


def _get_proc_address(_ctx, name):
    glctx = QOpenGLContext.currentContext()
    if glctx is None:
        return 0
    try:
        addr = glctx.getProcAddress(bytes(name))
        return int(addr) if addr else 0
    except Exception:
        return 0


class MpvViewport(QOpenGLWidget):
    """QGraphicsView 的 viewport。render context 在 initializeGL 创建、随 GL 上下文销毁而释放
    (必须先于 mpv.terminate, 否则 libmpv 会 abort)。"""

    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        self.ctx = None
        self._proc = None

    def initializeGL(self):
        self._free_ctx()
        m = mpv_runtime.module()
        self._proc = m.MpvGlGetProcAddressFn(_get_proc_address)      # 必须持有引用, 否则被 GC
        try:
            self.ctx = m.MpvRenderContext(self.player.mpv, "opengl",
                                          opengl_init_params={"get_proc_address": self._proc})
        except Exception as exc:
            log.error("mpv render context 创建失败(需要 OpenGL 2.1+/GLES2): %s", exc)
            self.ctx = None
            return
        self.ctx.update_cb = self.player._on_render_update
        ctx = self.context()
        if ctx is not None:
            ctx.aboutToBeDestroyed.connect(self._free_ctx)

    def _free_ctx(self):
        if self.ctx is not None:
            try:
                self.ctx.free()
            except Exception:
                pass
            self.ctx = None

    def ack(self):
        """GUI 线程: 消费 mpv 的"有新帧"标记。"""
        if self.ctx is None:
            return False
        try:
            return bool(self.ctx.update())
        except Exception:
            return True

    def render_frame(self):
        """在 QGraphicsView.drawBackground 的 beginNativePainting 之后调用: 当前帧铺满整个视口(设备像素)。"""
        if self.ctx is None:
            return False
        dpr = self.devicePixelRatioF()
        w = max(1, int(round(self.width() * dpr)))
        h = max(1, int(round(self.height() * dpr)))
        try:
            self.ctx.render(flip_y=MPV_FLIP_Y,
                            opengl_fbo={"w": w, "h": h, "fbo": int(self.defaultFramebufferObject())})
        except Exception as exc:
            log.debug("mpv render 失败: %s", exc)
            return False
        try:                                        # mpv 会改 GL 状态; Qt 的 GL 画笔靠 endNativePainting 重同步, 这里再兜底
            f = QOpenGLContext.currentContext().functions()
            f.glDisable(0x0C11)                     # GL_SCISSOR_TEST
            f.glDisable(0x0B71)                     # GL_DEPTH_TEST
            f.glDisable(0x0B90)                     # GL_STENCIL_TEST
            f.glUseProgram(0)
            f.glActiveTexture(0x84C0)               # GL_TEXTURE0
        except Exception:
            pass
        return True

    def paintGL(self):                              # 独立显示时(非 viewport)也能画
        self.render_frame()

    def video_rect_device(self):
        """mpv 实际把画面放在了视口哪个位置(设备像素 x, y, w, h); 用于核对叠加层对齐。"""
        try:
            d = self.player.mpv.osd_dimensions
            return (int(d["ml"]), int(d["mt"]),
                    int(d["w"]) - int(d["ml"]) - int(d["mr"]), int(d["h"]) - int(d["mt"]) - int(d["mb"]))
        except Exception:
            return None
