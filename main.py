import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import truststore                          # 让 requests/urllib3 信任 Windows 系统证书库:
    truststore.inject_into_ssl()              # 杀毒软件/代理拦截 HTTPS(自签名证书)时不报 CERTIFICATE_VERIFY_FAILED
except Exception:
    pass

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QIcon, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from core.paths import base_dir, resource_dir

BASE = base_dir()
LOG_DIR = os.path.join(BASE, "logs")
LOG_PATH = os.path.join(LOG_DIR, "suisen.log")


def _setup_logging(debug=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        if os.path.getsize(LOG_PATH) > 5 * 1024 * 1024:
            os.remove(LOG_PATH)
    except OSError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()])


def _excepthook(tp, val, tb):
    logging.error("未捕获异常:\n%s", "".join(traceback.format_exception(tp, val, tb)))
    try:
        QMessageBox.critical(None, "程序错误",
                             f"发生未预期的错误:\n{val}\n\n详细信息已写入日志:\n{LOG_PATH}")
    except Exception:
        pass


def _enforce_quota(root, max_mb, keep=("suisen.log",)):
    root = Path(root)
    if not root.is_dir():
        return
    entries = []
    for p in root.iterdir():
        if p.name in keep:
            continue
        try:
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else p.stat().st_size)
            entries.append((p.stat().st_mtime, size, p))
        except OSError:
            continue
    total = sum(s for _, s, _ in entries)
    limit = max(64, int(max_mb)) * 1048576
    for _, size, p in sorted(entries):
        if total <= limit:
            break
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        total -= size


def _cleanup(cfg):
    """.work/ 里的渲染日志每次启动清空; 定位快照只删没有任何工作区状态引用的(定位结果现在跨重启保留);
    state/(断点状态) 与 erased/(擦除缓存)保留, erased/ 只清理无人引用或超过保留天数的;
    logs/ 只在调试模式下才会有运行产物。"""
    try:
        from core.v6_common import LOGS, WORK, cleanup_run_folders
        from core.workspace import cleanup_erased, referenced_locate_folders
        keep = referenced_locate_folders()
        cleanup_run_folders(("locate-",), root=WORK, keep=keep)
        cleanup_run_folders(("render-",), root=WORK)
        if cfg.debug_mode:
            cleanup_run_folders(("locate-",), root=LOGS, keep=keep)
            cleanup_run_folders(("render-",), max_age_hours=24 * 7, root=LOGS)
        keep_names = {Path(k).name for k in keep}
        _enforce_quota(LOGS, cfg.logs_max_mb, keep=("suisen.log", *keep_names))
        _enforce_quota(WORK, cfg.logs_max_mb, keep=("ollama-managed.log", "state", "erased", *keep_names))
        cleanup_erased(float(getattr(cfg, "erase_cache_days", 3)))
    except Exception as exc:
        logging.warning("清理临时目录失败: %s", exc)


def _gl_defaults():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setSwapInterval(1)
    fmt.setSamples(4)          # Qt 的 GL 画笔只有多重采样时才对路径抗锯齿(字幕字形/定位框)
    QSurfaceFormat.setDefaultFormat(fmt)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


def _ensure_player(app, cfg):
    from core import mpv_runtime
    if mpv_runtime.prepare(cfg):
        return True
    if QMessageBox.question(None, "缺少播放组件",
                            "预览播放器需要 libmpv(libmpv-2.dll, 约 40~60MB, 支持断点续传)。现在下载到 models/bin/?\n"
                            "选[No]可稍后在[设置→杂项]指定已有的 libmpv-2.dll 路径。") != QMessageBox.StandardButton.Yes:
        return False
    dlg = QProgressDialog("下载 libmpv…", "取消", 0, 100)
    dlg.setWindowTitle("水仙翻译器")
    dlg.setMinimumDuration(0)
    dlg.show()

    def prog(p, m):
        dlg.setValue(int(p))
        dlg.setLabelText(m)
        app.processEvents()
    try:
        mpv_runtime.ensure_libmpv(cfg, prog, dlg.wasCanceled)
    except Exception as exc:
        QMessageBox.critical(None, "下载失败", f"{exc}\n\n可手动把 libmpv-2.dll 放到 models/bin/ 后重启。")
        return False
    finally:
        dlg.close()
    return mpv_runtime.prepare(cfg)


def main():
    from config import AppConfig
    cfg = AppConfig.load()
    _setup_logging(cfg.debug_mode)
    sys.excepthook = _excepthook
    _cleanup(cfg)
    _gl_defaults()                                   # 必须在 QApplication 之前

    if os.name == "nt":                              # 任务栏图标: Windows 按 AppUserModelID 分组/取图标
        try:                                          # 不设置会归到默认 Python 组, 显示成通用图标
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Suisen.Translator")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from PySide6.QtGui import QFontDatabase
    for font in (Path(BASE) / "models/fonts").glob("*.otf"):
        QFontDatabase.addApplicationFont(str(font))
    app.setApplicationName("水仙翻译器 Suisen Translator")
    logo = os.path.join(resource_dir(), "assets", "logo.png")
    if os.path.isfile(logo):
        app.setWindowIcon(QIcon(logo))
    from gui.theme import apply_theme
    apply_theme(app, cfg.ui_theme)
    if not _ensure_player(app, cfg):
        sys.exit(1)
    from gui.main_window import MainWindow          # 硬件探测在窗口内后台进行, 不阻塞首屏
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
