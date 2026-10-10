"""路径解析: 开发模式 = 项目根目录; 打包(frozen)模式 = exe 同级目录。
规则: config.json/models/logs/.work 等可写数据走 base_dir();
     assets/logo.png 这类只读打包资源走 resource_dir()(PyInstaller onedir 里在 _internal)。"""
import os
import sys


def frozen():
    return bool(getattr(sys, "frozen", False))


def base_dir():
    """可写数据根目录: 打包后是 exe 所在目录; 开发时是项目根目录。"""
    if frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_dir():
    """打包资源(assets 等)的读取位置: frozen 时在 sys._MEIPASS(onedir 的 _internal)。"""
    if frozen():
        return getattr(sys, "_MEIPASS", base_dir())
    return base_dir()


def base_path():
    from pathlib import Path
    return Path(base_dir())
