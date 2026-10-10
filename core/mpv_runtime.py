"""libmpv 运行时: 定位 / 按需下载 libmpv-2.dll, 并在 import mpv(python-mpv) 之前把它加入 DLL 搜索路径。
与 FFmpeg 同样处理(GPL 组件不随包分发): 找不到时从 mpv 官方 Windows 构建(shinchiro)下载 mpv-dev 包,
只解出 libmpv-2.dll 到 models/bin/。也可在 [设置→杂项] 指定现成的 dll 路径。"""
from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from core.paths import base_dir

log = logging.getLogger("suisen.mpv")
BASE = Path(base_dir())
BIN_DIR = BASE / "models" / "bin"
DLL_NAMES = ("libmpv-2.dll", "mpv-2.dll", "mpv-1.dll")
_RELEASES = "https://api.github.com/repos/shinchiro/mpv-winbuild-cmake/releases/latest"
_ASSET_RE = re.compile(r"^mpv-dev-x86_64-\d{8}-git-[0-9a-f]+\.7z$")      # 非 v3 的通用 x86_64 构建
_state = {"module": None, "path": ""}


def locate(cfg=None):
    """返回 libmpv 的文件路径(非 Windows 返回库名); 找不到返回 ''。"""
    cands = []
    configured = str(getattr(cfg, "libmpv_path", "") or "").strip()
    if configured:
        cands.append(Path(configured))
    for folder in (BIN_DIR, BASE, Path(getattr(sys, "_MEIPASS", str(BASE)))):
        cands += [folder / n for n in DLL_NAMES]
    for p in cands:
        if p.is_file():
            return str(p)
    if os.name != "nt":
        import ctypes.util
        return ctypes.util.find_library("mpv") or ""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for n in DLL_NAMES:
            if d and (Path(d) / n).is_file():
                return str(Path(d) / n)
    return ""


def prepare(cfg=None):
    """把 dll 所在目录加入搜索路径(python-mpv 用 ctypes.util.find_library 按名字在 %PATH% 里找)。"""
    path = locate(cfg)
    if not path:
        return False
    if os.name == "nt" and os.path.isfile(path):
        folder = str(Path(path).parent)
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(folder)
        except Exception:
            pass
        try:
            ctypes.CDLL(path)                       # 预加载: 依赖缺失(如 VC 运行库)在这里就能报出来
        except OSError as exc:
            log.warning("libmpv 加载失败 %s: %s", path, exc)
            return False
    _state["path"] = path
    return True


def module():
    """延迟 import python-mpv(必须在 prepare 之后)。"""
    if _state["module"] is None:
        if not _state["path"] and not prepare():
            raise RuntimeError("未找到 libmpv(libmpv-2.dll)。请到[设置→杂项]指定路径, 或重启程序让其自动下载到 models/bin/")
        try:
            import mpv
        except ImportError as exc:
            raise RuntimeError("未安装 python-mpv: pip install python-mpv") from exc
        except OSError as exc:
            raise RuntimeError(f"加载 libmpv 失败: {exc}") from exc
        _state["module"] = mpv
    return _state["module"]


def ensure_libmpv(cfg, progress=lambda p, m: None, abort=None):
    """缺失时下载并解出 libmpv-2.dll(断点续传, 可取消)。返回 dll 路径。"""
    found = locate(cfg)
    if found:
        return found
    if os.name != "nt":
        raise RuntimeError("请用系统包管理器安装 libmpv(如 apt install libmpv2 / brew install mpv)")
    import requests
    from core.model_manager import download
    url = str(getattr(cfg, "libmpv_url", "") or "").strip()
    if not url:
        progress(0, "查询 libmpv 最新构建…")
        r = requests.get(_RELEASES, timeout=(10, 60))
        r.raise_for_status()
        asset = next((a for a in r.json().get("assets", []) if _ASSET_RE.match(a.get("name", ""))), None)
        if asset is None:
            raise RuntimeError("未找到 mpv-dev-x86_64 构建; 请手动下载 libmpv-2.dll 放到 models/bin/")
        url = asset["browser_download_url"]
    archive = BASE / "models" / "downloads" / Path(url).name
    download(url, archive, progress, abort)
    progress(99, "解出 libmpv-2.dll…")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / "libmpv-2.dll"
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            name = next((n for n in z.namelist() if n.replace("\\", "/").endswith("libmpv-2.dll")), None)
            if not name:
                raise RuntimeError("压缩包里没有 libmpv-2.dll")
            target.write_bytes(z.read(name))
    else:
        _extract_7z_libmpv(archive, target)
    for _attempt in range(3):
        try:
            archive.unlink(missing_ok=True)     # 源 7z 删掉失败(被占用)不影响: dll 已解出, 残留包下次清理
            break
        except OSError:
            time.sleep(0.4)                     # 解包进程刚退出时句柄可能还没释放, 等一下再删
    progress(100, "libmpv 已安装到 models/bin/libmpv-2.dll")
    return str(target)


def _extract_7z_libmpv(archive, target):
    """从 7z 解出 libmpv-2.dll: 先试 py7zr(快); 该官方包用 BCJ2 滤镜, py7zr 不支持时
    回退 Windows 10+ 自带的 tar.exe(bsdtar, 支持 BCJ2)。只解出成员, 不解整个包。"""
    try:
        import py7zr
        with py7zr.SevenZipFile(str(archive), "r") as z:
            names = [n for n in z.getnames() if n.replace("\\", "/").endswith("libmpv-2.dll")]
            if not names:
                raise RuntimeError("压缩包里没有 libmpv-2.dll")
            tmp = Path(tempfile.mkdtemp(prefix="suisen_mpv_"))
            try:
                z.extract(str(tmp), targets=[names[0]])      # py7zr 1.x 没有 read(), 解到临时目录再读
                found = next(tmp.rglob("libmpv-2.dll"), None)
                if found is None:
                    raise RuntimeError("解压后未找到 libmpv-2.dll")
                target.write_bytes(found.read_bytes())
                return
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:
        log.debug("py7zr 解压失败, 回退 tar.exe: %s", exc)
    # 回退: Windows 10+ 自带 tar.exe(bsdtar, 支持 BCJ2 滤镜)
    tar = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "tar.exe")
    if not os.path.isfile(tar):
        raise RuntimeError("解压该 7z 需要 py7zr(支持 BCJ2 滤镜)或 Windows 10+ 自带的 tar.exe; "
                           "或手动把 libmpv-2.dll 放到 models/bin/")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run([tar, "-tf", str(archive)], capture_output=True, timeout=120, creationflags=flags)
    member = next((n.strip() for n in r.stdout.decode("utf-8", "replace").splitlines()
                   if n.replace("\\", "/").rstrip("/").endswith("libmpv-2.dll")), None)
    if not member:
        raise RuntimeError("压缩包里没有 libmpv-2.dll")
    tmp = Path(tempfile.mkdtemp(prefix="suisen_mpv_"))
    try:
        r = subprocess.run([tar, "-xf", str(archive), "-C", str(tmp), member],
                           capture_output=True, timeout=300, creationflags=flags)
        found = tmp / member
        if r.returncode != 0 or not found.is_file():
            raise RuntimeError("tar 解压失败: " + r.stderr.decode("utf-8", "replace")[:200])
        target.write_bytes(found.read_bytes())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
