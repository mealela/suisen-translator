from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import requests
from PySide6.QtCore import QThread, Signal

from config import (ALT_MT_TAG, BIG_VLM_TAG, DEFAULT_MT_TAG, DEFAULT_VLM_TAG,
                    LIGHT_MT_TAG, LIGHT_VLM_TAG, MANAGED_OLLAMA)
from core.paths import base_dir

BASE = Path(base_dir())
ROOT = BASE / "models"
MODELS_DIR = str(ROOT)
RUNTIME_DIR = ROOT / "runtime" / "py311"
OLLAMA_URL = MANAGED_OLLAMA
_service = None
_service_lock = threading.Lock()
log = logging.getLogger("suisen.mm")
_MARKER = BASE / ".work" / "ollama-managed.json"     # 我们自己拉起的托管 Ollama 的 PID 标记(回收用)


def _write_marker(pid):
    try:
        _MARKER.parent.mkdir(exist_ok=True)
        _MARKER.write_text(json.dumps({"pid": int(pid), "suisen_pid": os.getpid(), "time": time.time()}),
                           encoding="utf-8")
    except OSError:
        pass


def _read_marker():
    try:
        return json.loads(_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _remove_marker():
    try:
        _MARKER.unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid):
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(int(pid))
    except Exception:
        return False


def _pid_is_ollama(pid):
    try:
        import psutil
        return "ollama" in psutil.Process(int(pid)).name().lower()
    except Exception:
        return False


def _reclaim_orphan_if_ours():
    """端口被占时, 若占用者是我们自己拉起、母体已死的托管 Ollama(标记文件可证) → 回收它。
    三重确认(标记 PID + 母体已死 + 仍是 ollama 进程), 绝不误杀系统/别的实例的 Ollama。
    返回 True=已回收可继续启动; False=不是我们的, 不动。"""
    marker = _read_marker()
    if not marker:
        return False
    pid = int(marker.get("pid") or 0)
    suisen_pid = int(marker.get("suisen_pid") or 0)
    if pid <= 0 or suisen_pid <= 0:
        return False
    if _pid_alive(suisen_pid):
        return False                      # 母体还活着: 那是另一个运行中实例的, 不动
    if not (_pid_alive(pid) and _pid_is_ollama(pid)):
        _remove_marker()                  # 标记里的进程已不在: 端口是别的进程占的, 不动
        return False
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
        log.info("回收残留的托管 Ollama(旧实例子进程 PID %d)", pid)
    except Exception:
        pass
    _remove_marker()
    return True

# size_gb 用于下载前估算总体积(已按 Ollama 库页面核实)
REGISTRY = [
    dict(id="qwen3vl8", kind="ollama_vlm", tag=DEFAULT_VLM_TAG,
         name="阿里 · Qwen3-VL 8B(默认读字·推荐)", size="约6.1 GB", size_gb=6.1,
         desc="看得懂画面的AI, 负责读出字幕文字。建议显存 8GB / 内存 16GB 以上。"),
    dict(id="glmocr", kind="ollama_vlm", tag=LIGHT_VLM_TAG,
         name="智谱 · GLM-OCR Q8(轻量读字·低配)", size="约1.6 GB", size_gb=1.6,
         desc="更省资源的读字模型, 低配电脑或纯CPU可用。只能读字, 不能判别字幕/水印。"),
    dict(id="qwen38", kind="ollama_vlm", tag=BIG_VLM_TAG,
         name="阿里 · Qwen3.8 27B(高配, 原生多模态)", size="约18 GB", size_gb=18.0,
         desc="质量更高, 需要 24GB 以上显存或 32GB 内存。"),
    dict(id="qwen3_8b", kind="ollama_llm", tag=DEFAULT_MT_TAG,
         name="阿里 · Qwen3 8B 翻译(默认翻译·推荐)", size="约5.2 GB", size_gb=5.2,
         desc="通用大模型, 中日英字幕翻译质量好; 建议显存 8GB 或内存 16GB 以上。"),
    dict(id="deepseek_r1_8b", kind="ollama_llm", tag=ALT_MT_TAG,
         name="DeepSeek-R1 8B 翻译(备选)", size="约5.2 GB", size_gb=5.2,
         desc="推理型模型, 长句/语境更强, 略慢; 已自动关闭思考输出。"),
    dict(id="qwen3_4b", kind="ollama_llm", tag=LIGHT_MT_TAG,
         name="阿里 · Qwen3 4B 翻译(低配)", size="约2.6 GB", size_gb=2.6,
         desc="低配电脑 / 纯CPU 可用的翻译模型。"),
    dict(id="lama", kind="erase", name="LAMA · 智能擦除(默认擦除·推荐)",
         size="约0.5 GB(无N卡) / 约3 GB(有N卡)", size_gb=0.5,
         desc="把原字幕从画面上抹掉并自动补全背景; 首次安装自动下载独立运行时与权重: "
              "有 NVIDIA 显卡装 CUDA 版 torch(约2.6GB, 自动加速), 否则装 CPU 版(约200MB)。"),
    dict(id="propainter", kind="erase", name="ProPainter · 高级擦除",
         size="依官方文件", size_gb=1.0,
         desc="视频级修复, 质量更高; 注意其许可证禁止商用。"),
]

RECOMMENDED_IDS = ("qwen3vl8", "qwen3_8b", "lama")
_RANK = {"qwen38": 3, "qwen3vl8": 2, "glmocr": 1,
         "qwen3_8b": 3, "deepseek_r1_8b": 2, "qwen3_4b": 1}
_MIN_RAM_GB = {"qwen38": 24, "qwen3vl8": 10, "glmocr": 4,
               "qwen3_8b": 12, "deepseek_r1_8b": 12, "qwen3_4b": 6}


def _cancel(abort):
    if abort and abort():
        raise RuntimeError("已取消")


LAMA_WEIGHTS_URL = ("https://github.com/enesmsahin/simple-lama-inpainting/"
                    "releases/download/v0.1.0/big-lama.pt")
PY_EMBED_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

# LAMA 共享运行时的 torch: 有 NVIDIA 驱动装 CUDA 构建, 否则装 CPU 构建; 版本锁定防止升级破坏兼容
TORCH_CUDA = ("2.6.0+cu124", "https://download.pytorch.org/whl/cu124", "约2.6GB")
TORCH_CPU = ("2.6.0+cpu", "https://download.pytorch.org/whl/cpu", "约200MB")
TORCH_PINS = (TORCH_CUDA[0], TORCH_CPU[0])


def _run_stage(cmd, cwd, abort, stage):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as output:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), stdout=output,
                                stderr=subprocess.STDOUT, creationflags=flags)
        while proc.poll() is None:
            _cancel(abort)
            time.sleep(0.2)
        _cancel(abort)
        if proc.returncode:
            output.seek(0)
            tail = output.read().decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"{stage}失败:\n{tail}")


def runtime_python():
    return RUNTIME_DIR / "python.exe"


def runtime_has(package_dir):
    return (RUNTIME_DIR / "Lib" / "site-packages" / package_dir).is_dir()


def ensure_runtime(progress=lambda p, m: None, abort=None):
    py = runtime_python()
    if py.is_file():
        return py
    if os.name != "nt":
        raise RuntimeError("非 Windows 请使用系统 Python 并自行安装所需依赖")
    progress(0, "下载 Python 运行时…")
    archive = ROOT / "downloads" / Path(PY_EMBED_URL).name
    download(PY_EMBED_URL, archive, progress, abort)
    safe_extract(archive, str(RUNTIME_DIR))
    pth = next(RUNTIME_DIR.glob("python*._pth"))
    text = pth.read_text(encoding="utf-8")
    if "#import site" in text:
        pth.write_text(text.replace("#import site", "import site"), encoding="utf-8")
    progress(6, "安装 pip…")
    getpip = ROOT / "downloads" / "get-pip.py"
    download(GET_PIP_URL, getpip, progress, abort)
    _run_stage([py, getpip], RUNTIME_DIR, abort, "pip 安装")
    return py


def _runtime_torch_version():
    py = runtime_python()
    if not py.is_file():
        return ""
    try:
        r = subprocess.run([str(py), "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True, timeout=90,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def pip_install(packages, progress=lambda p, m: None, abort=None, note="", extra_args=()):
    py = ensure_runtime(progress, abort)
    progress(10, (note or "安装依赖") + ": " + " ".join(packages))
    _run_stage([py, "-m", "pip", "install", "--no-cache-dir", *extra_args, *packages],
               RUNTIME_DIR, abort, note or "依赖安装")
    return py


def _install_lama(progress, abort):
    """安装 LAMA: 权重 + 共享运行时里的 torch(按有无 N 卡选 CUDA/CPU 构建) + simple-lama-inpainting。"""
    import importlib.util
    from core.hardware import cuda_capable
    from core.paths import resource_dir

    lama_dir = ROOT / "inpaint/lama"
    lama_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(Path(resource_dir()) / "assets/lama_runner.py"), lama_dir / "lama_runner.py")

    weights = lama_dir / "big-lama.pt"
    if not weights.is_file():
        progress(0, "下载 LAMA 权重(约200MB)…")
        download(LAMA_WEIGHTS_URL, weights, progress, abort)

    if importlib.util.find_spec("simple_lama_inpainting"):
        return
    have = _runtime_torch_version()
    if runtime_python().is_file() and runtime_has("simple_lama_inpainting") and have in TORCH_PINS:
        return
    if have not in TORCH_PINS:
        pin, index, size = TORCH_CUDA if cuda_capable() else TORCH_CPU
        flavor = "CUDA 版, 有 N 卡自动加速" if pin == TORCH_CUDA[0] else "CPU 版(未检测到 NVIDIA 显卡)"
        progress(5, f"安装 torch {pin}({flavor}; {size}, 固定只下这个版本)…")
        pip_install([f"torch=={pin}"], progress, abort, "LAMA 依赖安装(torch)",
                    extra_args=("--index-url", index))
    if not runtime_has("simple_lama_inpainting"):
        progress(70, "安装 simple-lama-inpainting…")
        pip_install(["simple-lama-inpainting"], progress, abort, "LAMA 依赖安装(simple-lama)")
    progress(95, "LAMA 运行时安装完成")


def _json(url):
    r = requests.get(url, timeout=(10, 60))
    r.raise_for_status()
    return r.json()


def load_registry():
    entries = {x["id"]: dict(x) for x in REGISTRY}
    path = ROOT / "registry.json"
    if path.is_file():
        try:
            for item in json.loads(path.read_text(encoding="utf-8")):
                if isinstance(item, dict) and item.get("id"):
                    entries[item["id"]] = item
        except (ValueError, OSError):
            pass
    return list(entries.values())


def marker(item):
    return ROOT / "installed" / (item["id"] + ".json")


def is_installed(item):
    kind = item["kind"]
    if item["id"] == "lama":
        if not (ROOT / "inpaint/lama/big-lama.pt").is_file():
            return False
        import importlib.util
        if importlib.util.find_spec("simple_lama_inpainting"):
            return True
        if runtime_python().is_file() and runtime_has("simple_lama_inpainting"):
            return True
        return (ROOT / "inpaint/lama/runtime/python.exe").is_file()
    if item["id"] == "propainter":
        p = ROOT / "inpaint/ProPainter"
        return ((p / "inference_propainter.py").is_file()
                and all((p / "weights" / x).is_file() for x in (
                    "ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")))
    if kind.startswith("ollama_"):
        return marker(item).is_file() and (ROOT / "ollama").is_dir()
    return marker(item).is_file()


def download(url, destination, progress, abort, expected=None, sha256=None):
    """同一 URL 断点续传。416 不冒充下载成功。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(destination) + ".part")
    info = Path(str(part) + ".json")
    old = {}
    if info.exists():
        try:
            old = json.loads(info.read_text(encoding="utf-8"))
        except ValueError:
            pass
    if old.get("url") != url:
        part.unlink(missing_ok=True)
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
        if old.get("etag"):
            headers["If-Range"] = old["etag"]
    _cancel(abort)
    with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as response:
        if response.status_code == 416:
            part.unlink(missing_ok=True)
            info.unlink(missing_ok=True)
            raise RuntimeError("断点失效; 已清理临时分片, 请再次下载")
        response.raise_for_status()
        if response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {pos}-"):
                raise RuntimeError("服务端返回了不匹配的 Content-Range")
        else:
            pos = 0
        total = expected or (pos + int(response.headers.get("Content-Length") or 0))
        info.write_text(json.dumps({"url": url, "etag": response.headers.get("ETag", "")}), encoding="utf-8")
        with part.open("ab" if pos else "wb") as output:
            for chunk in response.iter_content(1024 * 256):
                _cancel(abort)
                if not chunk:
                    continue
                output.write(chunk)
                pos += len(chunk)
                progress(min(98, int(pos * 100 / total)) if total else 0,
                         f"{destination.name}: {pos / 1048576:.1f} MB")
    if total and pos != total:
        raise RuntimeError(f"文件长度不符: {pos}/{total}")
    if sha256:
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                _cancel(abort)
                digest.update(block)
        if digest.hexdigest().lower() != sha256.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError("SHA256 校验失败, 已删除分片")
    _cancel(abort)
    os.replace(part, destination)
    info.unlink(missing_ok=True)


def safe_extract(archive, folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    def target(name):
        p = (folder / name.replace("\\", "/")).resolve()
        if p != folder and folder not in p.parents:
            raise RuntimeError("压缩包包含越界路径")
        return p

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                mode = (item.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RuntimeError("拒绝压缩包符号链接")
                p = target(item.filename)
                if item.is_dir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(item) as src, p.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as t:
            for item in t.getmembers():
                p = target(item.name)
                if item.isdir():
                    p.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(item) as src, p.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise RuntimeError("拒绝非普通 tar 成员")


def _runtime(progress, abort):
    existing = shutil.which("ollama")
    if existing:
        return existing
    exe = ROOT / "bin/ollama/ollama.exe"
    if exe.is_file():
        return str(exe)
    if os.name != "nt":
        raise RuntimeError("请先安装官方 Ollama, 再重新启用模型")
    release = _json("https://api.github.com/repos/ollama/ollama/releases/latest")
    asset = next((a for a in release["assets"] if a["name"] == "ollama-windows-amd64.zip"), None)
    if asset is None:
        raise RuntimeError("官方发行资产名已变化, 请手动安装 Ollama")
    archive = ROOT / "downloads" / asset["name"]
    digest = str(asset.get("digest") or "")
    download(asset["browser_download_url"], archive, progress, abort, asset.get("size"),
             digest[7:] if digest.startswith("sha256:") else None)
    safe_extract(archive, exe.parent)
    if not exe.is_file():
        raise RuntimeError("运行时压缩包中未找到 ollama.exe")
    return str(exe)


_PULL_TEXT = (("pulling manifest", "获取模型清单…"), ("verifying", "校验文件…"),
              ("writing manifest", "写入清单…"), ("removing", "清理旧层…"), ("success", "完成"))


def _pull_message(status, done, total):
    s = str(status or "")
    for prefix, text in _PULL_TEXT:
        if s.startswith(prefix):
            return text
    if total:
        return f"下载 {done / 1048576:.0f} / {total / 1048576:.0f} MB"
    return s or "连接 Ollama…"


def ensure_ollama(progress=lambda p, m: None, abort=None):
    global _service
    with _service_lock:
        if _service is not None and _service.poll() is None:
            return OLLAMA_URL
        try:
            busy = requests.get(OLLAMA_URL + "/api/version", timeout=1).ok
        except requests.RequestException:
            busy = False
        if busy:
            if not _reclaim_orphan_if_ours():
                raise RuntimeError("11435 端口已被其他 Ollama 占用。请关闭旧的 Suisen 实例后重试。")
            for _ in range(40):                 # 回收后等端口释放
                try:
                    if not requests.get(OLLAMA_URL + "/api/version", timeout=1).ok:
                        break
                except requests.RequestException:
                    break
                time.sleep(0.2)
        exe = _runtime(progress, abort)
        environment = dict(os.environ)
        environment["OLLAMA_HOST"] = "127.0.0.1:11435"
        environment["OLLAMA_MODELS"] = str(ROOT / "ollama")
        environment["OLLAMA_NUM_PARALLEL"] = "1"
        environment["OLLAMA_MAX_LOADED_MODELS"] = "1"
        work = BASE / ".work"
        work.mkdir(exist_ok=True)
        log_path = work / "ollama-managed.log"
        with log_path.open("ab") as output:
            _service = subprocess.Popen([exe, "serve"], env=environment, stdout=output,
                                        stderr=subprocess.STDOUT,
                                        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        _write_marker(_service.pid)             # 标记"这个托管 Ollama 是我拉起的"(供下次启动回收孤儿)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            _cancel(abort)
            if _service.poll() is not None:
                raise RuntimeError(f"Ollama 启动失败, 请查看 {log_path}")
            try:
                if requests.get(OLLAMA_URL + "/api/version", timeout=1).ok:
                    return OLLAMA_URL
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"Ollama 启动超时, 请查看 {log_path}")


def find_item(item_id=None, tag=None):
    for x in load_registry():
        if item_id and x.get("id") == item_id:
            return x
        if tag and x.get("tag") == tag:
            return x
    return None


def _total_ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1073741824
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1073741824
    except Exception:
        return 16.0


def device_allows(item):
    return _total_ram_gb() >= _MIN_RAM_GB.get(item.get("id", ""), 0)


def pick_auto(kind):
    cands = [x for x in load_registry() if x.get("kind") == kind and is_installed(x)]
    fit = [x for x in cands if device_allows(x)] or cands
    if not fit:
        return None
    return max(fit, key=lambda x: _RANK.get(x.get("id", ""), 0))


def is_managed(ep):
    return (ep.get("provider") == "ollama"
            and ep.get("ollama_url", "").rstrip("/") == OLLAMA_URL)


def align_endpoints(cfg):
    changed = []
    for kind, attr in (("ollama_vlm", "ep_vlm"), ("ollama_llm", "ep_translate")):
        ep = dict(getattr(cfg, attr) or {})
        if not is_managed(ep):
            continue
        cur = find_item(tag=ep.get("ollama_model", ""))
        if cur is not None and is_installed(cur):
            continue
        best = pick_auto(kind)
        if best and best.get("tag") != ep.get("ollama_model"):
            ep["ollama_model"] = best["tag"]
            setattr(cfg, attr, ep)
            changed.append(best["name"])
    if changed:
        cfg.save()
    return changed


def required_items(cfg):
    """当前配置真正需要的本地模型: 只按实际会走到的路线判定。"""
    items = []
    need_vlm = (str(cfg.ocr_engine) in ("vlm", "auto")
                or (bool(getattr(cfg, "locator_vlm_verify", True))
                    and str(getattr(cfg, "locator_verify_mode", "ambiguous")) != "off"))
    need_mt = bool(getattr(cfg, "pipe_translate", True)) or bool(getattr(cfg, "filter_llm", True))
    for attr, need in (("ep_vlm", need_vlm), ("ep_translate", need_mt)):
        if not need:
            continue
        ep = dict(getattr(cfg, attr) or {})
        if is_managed(ep):
            it = find_item(tag=ep.get("ollama_model", ""))
            if it:
                items.append(it)
    if (getattr(cfg, "pipe_erase", True) and cfg.placement == "erase"
            and cfg.erase_engine in ("lama", "propainter")):
        it = find_item(item_id=cfg.erase_engine)
        if it:
            items.append(it)
    return items


def missing_required(cfg):
    return [x for x in required_items(cfg) if not is_installed(x)]


def total_size_gb(items):
    return sum(float(x.get("size_gb", 0) or 0) for x in items)


def ensure_endpoint_ready(ep, progress=lambda p, m: None, abort=None):
    """显式下载入口(带进度、可取消): 托管端点 → 启动服务; 模型没下载则当场下载。"""
    if not is_managed(ep):
        return
    ensure_ollama(progress, abort)
    item = find_item(tag=ep.get("ollama_model", ""))
    if item is not None and not is_installed(item):
        install_sync(item, progress, abort)


def prepare_endpoint(ep):
    """客户端构造时调用: 托管端点只负责把服务拉起来; 模型未下载直接报错——
    绝不在没有进度、不能取消的地方静默下载数 GB。"""
    if not is_managed(ep):
        return
    ensure_ollama()
    item = find_item(tag=ep.get("ollama_model", ""))
    if item is not None and not is_installed(item):
        raise RuntimeError(f"本地模型 [{item['name']}] 尚未下载: 请到[模型库]安装, 或在[设置]换用其他模型")


def shutdown():
    global _service
    if _service is not None and _service.poll() is None:
        _service.terminate()
        try:
            _service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _service.kill()
            _service.wait()
    _service = None
    _remove_marker()


atexit.register(shutdown)


def install_sync(item, progress=lambda p, m: None, abort=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    _cancel(abort)
    if is_installed(item):
        return
    kind = item["kind"]
    if kind.startswith("ollama_"):
        base = ensure_ollama(progress, abort)
        success = False
        with requests.post(base + "/api/pull", json={"model": item["tag"], "stream": True},
                           stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            layers = {}
            for line in response.iter_lines():
                _cancel(abort)
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                digest = data.get("digest")
                if digest and data.get("total"):
                    layers[digest] = (int(data.get("completed") or 0), int(data["total"]))
                done = sum(c for c, _ in layers.values())
                total = sum(t for _, t in layers.values())
                progress(min(99, int(done * 100 / total)) if total else 0,
                         _pull_message(data.get("status"), done, total))
                success = data.get("status") == "success"
        if not success:
            raise RuntimeError("Ollama 没有确认拉取成功")
    elif item["id"] == "lama":
        _install_lama(progress, abort)
    elif item["id"] == "propainter":
        commit = _json("https://api.github.com/repos/sczhou/ProPainter/commits/main")["sha"]
        archive = ROOT / "downloads" / f"propainter-{commit}.zip"
        download(f"https://github.com/sczhou/ProPainter/archive/{commit}.zip", archive, progress, abort)
        destination = ROOT / "inpaint/ProPainter"
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                safe_extract(archive, temp)
                entry = next(Path(temp).rglob("inference_propainter.py"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry.parent), str(destination))
        for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
            target = destination / "weights" / name
            if not target.is_file():
                download(f"https://github.com/sczhou/ProPainter/releases/download/v0.1.0/{name}",
                         target, progress, abort)
    else:
        raise RuntimeError("此自定义条目没有安装适配器; 请使用已支持的 kind 及字段")
    _cancel(abort)
    m = marker(item)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(100, "下载完成")


def apply_to_config(item, cfg):
    """[启用]: 只改与该模型直接相关的字段, 不覆盖用户的其他选择(如"译文放下方")。"""
    kind = item["kind"]
    if kind in ("ollama_vlm", "ollama_llm"):
        attr = "ep_vlm" if kind == "ollama_vlm" else "ep_translate"
        ep = dict(getattr(cfg, attr) or {})
        ep.update(provider="ollama", ollama_url=OLLAMA_URL, ollama_model=item["tag"])
        setattr(cfg, attr, ep)
        if kind == "ollama_vlm":
            cfg.ocr_engine = "vlm"
        note = "已设为当前读字模型。" if kind == "ollama_vlm" else "已设为当前翻译模型。"
    elif item["id"] in ("lama", "propainter"):
        cfg.erase_engine = item["id"]
        if cfg.placement == "cover":
            cfg.placement = "erase"
        note = ("已设为擦除引擎" + ("。" if cfg.placement == "erase"
                                    else "; 当前译文位置为[保留原字幕, 译文放下方], 要擦除请到[设置→原字处理]切换。"))
    else:
        note = ""
    cfg.save()
    return "配置已更新。" + note


class InstallWorker(QThread):
    progress = Signal(int, str)
    done = Signal(str, str)
    failed = Signal(str, str)

    def __init__(self, item, parent=None):
        super().__init__(parent)
        self.item = item
        self._abort = False
        self.finished.connect(self.deleteLater)

    def stop(self):
        self._abort = True

    def run(self):
        try:
            install_sync(self.item, self.progress.emit, lambda: self._abort)
            self.done.emit(self.item["id"], "安装完成")
        except Exception as exc:
            self.failed.emit(self.item["id"], str(exc))
