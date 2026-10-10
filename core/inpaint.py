"""无痕擦除引擎: 原字幕区域视频修复(去字), 替代色块遮罩。
- lama:        simple-lama-inpainting, 逐帧; 进程内没有依赖时走共享运行时子进程(常驻, 小批处理)
- opencv:      Telea 修复, 零依赖
- sttn / propainter / diffueraser: third_party 或 models/inpaint 下的 repo, 时序分块
- api:         IOPaint 兼容 HTTP 接口
擦除帧经 rawvideo 管道直喂 FFmpeg(见 v6_render)。"""
import base64
import glob
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import requests

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from core.paths import base_dir

log = logging.getLogger("suisen.inpaint")
BASE = base_dir()
THIRD = os.path.join(BASE, "third_party")

# (名称, 代号, 是否高级)。高级引擎需用户自备 repo/权重, 在设置里收进"高级"区
ENGINES = [("AI 智能擦除 · LAMA(推荐, 首次使用自动下载)", "lama", False),
           ("快速擦除 · 基础修复(立即可用, 效果一般)", "opencv", False),
           ("STTN 视频级修复(需自行放置)", "sttn", True),
           ("ProPainter 视频级修复(禁止商用)", "propainter", True),
           ("DiffuEraser 扩散修复(权重极大)", "diffueraser", True),
           ("自建服务接口(IOPaint 兼容)", "api", True)]
BASIC_ENGINES = [(n, c) for n, c, adv in ENGINES if not adv]
ADVANCED_ENGINES = [(n, c) for n, c, adv in ENGINES if adv]

REPO_DIRS = {"sttn": "STTN", "propainter": "ProPainter",
             "diffueraser": "DiffuEraser", "lama_ext": "lama"}
TEMPORAL = {"sttn", "propainter", "diffueraser"}

DEFAULT_CMDS = {
    "sttn": '{python} test.py --video "{video}" --mask "{maskdir}" --ckpt "checkpoints/sttn.pth"',
    "propainter": '"{python}" inference_propainter.py --video "{framesdir}" '
                  '--mask "{maskdir}" --output "{outdir}" --save_frames --fp16',
    "diffueraser": '{python} run_diffueraser.py --input_video "{video}" '
                   '--input_mask "{maskvideo}" --save_path "{outdir}"',
    "lama_ext": '"{python}" lama_runner.py --frames "{framesdir}" --masks "{maskdir}" --out "{outdir}"',
}


class EraseError(RuntimeError):
    pass


def ffmpeg_bin(cfg=None):
    configured = str(getattr(cfg, "ffmpeg_path", "") if cfg else "").strip()
    if configured and os.path.isfile(configured):
        return configured
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for cand in (os.path.join(BASE, "models", "bin", exe), os.path.join(BASE, exe)):
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffmpeg") or ""


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def ensure_ffmpeg(cfg, progress=lambda p, m: None, abort=None):
    exe = ffmpeg_bin(cfg)
    if exe:
        return exe
    if os.name != "nt":
        raise RuntimeError("未找到 FFmpeg; 请用系统包管理器安装或把它放进 PATH")
    import zipfile
    from core.model_manager import download
    from core.paths import base_path
    target_dir = base_path() / "models" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "ffmpeg.exe"
    archive = target_dir / "ffmpeg-release-essentials.zip"
    progress(0, "未找到 FFmpeg, 开始自动下载(约110MB, 支持断点续传)...")
    download(FFMPEG_URL, archive, progress, abort)
    progress(99, "解压 ffmpeg.exe...")
    with zipfile.ZipFile(archive) as z:
        name = next((n for n in z.namelist() if n.replace("\\", "/").endswith("/bin/ffmpeg.exe")), None)
        if name is None:
            raise RuntimeError("FFmpeg 压缩包结构已变, 未找到 bin/ffmpeg.exe")
        tmp = target.with_name("ffmpeg.exe.tmp")
        with z.open(name) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, target)
    archive.unlink(missing_ok=True)
    progress(100, "FFmpeg 已安装到 models/bin/ffmpeg.exe")
    return str(target)


def check_engine(cfg):
    import importlib.util
    if cfg.erase_engine == "opencv":
        return True, "OpenCV 可用; 适合验证链路, 背景修复质量一般"
    if cfg.erase_engine == "lama":
        if importlib.util.find_spec("simple_lama_inpainting"):
            return True, "进程内依赖已就绪; 首次运行会加载权重"
        from core.model_manager import runtime_has, runtime_python
        root = os.path.join(BASE, "models", "inpaint", "lama")
        weights_ok = os.path.isfile(os.path.join(root, "big-lama.pt"))
        if weights_ok and runtime_python().is_file() and runtime_has("simple_lama_inpainting"):
            return True, "共享运行时已就绪; 以子进程方式执行"
        if weights_ok and os.path.isfile(os.path.join(root, "runtime", "python.exe")):
            return True, "模型库运行时已安装; 以子进程方式执行"
        return False, "未安装 LAMA, 请到[模型库]安装(自动下载运行时与权重)"
    if cfg.erase_engine == "api":
        return bool(cfg.erase_api_url.strip()), "地址检查不等于实际推理测试, 建议先用一小段视频试跑"
    repo = os.path.join(BASE, "models", "inpaint", REPO_DIRS.get(cfg.erase_engine, cfg.erase_engine))
    if not os.path.isdir(repo):
        repo = os.path.join(THIRD, REPO_DIRS.get(cfg.erase_engine, cfg.erase_engine))
    return os.path.isdir(repo), ("代码目录存在; 仍需匹配的 Python/CUDA 依赖与完整权重"
                                 if os.path.isdir(repo) else "请先在模型库安装相应引擎")


# ================= 逐帧工具 =================

def _masked_change(before, after, mask):
    selected = mask > 0
    if not np.any(selected):
        return 0.0
    values = cv2.absdiff(before, after)[selected]
    return float(values.mean()) if values.size else 0.0


def _roi_of(masks_list, w, h, pad=48):
    xs1, ys1, xs2, ys2 = [], [], [], []
    for m in masks_list:
        ys, xs = np.nonzero(m)
        if len(xs):
            xs1.append(xs.min()); xs2.append(xs.max())
            ys1.append(ys.min()); ys2.append(ys.max())
    if not xs1:
        return None
    x1 = max(0, int(min(xs1)) - pad)
    y1 = max(0, int(min(ys1)) - pad)
    x2 = min(w, int(max(xs2)) + pad)
    y2 = min(h, int(max(ys2)) + pad)
    x2 -= (x2 - x1) % 2
    y2 -= (y2 - y1) % 2
    if x2 - x1 < 64 or y2 - y1 < 64:
        x2, y2 = min(w, x1 + 64), min(h, y1 + 64)
        x1, y1 = max(0, x2 - 64), max(0, y2 - 64)
    return x1, y1, x2, y2


def _rois_of(mask, w, h, pad=48):
    """把掩码按 pad 膨胀后连通域拆成若干独立 ROI: 分散的多个字幕框不再合并成一个
    覆盖大片空隙的大 ROI(LAMA 按像素计费, 拆开后处理面积大幅下降, 画质不变);
    相邻(≤2*pad)的框仍合并共享上下文。返回 ROI 列表(空掩码返回 [])。"""
    if not np.any(mask):
        return []
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1))
    dilated = cv2.dilate(mask, kern)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(dilated, 8)
    rois = []
    for i in range(1, n):
        x, y, ww, hh, _area = (int(v) for v in stats[i])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + ww), min(h, y + hh)
        x2 -= (x2 - x1) % 2
        y2 -= (y2 - y1) % 2
        if x2 - x1 < 64 or y2 - y1 < 64:
            x2, y2 = min(w, x1 + 64), min(h, y1 + 64)
            x1, y1 = max(0, x2 - 64), max(0, y2 - 64)
        rois.append((x1, y1, x2, y2))
    rois.sort()
    return rois


def _feather_paste(dst, src, mask, feather=5):
    core = mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(core, (0, 0), max(0.1, feather / 2.0))
    alpha = np.maximum(core, soft)[..., None]
    return np.clip(dst.astype(np.float32) * (1 - alpha) + src.astype(np.float32) * alpha,
                   0, 255).astype(np.uint8)


# ================= 逐帧引擎 =================

def _shrink_pair(crop, mk, max_side):
    """LAMA 输入降采样: ROI 长边超过 max_side 时等比缩小(背景补全是低频任务, 回贴后几乎无损),
    4K 字幕条 ROI 的推理像素数可降 3~4 倍。未缩放时原样返回。"""
    h, w = crop.shape[:2]
    side = max(h, w)
    if max_side <= 0 or side <= max_side:
        return crop, mk
    k = max_side / float(side)
    nw = max(8, int(round(w * k)) // 8 * 8)
    nh = max(8, int(round(h * k)) // 8 * 8)
    c = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mk, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return c, np.where(m > 0, 255, 0).astype(np.uint8)


class LamaEraser:
    def __init__(self, max_side=1920):
        self.max_side = int(max_side or 0)
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError:
            raise EraseError("未安装 simple-lama-inpainting: pip install simple-lama-inpainting")
        try:
            self.model = SimpleLama()
        except Exception as exc:
            raise EraseError(f"LAMA 模型加载失败(首次运行需下载 big-lama 权重; 请检查网络或换用其他擦除引擎): {exc}")
        try:
            test = np.full((64, 64, 3), 40, np.uint8)
            cv2.putText(test, "AB", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            mk = np.zeros((64, 64), np.uint8)
            mk[16:52, 4:60] = 255
            if _masked_change(test, self.erase(test, mk), mk) < 1.0:
                raise EraseError("LAMA 自检未产生像素变化(模型/显卡驱动异常), 请改用 OpenCV 或 API 引擎")
        except EraseError:
            raise
        except Exception as exc:
            raise EraseError(f"LAMA 自检失败: {exc}")

    def erase(self, frame, mask):
        from PIL import Image
        h, w = frame.shape[:2]
        rois = _rois_of(mask, w, h)                # 与子进程路径一致: 分散的多个字幕框各自处理, 不合并成大 ROI
        if not rois:
            return frame
        out = frame.copy()
        for x1, y1, x2, y2 in rois:
            crop, mk = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
            small, smk = _shrink_pair(crop, mk, self.max_side)
            img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            try:
                res = self.model(img, Image.fromarray(smk).convert("L"))
            except Exception as exc:
                raise EraseError(f"LAMA 逐帧修复失败: {exc}")
            res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
            if res.shape[:2] != crop.shape[:2]:
                res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
            out[y1:y2, x1:x2] = _feather_paste(crop, res, mk)
        return out


class CvEraser:
    def erase(self, frame, mask):
        h, w = frame.shape[:2]
        roi = _roi_of([mask], w, h)
        if roi is None:
            return frame
        x1, y1, x2, y2 = roi
        crop, mk = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        res = cv2.inpaint(crop, mk, 5, cv2.INPAINT_TELEA)
        out = frame.copy()
        out[y1:y2, x1:x2] = _feather_paste(crop, res, mk)
        return out


class ApiEraser:
    def __init__(self, url):
        self.url = url.strip()
        if not self.url:
            raise EraseError("擦除 API 地址为空")

    def erase(self, frame, mask):
        h, w = frame.shape[:2]
        roi = _roi_of([mask], w, h)
        if roi is None:
            return frame
        x1, y1, x2, y2 = roi
        crop, mk = frame[y1:y2, x1:x2], mask[y1:y2, x1:x2]
        b_img = base64.b64encode(cv2.imencode(".png", crop)[1]).decode()
        b_msk = base64.b64encode(cv2.imencode(".png", mk)[1]).decode()
        try:
            r = requests.post(self.url, json={"image": b_img, "mask": b_msk}, timeout=120)
        except Exception as e:
            raise EraseError(f"擦除 API 连接失败: {e}")
        if r.status_code != 200:
            raise EraseError(f"擦除 API 返回 {r.status_code}: {r.text[:160]}")
        try:
            data = base64.b64decode(r.json().get("image", ""))
        except Exception:
            data = r.content
        res = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if res is None:
            raise EraseError("擦除 API 返回的图像无法解析")
        if res.shape[:2] != crop.shape[:2]:
            res = cv2.resize(res, (crop.shape[1], crop.shape[0]))
        out = frame.copy()
        out[y1:y2, x1:x2] = _feather_paste(crop, res, mk)
        return out


# ================= 子进程引擎 =================

def _run_cmd(cmd, cwd, abort, tag):
    parts = shlex.split(cmd, posix=(os.name != "nt"))
    if os.name == "nt":
        parts = [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile(mode="w+b") as output:
        proc = subprocess.Popen(parts, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
        while proc.poll() is None:
            if abort():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise EraseError("已取消")
            time.sleep(0.1)
        if proc.returncode:
            output.seek(0)
            tail = output.read().decode("utf-8", "replace")[-3000:]
            raise EraseError(f"{tag}失败:\n{tail}")


def _pngs_to_mp4(png_dir, out_mp4, fps, ffmpeg):
    if not ffmpeg:
        raise EraseError("未找到 FFmpeg")
    cmd = [ffmpeg, "-y", "-framerate", f"{fps:.4f}", "-i", os.path.join(png_dir, "%06d.png"),
           "-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv444p", out_mp4]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    if r.returncode != 0:
        raise EraseError("帧序列转 mp4 失败")


def _read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _find_output(outdir, maskdir):
    for directory in sorted(glob.glob(os.path.join(outdir, "**", "frames"), recursive=True)):
        if glob.glob(os.path.join(directory, "*.png")):
            return directory, "frames"
    candidate = maskdir.rstrip("/\\") + "_result.mp4"
    if os.path.isfile(candidate):
        return candidate, "video"
    files = sorted(glob.glob(os.path.join(outdir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
    return (files[-1], "video") if files else (None, None)


class RepoEraser:
    """chunk 级: 导出裁剪帧+遮罩 → 调 repo → 读回结果回贴。"""

    def __init__(self, name, cfg, fps):
        self.name, self.cfg, self.fps = name, cfg, fps
        self.ffmpeg = ffmpeg_bin(cfg)
        managed = os.path.join(BASE, "models", "inpaint", REPO_DIRS[name])
        self.repo = managed if os.path.isdir(managed) else os.path.join(THIRD, REPO_DIRS[name])
        if not os.path.isdir(self.repo):
            raise EraseError("未安装相应时序引擎, 请到模型库下载")
        self.python = cfg.erase_python.strip()
        if not self.python and not getattr(sys, "frozen", False):
            self.python = sys.executable
        if not self.python:
            try:
                from core.model_manager import runtime_python
                shared = runtime_python()
                self.python = str(shared) if shared.is_file() else ""
            except Exception:
                self.python = ""
        if not self.python:
            raise EraseError("打包版本里没有内置 Python; 请在[设置→原字处理]的高级里为时序引擎指定 Python 路径"
                             "(或到[模型库]安装一次 LAMA 以生成共享运行时)")
        self._server = None
        self._server_log = None
        self._jobdir = None
        self._task_n = 0
        self.max_side = int(getattr(cfg, "erase_lama_max_side", 1920) or 0)

    # ---- lama_ext 常驻子进程: 整个擦除过程只起一个 python, 模型只加载一次 ----
    def _ensure_server(self):
        if self._server is not None and self._server.poll() is None:
            return
        self._jobdir = tempfile.mkdtemp(prefix="suisen_lamasrv_")
        self._server_log = open(os.path.join(self._jobdir, "server.log"), "wb")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        batch = int(getattr(self.cfg, "erase_lama_batch", 1) or 1)
        self._server = subprocess.Popen([self.python, "lama_runner.py", "--serve", self._jobdir,
                                         "--batch", str(batch)],
                                        cwd=self.repo, stdout=self._server_log,
                                        stderr=subprocess.STDOUT, creationflags=flags)

    def submit(self, frames, masks):
        """写出任务目录并触发常驻 serve, 立即返回句柄(供 collect 收尾); 不阻塞等待。
        分散的多个字幕框按各自 ROI 拆成多条(帧,遮罩)对, 减少 LAMA 处理面积。"""
        dirty = [i for i, m in enumerate(masks) if np.any(m)]
        if not dirty:
            return (list(frames), list(masks), [], None, [])
        self._ensure_server()
        self._task_n += 1
        task = os.path.join(self._jobdir, f"task-{self._task_n:06d}")
        fdir, mdir = os.path.join(task, "frames"), os.path.join(task, "masks")
        os.makedirs(fdir)
        os.makedirs(mdir)
        boxes = []                              # [(帧索引, (x1,y1,x2,y2)), ...] 每条输出一对
        j = 0
        for i in dirty:
            f, m = frames[i], masks[i]
            h, w = f.shape[:2]
            for x1, y1, x2, y2 in _rois_of(m, w, h):
                boxes.append((i, (x1, y1, x2, y2)))
                cs, ms = _shrink_pair(f[y1:y2, x1:x2], m[y1:y2, x1:x2], self.max_side)
                png = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]            # 临时文件: 最低压缩, 省 CPU
                cv2.imwrite(os.path.join(fdir, f"{j:06d}.png"), cs, png)
                cv2.imwrite(os.path.join(mdir, f"{j:06d}.png"), ms, png)
                j += 1
        open(os.path.join(task, "ready"), "w").close()
        return (list(frames), list(masks), boxes, task, dirty)

    def collect(self, handle, abort):
        """等待句柄对应的任务完成并读回结果(顺序与输入一致); 出错抛 EraseError。"""
        frames, masks, boxes, task, dirty = handle
        if not dirty:
            return frames
        try:
            done, error = os.path.join(task, "done"), os.path.join(task, "error")
            while True:
                if abort():
                    raise EraseError("已取消")
                if os.path.isfile(done):
                    break
                if os.path.isfile(error):
                    detail = ""
                    try:
                        detail = open(error, encoding="utf-8", errors="replace").read()[:400]
                    except OSError:
                        pass
                    raise EraseError("LAMA 修复失败: " + detail)
                if self._server.poll() is not None:
                    tail = ""
                    try:
                        self._server_log.flush()
                        tail = open(self._server_log.name, encoding="utf-8", errors="replace").read()[-1500:]
                    except Exception:
                        pass
                    raise EraseError(f"LAMA 子进程意外退出({self._server.returncode}):\n{tail}")
                time.sleep(0.02)
            pngs = sorted(glob.glob(os.path.join(task, "out", "frames", "*.png")))
            if len(pngs) < len(boxes):
                raise EraseError(f"LAMA 输出帧数不足({len(pngs)}/{len(boxes)})")
            out = list(frames)
            touched = {}                          # 同帧多 ROI: 修复结果累计回贴到同一帧
            for j, p in enumerate(pngs[:len(boxes)]):
                i, (x1, y1, x2, y2) = boxes[j]
                f, m = frames[i], masks[i]
                r = cv2.imread(p)
                if r is None:
                    raise EraseError(f"LAMA 输出帧无法读取: {p}")
                if r.shape[:2] != (y2 - y1, x2 - x1):
                    r = cv2.resize(r, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
                if i not in touched:
                    touched[i] = frames[i].copy()
                    out[i] = touched[i]
                g = touched[i]
                g[y1:y2, x1:x2] = _feather_paste(g[y1:y2, x1:x2], r, m[y1:y2, x1:x2])
            return out
        finally:
            if task:
                shutil.rmtree(task, ignore_errors=True)

    def _serve_chunk(self, frames, masks, abort):
        return self.collect(self.submit(frames, masks), abort)

    def close(self):
        if self._server is None:
            return
        try:
            if self._server.poll() is None:
                try:
                    open(os.path.join(self._jobdir, "_stop"), "w").close()
                    self._server.wait(timeout=8)
                except Exception:
                    self._server.terminate()
                    try:
                        self._server.wait(timeout=5)
                    except Exception:
                        self._server.kill()
        finally:
            try:
                if self._server_log is not None:
                    self._server_log.close()
            except Exception:
                pass
            if self._jobdir:
                shutil.rmtree(self._jobdir, ignore_errors=True)
            self._server = self._server_log = self._jobdir = None

    def erase_chunk(self, frames, masks, abort):
        if self.name == "lama_ext":
            return self._serve_chunk(frames, masks, abort)
        h, w = frames[0].shape[:2]
        roi = _roi_of(masks, w, h)
        if roi is None:
            return frames
        x1, y1, x2, y2 = roi
        tmp = tempfile.mkdtemp(prefix="suisen_erase_")
        try:
            fdir, mdir, odir = (os.path.join(tmp, d) for d in ("frames", "masks", "out"))
            for d in (fdir, mdir, odir):
                os.makedirs(d, exist_ok=True)
            for i, (f, m) in enumerate(zip(frames, masks)):
                cv2.imwrite(os.path.join(fdir, f"{i:06d}.png"), f[y1:y2, x1:x2])
                cv2.imwrite(os.path.join(mdir, f"{i:06d}.png"), m[y1:y2, x1:x2])
            video = os.path.join(tmp, "in.mp4")
            maskvideo = os.path.join(tmp, "mask.mp4")
            if self.name in ("sttn", "diffueraser"):
                _pngs_to_mp4(fdir, video, self.fps, self.ffmpeg)
            if self.name == "diffueraser":
                _pngs_to_mp4(mdir, maskvideo, self.fps, self.ffmpeg)
            tpl = (self.cfg.erase_cmd or {}).get(self.name) or DEFAULT_CMDS[self.name]
            cmd = tpl.format(python=self.python, repo=self.repo, video=video, maskvideo=maskvideo,
                             framesdir=fdir, maskdir=mdir, outdir=odir)
            _run_cmd(cmd, self.repo, abort, f"擦除引擎[{self.name}]")
            out_path, kind = _find_output(odir, mdir)
            if out_path is None:
                raise EraseError(f"{self.name} 未产出结果文件, 请检查命令模板与 repo 输出约定")
            if kind == "video":
                res = _read_video_frames(out_path)
            else:
                res = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(out_path, "*.png")))]
            if len(res) < len(frames):
                raise EraseError(f"{self.name} 输出帧数不足({len(res)}/{len(frames)})")
            out = []
            for f, m, r in zip(frames, masks, res):
                if r.shape[:2] != (y2 - y1, x2 - x1):
                    r = cv2.resize(r, (x2 - x1, y2 - y1))
                g = f.copy()
                g[y1:y2, x1:x2] = _feather_paste(f[y1:y2, x1:x2], r, m[y1:y2, x1:x2])
                out.append(g)
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SafeEraser:
    """逐帧兜底: 主引擎异常或输出零变化时, 该帧改用 OpenCV 修复, 并在完成信息里计数。"""

    def __init__(self, primary, note=""):
        self.primary = primary
        self.cv = CvEraser()
        self.note = note
        self.fallback_frames = 0

    def erase(self, frame, mask):
        out = None
        if self.primary is not None:
            try:
                out = self.primary.erase(frame, mask)
            except Exception as exc:
                log.warning("主擦除引擎本帧失败, 回退 OpenCV: %s", exc)
        if out is None or _masked_change(frame, out, mask) < 0.8:
            out = self.cv.erase(frame, mask)
            self.fallback_frames += 1
        return out


class ExternalLamaEraser(RepoEraser):
    """LAMA 子进程执行: 优先共享运行时 models/runtime/py311, 其次旧版 lama/runtime。
    逐帧模型 → 按小批(无重叠)送子进程, 不缓冲长序列。"""

    def __init__(self, cfg, fps):
        try:
            super().__init__("lama_ext", cfg, fps)
        except EraseError:
            raise EraseError("LAMA 未安装, 请到[模型库]安装(自动下载运行时与权重)")
        from core.model_manager import runtime_python
        shared = runtime_python()
        legacy = os.path.join(self.repo, "runtime", "python.exe")
        if shared.is_file():
            self.python = str(shared)
        elif os.path.isfile(legacy):
            self.python = legacy
        if not self.python:
            raise EraseError("LAMA 运行时缺失, 请到[模型库]重新安装 LAMA")


class ParallelLamaEraser:
    """并行 LAMA: N 个独立 serve 子进程(各自 jobdir 与模型实例), 一批帧按连续段切给各进程并发执行。
    n<=1 时与单进程完全一致。显存代价: 每个进程约多占 1.5~2GB(GPU 有余量才建议 >1)。"""

    def __init__(self, cfg, fps, workers=1):
        n = max(1, int(workers or 1))
        self._workers = [ExternalLamaEraser(cfg, fps) for _ in range(n)]

    def erase_chunk(self, frames, masks, abort):
        if len(self._workers) <= 1 or len(frames) <= 1:
            return self._workers[0].erase_chunk(frames, masks, abort)
        n = min(len(self._workers), len(frames))
        per = (len(frames) + n - 1) // n
        handles = []                                # 先全部提交(非阻塞), 再按序收集
        for w, a in zip(self._workers, range(0, len(frames), per)):
            b = min(len(frames), a + per)
            handles.append((w, w.submit(frames[a:b], masks[a:b])))
        out = []
        for w, handle in handles:
            out.extend(w.collect(handle, abort))
        return out

    def close(self):
        for w in self._workers:
            try:
                w.close()
            except Exception:
                pass


def get_eraser(cfg, fps):
    """返回 (引擎, 模式)。模式: frame=逐帧 erase() / batch=小批 erase_chunk(无重叠) / temporal=时序分块"""
    engine = cfg.erase_engine
    if engine in TEMPORAL:
        return RepoEraser(engine, cfg, fps), "temporal"
    if engine == "opencv":
        return CvEraser(), "frame"
    if engine == "api":
        return SafeEraser(ApiEraser(cfg.erase_api_url), "api"), "frame"
    if engine == "lama":
        import importlib.util
        if importlib.util.find_spec("simple_lama_inpainting"):
            return SafeEraser(LamaEraser(int(getattr(cfg, "erase_lama_max_side", 1920) or 0)), "lama"), "frame"
        n = int(getattr(cfg, "erase_lama_workers", 1) or 1)
        if n > 1:
            return ParallelLamaEraser(cfg, fps, n), "batch"
        return ExternalLamaEraser(cfg, fps), "batch"
    raise EraseError(f"未知擦除引擎: {engine}")
