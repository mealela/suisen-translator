"""工作区状态持久化: 每个视频的定位/识别/翻译结果与擦除缓存记录在 .work/state/,
重启后队列从上次停下的阶段继续。擦除中间视频放在 .work/erased/, 只在导出时被使用。"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from core.models import clone_state, restore_state
from core.v6_common import WORK

STATE_DIR = WORK / "state"
ERASED_DIR = WORK / "erased"


def video_key(video):
    return hashlib.sha1(os.path.abspath(str(video)).encode("utf-8")).hexdigest()[:16]


def state_path(video):
    return STATE_DIR / (video_key(video) + ".json")


def erased_path(video, signature):
    return ERASED_DIR / f"{video_key(video)}-{signature[:12]}.mp4"


def purge_erased_for(video, keep=None):
    if not ERASED_DIR.is_dir():
        return
    prefix = video_key(video) + "-"
    for p in ERASED_DIR.glob(prefix + "*"):
        if keep and p.resolve() == Path(keep).resolve():
            continue
        try:
            p.unlink()
        except OSError:
            pass


def _segments_payload(segments):
    from core.pipeline import segs_brief
    out = []
    for s in segs_brief(segments or []):
        snaps = [list(x) for x in s.get("snaps") or []
                 if isinstance(x, (list, tuple)) and len(x) >= 4 and isinstance(x[1], str)]
        s = dict(s)
        s["snaps"] = snaps
        s["evidence"] = {k: v for k, v in (s.get("evidence") or {}).items() if k != "quad_track"} \
            if len(json.dumps(s.get("evidence") or {}, default=str)) > 200000 else s.get("evidence") or {}
        out.append(s)
    return out


def save_state(video, events, masks, meta=None, erased=None, segments=None, locate_done=False):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = clone_state(events, masks)
    data.update({
        "video": os.path.abspath(str(video)),
        "meta": {k: v for k, v in dict(meta or {}).items() if isinstance(v, (int, float, str, bool))},
        "erased": dict(erased) if erased else None,
        "segments": _segments_payload(segments),
        # 定位是否完整完成: 中途取消的部分结果不得被续跑当成"定位完成"直接读字
        "locate_done": bool(locate_done),
        "saved": time.time(),
    })
    path = state_path(video)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_state(video):
    path = state_path(video)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        events, masks = restore_state(data)
    except Exception:
        return None
    erased = data.get("erased") or None
    if erased and not os.path.isfile(str(erased.get("path", ""))):
        erased = None
    segments = [s for s in (data.get("segments") or []) if isinstance(s, dict)
                and any(isinstance(x, list) and len(x) >= 2 and isinstance(x[1], str) and os.path.isfile(x[1])
                        for x in (s.get("snaps") or []))]
    return {"events": events, "masks": masks, "meta": dict(data.get("meta") or {}),
            "erased": erased, "segments": segments,
            "locate_done": bool(data.get("locate_done", False))}


def drop_state(video):
    try:
        state_path(video).unlink(missing_ok=True)
    except OSError:
        pass
    purge_erased_for(video)
    prev = WORK / "task-preview" / video_key(video)     # 任务预览帧(擦除进行中的进度画面)一并清掉
    if prev.is_dir():
        shutil.rmtree(prev, ignore_errors=True)


def _all_states():
    if not STATE_DIR.is_dir():
        return
    for p in STATE_DIR.glob("*.json"):
        try:
            yield json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue


def referenced_locate_folders():
    """被某个状态文件的定位结果引用的快照目录(启动清理时要留下)。"""
    folders = set()
    for data in _all_states():
        for s in data.get("segments") or []:
            for x in s.get("snaps") or []:
                if isinstance(x, list) and len(x) >= 2 and isinstance(x[1], str):
                    folders.add(str(Path(x[1]).resolve().parent))
    return folders


def cleanup_erased(max_age_days=3.0):
    if not ERASED_DIR.is_dir():
        return 0
    referenced = set()
    for data in _all_states():
        er = data.get("erased") or {}
        if er.get("path"):
            referenced.add(Path(er["path"]).resolve())
    n, now = 0, time.time()
    for p in ERASED_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            old = now - p.stat().st_mtime > max_age_days * 86400
            if p.resolve() not in referenced or old:
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def erased_usage():
    """擦除缓存占用字节数"""
    if not ERASED_DIR.is_dir():
        return 0
    return sum(p.stat().st_size for p in ERASED_DIR.iterdir() if p.is_file())


def clear_erased():
    """清空全部擦除缓存, 并把各状态里的 erased 记录置空。返回删除文件数。"""
    n = 0
    if ERASED_DIR.is_dir():
        for p in ERASED_DIR.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                    n += 1
                else:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    preview_root = WORK / "task-preview"            # 任务预览帧也属擦除相关缓存
    if preview_root.is_dir():
        for p in preview_root.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            except OSError:
                pass
    if STATE_DIR.is_dir():
        for p in STATE_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("erased"):
                    data["erased"] = None
                    p.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            except (OSError, ValueError):
                continue
    return n
