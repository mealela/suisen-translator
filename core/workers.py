import os
import threading
import time
from copy import deepcopy

import cv2
from PySide6.QtCore import QThread, Signal

from core import ai
from core.ass_generator import generate_ass, generate_srt
from core.layout import ensure_font, pick_font
from core.models import prepare_cover_masks, stabilize_tracks
from core.pipeline import (run_auto_pipeline, segs_brief, segs_from_brief,
                           stage_locate, stage_read)
from core.v6_common import grab_frame, video_meta


def resolve_font(cfg):
    """成品/字幕文件实际使用的字体: 开着自动回退时按目标语言挑能显示的字体(防豆腐块)。"""
    if getattr(cfg, "font_fallback", True):
        return pick_font(cfg.subtitle_font, cfg.target_lang)
    return cfg.subtitle_font


def output_path(folder, filename, cfg):
    """输出同名文件: 默认自动改名 'xxx (2).mp4'; 设置为覆盖时直接返回原名。"""
    path = os.path.join(folder, filename)
    if str(getattr(cfg, "out_overwrite", "rename")) == "overwrite" or not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(filename)
    for n in range(2, 1000):
        cand = os.path.join(folder, f"{stem} ({n}){ext}")
        if not os.path.exists(cand):
            return cand
    return path


def ass_layout_for(cfg, font):
    """ASS 排布回调: 与画面合成同一套折行/有效字号/拉伸/竖排多列(见 v6_paint.ass_layout)。"""
    from core.v6_paint import ass_layout
    return lambda ev: ass_layout(ev, font, cfg.placement, cfg.out_bilingual, cfg.subtitle_align)


def subtitle_text(events, masks, w, h, cfg):
    """字幕文件内容的唯一出口(主窗口[导出字幕文件] / 05 导出附带 / 队列导出), 格式按 cfg.subtitle_format。"""
    if cfg.subtitle_format == "srt":
        if not any(e.display_text() for e in events):
            return ""
        return generate_srt(events, cfg.out_bilingual)
    if not events and not masks:
        return ""
    font = resolve_font(cfg)
    return generate_ass(events, masks, w, h, font, include_masks=True, bilingual=cfg.out_bilingual,
                        placement=cfg.placement, pipe_erase=cfg.pipe_erase,
                        mask_style=cfg.mask_style, mask_alpha=float(cfg.mask_bar_alpha) / 100.0,
                        layout=ass_layout_for(cfg, font), align=cfg.subtitle_align)


from core.v6_common import fmt_mmss      # 统一时间格式化(秒→MM:SS)


# ================= 翻译 =================

def run_translate_pipeline(events, cfg, progress, abort, updated=None, force=False, video=None):
    """批量翻译; 主轮结束后对仍无译文的条目自动逐条补译一次。"""
    from core.model_manager import prepare_endpoint
    from core.ocr_protocols import validate_ocr
    from core.prompts import get_prompt
    from core.v6_common import cache_get, cache_key, cache_put, endpoint_identity, publish
    endpoint = cfg.ep("translate")
    prepare_endpoint(endpoint)      # 只校验+拉起服务; 模型未安装就报错去[模型库], 绝不静默下载
    client = ai.TextClient(endpoint, abort=abort, timeout=getattr(cfg, "request_timeout", 300))
    system = get_prompt(cfg, "translate_system")
    batch_prompt = get_prompt(cfg, "translate_batch")
    ordered = sorted(events, key=lambda e: (e.start, e.y, e.x))
    selected = [e for e in ordered if e.text.strip() and (force or not e.translated.strip())]
    batch_size = 1 if endpoint.get("provider") == "local" else max(1, int(cfg.batch_size))

    cap = cv2.VideoCapture(video) if video and os.path.isfile(video) else None
    if cap is not None and not cap.isOpened():
        cap.release()
        cap = None

    def preview(ev):
        if cap is None:
            return
        t = ev.sample_time if ev.start <= ev.sample_time < ev.end else ev.start + 0.05
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if ok:
            publish(video, "translate", t, frame, force=True)

    context = []
    state = {"updated": 0}

    def accept(ev, text, key=None):
        ev.translated = text
        if key:
            cache_put(key, text)
        if ev.review_reason.startswith("翻译"):
            ev.review_reason = ""
        state["updated"] += 1

    def translate_pass(items, size, pbase, pspan, label):
        nonlocal context
        for start in range(0, len(items), size):
            if abort():
                return
            batch = items[start:start + size]
            missing, keys = [], {}
            for ev in batch:
                try:
                    validate_ocr(ev.text, max_chars=500)
                except ValueError:
                    ev.review_reason = "原文异常, 未提交翻译; 请重新识别"
                    continue
                key = cache_key("translate-v6", endpoint_identity(endpoint), cfg.target_lang,
                                system, batch_prompt, context[-2:] if cfg.ctx_translate else [], ev.text)
                keys[ev.id] = key
                hit = None if force else cache_get(key)
                if isinstance(hit, str) and hit:
                    accept(ev, hit)
                else:
                    missing.append(ev)
            if missing:
                try:
                    outputs = ai.translate_batch(client, [e.text for e in missing], cfg.target_lang,
                                                 system, batch_prompt,
                                                 context[-2:] if cfg.ctx_translate else None)
                except ai.AIError as exc:
                    if "已取消" in str(exc):
                        return
                    outputs = []
                    for ev in missing:
                        if abort():
                            break
                        try:
                            outputs.append(ai.translate_one(client, ev.text, cfg.target_lang, system,
                                                            context[-2:] if cfg.ctx_translate else None))
                        except Exception:
                            outputs.append("")
                except Exception:
                    outputs = [""] * len(missing)
                for ev, text in zip(missing, outputs):
                    text = str(text or "").strip()
                    if not text or len(text) > max(500, len(ev.text) * 10):
                        ev.review_reason = "翻译为空或异常, 请重试"
                        continue
                    accept(ev, text, keys[ev.id])
            for ev in batch:
                if ev.translated:
                    context.append((ev.text, ev.translated))
            context = context[-2:]
            if updated:
                updated()
            shown = next((e for e in reversed(batch) if e.translated), None)
            if shown is not None:
                preview(shown)
            done_n = min(len(items), start + len(batch))
            progress(pbase + int(done_n * pspan / max(1, len(items))), f"{label} {done_n}/{len(items)}")

    try:
        translate_pass(selected, batch_size, 0, 90, "翻译")
        retry = [e for e in selected if not e.translated.strip()
                 and not e.review_reason.startswith("原文异常")]
        if retry and not abort():
            progress(90, f"发现 {len(retry)} 条漏译, 自动补译一次...")
            translate_pass(retry, 1, 90, 10, "补译")
    finally:
        if cap is not None:
            cap.release()
    if updated:
        updated()
    fixed = stabilize_tracks(events, lock_all=bool(getattr(cfg, "subtitle_lock_position", True)))
    failures = sum(1 for e in selected if not e.translated.strip())
    progress(100, f"翻译结束: 更新 {state['updated']}, 待复核 {failures}"
                  + (f", 位置固定 {fixed}" if fixed else ""))
    return state["updated"]


# ================= QThread 封装 =================

class _BaseWorker(QThread):
    progress = Signal(int, str)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._abort = False
        self._pause = False
        self.finished.connect(self.deleteLater)

    def stop(self):
        self._abort = True

    def set_paused(self, v):
        self._pause = bool(v)

    def paused(self):
        return self._pause

    def aborted(self):
        while self._pause and not self._abort:
            time.sleep(0.12)
        return self._abort


class LocateWorker(_BaseWorker):
    snapshot = Signal(object)
    done = Signal(object)

    def __init__(self, video, cfg, t0=0.0, t1=None, parent=None):
        super().__init__(parent)
        self.video, self.cfg, self.t0, self.t1 = video, deepcopy(cfg), t0, t1

    def run(self):
        try:
            segs, meta = stage_locate(self.video, self.cfg, self.progress.emit, self.aborted,
                                      self.t0, self.t1, partial=self.snapshot.emit)
            if self._abort:
                self.failed.emit("已取消")
            else:
                self.done.emit({"segs": segs, "meta": meta})
        except Exception as e:
            self.failed.emit(f"定位失败: {e}")


class ReadWorker(_BaseWorker):
    snapshot = Signal(object)
    done = Signal(object)

    def __init__(self, video, segs, cfg, parent=None):
        super().__init__(parent)
        self.video, self.segs, self.cfg = video, deepcopy(segs), deepcopy(cfg)

    def run(self):
        try:
            events, _snapmap, vu = stage_read(self.video, self.segs, self.cfg, self.progress.emit,
                                              self.aborted, partial=self.snapshot.emit)
            if self._abort:
                self.failed.emit("已取消")
            else:
                self.done.emit({"events": events, "segs": self.segs, "vlm_used": vu})
        except Exception as e:
            self.failed.emit(f"读字失败: {e}")


class TranslateWorker(_BaseWorker):
    updated = Signal(object)
    done = Signal(int)

    def __init__(self, events, cfg, force=False, video=None, parent=None):
        super().__init__(parent)
        self.events, self.cfg, self.force = deepcopy(events), deepcopy(cfg), force
        self.video = video

    def run(self):
        try:
            self.progress.emit(0, "加载翻译引擎...")
            n = run_translate_pipeline(self.events, self.cfg, self.progress.emit, self.aborted,
                                       updated=lambda: self.updated.emit(deepcopy(self.events)),
                                       force=self.force, video=self.video)
            if self._abort:
                self.failed.emit("已取消")
            else:
                self.done.emit(n)
        except Exception as e:
            self.failed.emit(f"翻译失败: {e}")


class RenderWorker(_BaseWorker):
    """[05 导出]: 整段成品, 或选中字幕覆盖的片段(t0/t1)。frames_source=已擦除画面(有则复用); erase=是否现场擦除。"""
    done = Signal(str)

    def __init__(self, video, cfg, events, masks, out_path, t0=0.0, t1=None,
                 frames_source=None, erase=None, parent=None):
        super().__init__(parent)
        self.video, self.cfg = video, deepcopy(cfg)
        self.events, self.masks = deepcopy(events), deepcopy(masks)
        self.out_path, self.t0, self.t1 = out_path, t0, t1
        self.frames_source, self.erase = frames_source, erase

    def run(self):
        try:
            from core.v6_render import render_video
            ensure_font(self.cfg.subtitle_font, self.progress.emit, self.aborted)
            render_video(self.video, self.events, self.masks, self.out_path,
                         self.cfg, self.progress.emit, self.aborted,
                         t0=self.t0 or 0.0, t1=self.t1,
                         frames_source=self.frames_source, erase=self.erase)
            if self._abort:
                raise RuntimeError("已取消")
            self.done.emit(self.out_path)
        except Exception as exc:
            self.failed.emit(f"导出失败: {exc}")


class EraseWorker(_BaseWorker):
    """[04 原字处理] 工作区擦除。previous=上次记录(增量); redo=[(t0,t1),...] 强制重擦的时间段(选中字幕/整段重擦)。"""
    done = Signal(object)       # {"video", "path", "sig", "items", "params", "file"}

    def __init__(self, video, cfg, events, previous=None, redo=None, parent=None):
        super().__init__(parent)
        self.video, self.cfg, self.events = video, deepcopy(cfg), deepcopy(events)
        self.previous = dict(previous) if previous else None
        self.redo = [(float(a), float(b)) for a, b in redo] if redo else None

    def run(self):
        try:
            from core.v6_render import erase_to_workspace
            _path, _sig, info = erase_to_workspace(self.video, self.events, self.cfg,
                                                   self.progress.emit, self.aborted,
                                                   previous=self.previous, redo=self.redo)
            if self._abort:
                self.failed.emit("已取消")
            else:
                info = dict(info)
                info["video"] = self.video
                self.done.emit(info)
        except Exception as exc:
            self.failed.emit(f"擦除失败: {exc}")


# ---- 单视频阶段加权进度 ----
STAGE_WEIGHTS = {"locate": 20, "read": 25, "translate": 20, "erase": 25, "export": 10}


class _StageProgress:
    """进度上报。进度条显示当前环节的线性进度(每环节 0→100: 定位50%=定位完一半,
    读字/翻译/擦除/导出同理); 加权总值(只增不减)另记, 仅供队列总进度/剩余时间估算。"""

    def __init__(self, stages, emit):
        total = sum(STAGE_WEIGHTS.get(s, 10) for s in stages) or 1
        self.spans, base = {}, 0.0
        for s in stages:
            span = STAGE_WEIGHTS.get(s, 10) * 100.0 / total
            self.spans[s] = (base, span)
            base += span
        self._emit = emit
        self.value = 0.0          # 加权总进度(只增不减, 供队列 ETA)
        self.stage_value = 0.0    # 当前环节进度(给进度条, 每环节重新计)

    def _push(self, stage_p, overall_p, msg):
        self.stage_value = max(0.0, min(100.0, float(stage_p)))
        self.value = max(self.value, min(100.0, float(overall_p)))
        self._emit(int(round(self.stage_value)), msg)

    def stage(self, *names):
        """该环节(或连续几个环节合并)的 progress(p, msg) 回调: p=环节内 0~100。"""
        known = [n for n in names if n in self.spans]
        if not known:
            return lambda p, msg: self._push(p, self.value, msg)
        base = self.spans[known[0]][0]
        span = sum(self.spans[n][1] for n in known)
        return lambda p, msg: self._push(
            p, base + max(0.0, min(100.0, float(p))) * span / 100.0, msg)

    def skip(self, name, msg):
        base, span = self.spans.get(name, (self.value, 0.0))
        self._push(100.0, base + span, msg)

    def note(self, msg):
        self._push(self.stage_value, self.value, msg)


class QueueWorker(_BaseWorker):
    """批量队列, 支持断点接续。
    · 轮到某个视频时才向 GUI 线程索取它的最新状态(requestState → supply_state): 队列运行期间对其他视频
      的手动编辑不会被旧快照覆盖。直接调用 run()(单元测试)或 GUI 未接信号时, 退回同步调用 state_provider。
    · 有识别结果 → 跳过定位/识别; 有定位结果 → 直接识别; 译文齐全 → 跳过翻译; 擦除缓存签名一致 → 复用。
    · 01/02/03/04 只在工作区内处理; 只有 05 导出会写文件。
    · 04 勾选=按处理方式处理原字幕: erase=AI擦除进缓存(或开了[直接导出]时在导出中一次完成) /
      cover=补齐遮罩几何 / below=不处理(如实提示)。
    · 单视频进度按阶段加权只增不减; 队列总进度与剩余时间按视频时长估算。"""
    videoStarted = Signal(int, str)
    requestState = Signal(int, str)
    itemStage = Signal(int, str)
    itemProgress = Signal(int, int, str)
    itemPartial = Signal(int, float, object)
    itemLocate = Signal(int, object)
    itemErased = Signal(int, object)
    itemDone = Signal(int, bool, str, object)
    overall = Signal(int, str)
    finishedAll = Signal(int, int)

    def __init__(self, paths, cfg, state_provider=None, skip_provider=None, parent=None):
        super().__init__(parent)
        self.paths, self.cfg = list(paths), deepcopy(cfg)
        self.state_provider = state_provider
        self.skip_provider = skip_provider
        self.async_state = False            # GUI 接上 requestState 后置真
        self._cond = threading.Condition()
        self._supplied = {}
        self._t_start = 0.0
        self._last_overall = 0.0
        self._ctx = None
        self._sp = None               # 当前视频的 _StageProgress(取加权总值供 ETA)

    # ---- 状态获取 ----
    @staticmethod
    def _copy_state(st):
        st = st or {}
        return {"events": deepcopy(list(st.get("events") or [])),
                "masks": deepcopy(list(st.get("masks") or [])),
                "meta": dict(st.get("meta") or {}),
                "segments": deepcopy(list(st.get("segments") or [])),
                "erased": dict(st["erased"]) if st.get("erased") else None,
                "locate_done": bool(st.get("locate_done", False))}

    def supply_state(self, path, state):
        """GUI 线程回传该视频的最新状态; 深拷贝在这里(GUI 线程)完成, 工作线程只读副本。"""
        with self._cond:
            self._supplied[path] = self._copy_state(state)
            self._cond.notify_all()

    def _fetch_state(self, idx, path):
        if self.async_state:
            with self._cond:
                self._supplied.pop(path, None)
            self.requestState.emit(idx, path)
            deadline = time.monotonic() + 15.0
            with self._cond:
                while path not in self._supplied and not self._abort and time.monotonic() < deadline:
                    self._cond.wait(0.2)
                got = self._supplied.pop(path, None)
            if got is not None:
                return got
        return self._copy_state(self.state_provider(path) if self.state_provider else None)

    @staticmethod
    def _duration(path):
        try:
            return max(1.0, float(video_meta(path)["duration"] or 0.0))
        except Exception:
            return 60.0

    def _plan(self, cfg, path, events, segments, erased):
        """本视频实际要跑的阶段(用于进度加权)。"""
        from core.v6_render import erase_signature, erase_targets_exist
        stages = []
        if not events:
            if cfg.pipe_read:
                has_segs = any(any(isinstance(x[1], str) and os.path.isfile(x[1]) for x in (s.snaps or []))
                               for s in segs_from_brief(segments))
                stages += ["read"] if has_segs else ["locate", "read"]
            else:
                if cfg.pipe_locate and not segments:
                    stages.append("locate")
                return stages
        if cfg.pipe_translate and (not events or any(e.text.strip() and not e.translated.strip()
                                                     for e in events)):
            stages.append("translate")
        direct = bool(getattr(cfg, "export_direct", False)) and cfg.pipe_render
        if cfg.pipe_erase and cfg.placement == "erase" and not direct:
            valid = bool(erased) and os.path.isfile(str(erased.get("path", ""))) and (
                not events or erased.get("sig") == erase_signature(events, cfg, path))
            if not valid and (not events or erase_targets_exist(events, cfg)):
                stages.append("erase")
        if cfg.pipe_render:
            stages.append("export")
        return stages

    def _item_progress(self, idx, p, msg):
        self.itemProgress.emit(idx, p, msg)
        now = time.monotonic()
        if self._ctx is None or now - self._last_overall < 0.3:
            return
        self._last_overall = now
        i, name, done_dur, cur_dur, total_dur = self._ctx
        # p 是当前环节的线性进度(给进度条); 队列总进度/剩余时间用加权总值
        overall = self._sp.value if self._sp is not None else p
        processed = done_dur + cur_dur * max(0, min(100, overall)) / 100.0
        pct = int(processed * 100 / total_dur)
        eta = ""
        elapsed = now - self._t_start
        if elapsed > 5 and processed > 0:
            eta = f" · 预计剩余 {fmt_mmss(elapsed * (total_dur - processed) / processed)}"
        self.overall.emit(min(99, pct), f"队列 {i + 1}/{len(self.paths)}: {name} · 总进度 {pct}%{eta}")

    def run(self):
        from core.inpaint import check_engine
        from core.v6_render import (erase_signature, erase_targets_exist,
                                    erase_to_workspace, render_video)
        ok_n = fail_n = 0
        cfg = self.cfg
        self._t_start = time.monotonic()
        durations = [self._duration(p) for p in self.paths]
        total_dur = sum(durations) or 1.0
        done_dur = 0.0
        for idx, path in enumerate(self.paths):
            cur_dur = durations[idx]
            name = os.path.basename(path)
            self._ctx = (idx, name, done_dur, cur_dur, total_dur)
            try:
                if self._abort:
                    self.itemDone.emit(idx, False, "已停止", None)
                    fail_n += 1
                    continue
                if self.skip_provider and self.skip_provider(path):
                    self.itemDone.emit(idx, False, "已移出队列, 跳过", None)
                    fail_n += 1
                    continue
                self.videoStarted.emit(idx, path)
                stored = self._fetch_state(idx, path)
                events = list(stored["events"])
                meta = dict(stored["meta"])
                segments = list(stored["segments"])
                erased = stored["erased"]
                user_masks = list(stored["masks"])
                locate_done = bool(stored.get("locate_done", False))
                report = None
                stages_done = []
                sp = _StageProgress(self._plan(cfg, path, events, segments, erased),
                                    lambda p, m, i=idx: self._item_progress(i, p, m))
                self._sp = sp
                part = lambda t, evs, i=idx: self.itemPartial.emit(i, t, deepcopy(evs))
                loc_part = lambda segs, i=idx: self.itemLocate.emit(i, list(segs))
                direct = bool(getattr(cfg, "export_direct", False)) and cfg.pipe_render

                def result():
                    return {"events": events, "masks": list(user_masks), "meta": meta,
                            "segments": segments, "report": report, "erased": erased,
                            "locate_done": locate_done}

                try:
                    # ---- ①② 定位/识别: 已有结果直接复用(断点接续) ----
                    if events:
                        self.itemStage.emit(idx, "复用识别结果")
                        if not meta.get("w"):
                            meta = video_meta(path)
                        report = {"reused": len(events)}
                        sp.note(f"复用已有 {len(events)} 条字幕, 跳过定位/识别")
                        part(-1.0, list(events))
                    elif cfg.pipe_read:
                        self.itemStage.emit(idx, "识别")
                        reuse = [s for s in segs_from_brief(segments)
                                 if locate_done
                                 and any(isinstance(x[1], str) and os.path.isfile(x[1])
                                        for x in (s.snaps or []))]
                        if reuse:
                            prog = sp.stage("read")
                            prog(0, f"复用已有定位结果({len(reuse)} 段), 直接识别")
                            if not meta.get("w"):
                                meta = video_meta(path)
                            events, _snapmap, calls = stage_read(
                                path, reuse, cfg, prog, self.aborted,
                                partial=lambda evs: part(-1.0, evs))
                            report = {"vlm_read": calls, "reused_segments": len(reuse)}
                            segments = segs_brief(reuse)
                        else:
                            locate_done = False          # 重新定位: 完成前清除"已完成"标记(防中途取消后误续)
                            events, meta, segs, report = run_auto_pipeline(
                                path, cfg, sp.stage("locate"), self.aborted,
                                partial=part, locate_partial=loc_part,
                                progress_read=sp.stage("read"))
                            segments = segs_brief(segs)
                            locate_done = True           # 定位+识别链条完成, 候选段此后可直接复用
                        if self._abort:
                            raise RuntimeError("已取消")
                        if not events:
                            self.itemDone.emit(idx, False, "未检测到字幕", result())
                            fail_n += 1
                            continue
                        stages_done.append("识别")
                    elif cfg.pipe_locate:
                        if segments:
                            sp.note(f"已有定位结果 {len(segments)} 段, 跳过")
                        else:
                            self.itemStage.emit(idx, "定位")
                            segs, meta = stage_locate(path, cfg, sp.stage("locate"), self.aborted,
                                                      partial=loc_part)
                            segments = segs_brief(segs)
                            if self._abort:
                                raise RuntimeError("已取消")
                        self.itemDone.emit(idx, True, f"完成: 定位{len(segments)}个候选段", result())
                        ok_n += 1
                        continue
                    else:
                        self.itemDone.emit(idx, False, "尚未识别, 且未勾选[01 定位]/[02 识别], 已跳过", None)
                        fail_n += 1
                        continue

                    # ---- ③ 翻译 ----
                    if cfg.pipe_translate:
                        untranslated = [e for e in events if e.text.strip() and not e.translated.strip()]
                        if untranslated:
                            self.itemStage.emit(idx, "翻译")
                            run_translate_pipeline(events, cfg, sp.stage("translate"), self.aborted,
                                                   updated=lambda: part(-1.0, list(events)), video=path)
                            stages_done.append("翻译")
                        else:
                            sp.skip("translate", "译文已齐全, 跳过翻译")
                    if self._abort:
                        raise RuntimeError("已取消")

                    # ---- ④ 原字处理: 工作区内, 不产出成品 ----
                    if cfg.pipe_erase:
                        if cfg.placement == "below":
                            sp.skip("erase", "译文放在原字幕下方: 原字幕保留, 04 不做处理")
                        elif cfg.placement == "erase":
                            sig = erase_signature(events, cfg, path)
                            valid = bool(erased) and erased.get("sig") == sig \
                                and os.path.isfile(str(erased.get("path", "")))
                            if valid:
                                sp.skip("erase", "画面已擦除且与当前字幕一致, 复用")
                            elif not erase_targets_exist(events, cfg):
                                erased = None
                                sp.skip("erase", "没有需要擦除的文字, 跳过擦除")
                            elif direct:
                                sp.skip("erase", "导出时一次完成擦除+嵌字(单次编码), 不另存擦除缓存")
                            else:
                                okE, msgE = check_engine(cfg)
                                if not okE and cfg.erase_engine != "lama":   # LAMA 缺失会在渲染前自动安装
                                    raise RuntimeError("无痕擦除不可用: " + msgE)
                                self.itemStage.emit(idx, "擦除")
                                _out, _sig, info = erase_to_workspace(
                                    path, events, cfg, sp.stage("erase"), self.aborted, previous=erased)
                                erased = dict(info)
                                self.itemErased.emit(idx, dict(erased))
                                stages_done.append("擦除")
                        else:
                            n = prepare_cover_masks(events, cfg.placement, cfg.pipe_erase)
                            sp.skip("erase", f"遮罩覆盖: 导出时 {n} 条字幕的原字区域按[{cfg.mask_style}]样式盖住")
                            part(-1.0, list(events))
                            stages_done.append("遮罩")
                    if self._abort:
                        raise RuntimeError("已取消")

                    # ---- ⑤ 导出: 只有这里才写文件 ----
                    outputs = []
                    if cfg.pipe_render:
                        self.itemStage.emit(idx, "导出")
                        prog = sp.stage("export")
                        if not meta.get("w"):
                            meta = video_meta(path)
                        out_dir = cfg.out_dir.strip() or os.path.dirname(path)
                        os.makedirs(out_dir, exist_ok=True)
                        base = os.path.splitext(name)[0] + cfg.out_suffix
                        ensure_font(cfg.subtitle_font, prog, self.aborted)
                        if cfg.pipe_subs:
                            text = subtitle_text(events, user_masks, meta["w"], meta["h"], cfg)
                            if text:
                                sub_path = output_path(out_dir, base + "." + cfg.subtitle_format, cfg)
                                with open(sub_path, "w", encoding="utf-8-sig") as f:
                                    f.write(text)
                                outputs.append(os.path.basename(sub_path))
                        out_v = output_path(out_dir, base + ".mp4", cfg)
                        frames_source, erase = None, False
                        if cfg.pipe_erase and cfg.placement == "erase":
                            if erased and os.path.isfile(str(erased.get("path", ""))) \
                                    and erased.get("sig") == erase_signature(events, cfg, path):
                                frames_source = erased["path"]
                            elif erase_targets_exist(events, cfg):
                                okE, msgE = check_engine(cfg)
                                if not okE and cfg.erase_engine != "lama":
                                    raise RuntimeError("无痕擦除不可用: " + msgE)
                                erase = True
                        render_video(path, events, user_masks, out_v, cfg, prog, self.aborted,
                                     frames_source=frames_source, erase=erase)
                        outputs.append(os.path.basename(out_v))
                    if outputs:
                        summary = "完成: " + " + ".join(outputs)
                    elif stages_done:
                        summary = "完成: 已" + "/".join(stages_done) + "(工作区, 未导出)"
                    else:
                        summary = "完成: 无需处理(所选步骤均已完成)"
                    self.itemDone.emit(idx, True, summary, result())
                    ok_n += 1
                except Exception as exc:
                    self.itemDone.emit(idx, False, f"失败: {exc}", result())
                    fail_n += 1
            finally:
                done_dur += cur_dur
        self.finishedAll.emit(ok_n, fail_n)


class RecognizeWorker(_BaseWorker):
    """字幕区[识别选中]: 对选中字幕的框选区域截帧读字; 路由/回退与自动流程同一实现(pipeline.read_region)。"""
    done = Signal(dict)

    def __init__(self, video, jobs, cfg, parent=None):
        super().__init__(parent)
        self.video, self.jobs, self.cfg = video, list(jobs), deepcopy(cfg)

    def run(self):
        import numpy as np
        from core.pipeline import LocalLineReader, read_region
        from core.v6_common import crop_quad
        out, diagnostics = {}, []
        try:
            cfg = self.cfg
            route = str(cfg.ocr_engine or "vlm")
            client = None
            if route in ("auto", "vlm"):
                try:
                    client = ai.VLMClient(cfg.ep("vlm"), cfg.vlm_detail_width, abort=self.aborted,
                                          timeout=getattr(cfg, "request_timeout", 300))
                except Exception as exc:
                    diagnostics.append(str(exc))
            local = [None]

            def get_local():
                if local[0] is None:
                    local[0] = LocalLineReader(cfg)
                return local[0]

            total = max(1, len(self.jobs))
            weak_vertical = 0
            for i, (eid, t, quad, vertical) in enumerate(self.jobs):
                if self.aborted():
                    break
                frame = grab_frame(self.video, t)
                if frame is None:
                    continue
                try:
                    crop = crop_quad(frame, np.float32(quad), border=3)
                except Exception:
                    continue
                text, conf, _source = read_region(crop, bool(vertical), cfg, client, get_local,
                                                  diagnostics, None, str(eid))
                if vertical and client is None:
                    weak_vertical += 1
                if text:
                    out[eid] = (text, float(conf))
                self.progress.emit(int((i + 1) * 100 / total), f"识别 {i + 1}/{total}")
            if self._abort:
                self.failed.emit("已取消")
                return
            tail = ""
            if route in ("auto", "vlm") and client is None and diagnostics:
                tail = f"AI 读字不可用({str(diagnostics[0])[:60]}), 已改用本地识别"
            elif weak_vertical:
                tail = f"{weak_vertical} 个竖排框只用了本地切字识别, 准确率有限; 配置 AI 视觉模型可改善"
            if tail:
                self.progress.emit(100, tail)
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(f"识别失败: {e}")


class HardwareProbeWorker(QThread):
    """后台硬件探测(nvidia-smi / PowerShell 可能各耗时数秒), 不阻塞界面。"""
    probed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.finished.connect(self.deleteLater)

    def run(self):
        from core import hardware
        try:
            self.probed.emit(hardware.probe())
        except Exception:
            self.probed.emit(None)
