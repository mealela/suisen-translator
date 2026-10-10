import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QLabel

from config import (ALT_MT_TAG, DEFAULT_MT_TAG, LIGHT_MT_TAG, MANAGED_OLLAMA,
                    AppConfig, _migrate_old)
from core import hardware
from core.models import (MaskEvent, SubtitleEvent, event_covered,
                         prepare_cover_masks)

app = QApplication.instance() or QApplication([])


class ParallelLamaTests(unittest.TestCase):
    """并行 LAMA 进程池: 帧按连续段切给各进程、先全部提交再按序收集、n=1 时与单进程一致。"""

    @staticmethod
    def _fake_worker_cls(log):
        class FakeWorker:
            instances = []

            def __init__(self, cfg, fps):
                self.calls = []
                FakeWorker.instances.append(self)

            def submit(self, frames, masks):
                self.calls.append(len(frames))
                log.append(("submit", len(FakeWorker.instances), len(frames)))
                return (list(frames), list(masks), [], None, [i for i, m in enumerate(masks) if np.any(m)])

            def collect(self, handle, abort):
                log.append(("collect", 0, 0))
                frames, masks, boxes, task, dirty = handle
                out = list(frames)
                for i in dirty:
                    out[i] = np.clip(frames[i].astype(np.int16) + 1, 0, 255).astype(np.uint8)
                return out

            def erase_chunk(self, frames, masks, abort):
                return self.collect(self.submit(frames, masks), abort)

            def close(self):
                pass
        return FakeWorker

    def test_pool_distributes_contiguous_groups_and_keeps_order(self):
        from core import inpaint
        log = []
        fake = self._fake_worker_cls(log)
        with mock.patch.object(inpaint, "ExternalLamaEraser", fake):
            pool = inpaint.ParallelLamaEraser(AppConfig(), 25.0, workers=3)
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(7)]
        masks = [np.full((8, 8), 255, np.uint8) for _ in range(7)]
        out = pool.erase_chunk(frames, masks, lambda: False)
        self.assertEqual([w.calls for w in fake.instances], [[3], [3], [1]],
                         "7 帧应按 3/3/1 连续切给 3 个进程")
        submits = [x for x in log if x[0] == "submit"]
        collects = [x for x in log if x[0] == "collect"]
        self.assertEqual(len(submits), 3)
        self.assertEqual(len(collects), 3)
        self.assertLess(max(log.index(s) for s in submits), min(log.index(c) for c in collects),
                        "必须先全部提交(非阻塞)再开始收集")
        for i, f in enumerate(out):
            self.assertTrue(np.all(f == np.clip(frames[i].astype(np.int16) + 1, 0, 255)),
                            "收集结果必须与输入顺序一致")

    def test_pool_single_worker_matches_single_process(self):
        from core import inpaint
        fake = self._fake_worker_cls([])
        with mock.patch.object(inpaint, "ExternalLamaEraser", fake):
            pool = inpaint.ParallelLamaEraser(AppConfig(), 25.0, workers=1)
        self.assertEqual(len(pool._workers), 1)
        frames = [np.full((8, 8, 3), i, np.uint8) for i in range(4)]
        masks = [np.full((8, 8), 255, np.uint8) for _ in range(4)]
        out = pool.erase_chunk(frames, masks, lambda: False)
        self.assertEqual(len(out), 4)

    def test_get_eraser_parallel_lama(self):
        from core import inpaint
        cfg = AppConfig()
        cfg.erase_engine = "lama"
        cfg.erase_lama_workers = 3
        fake = self._fake_worker_cls([])
        with mock.patch("importlib.util.find_spec", return_value=None), \
                mock.patch.object(inpaint, "ExternalLamaEraser", fake):
            eraser, mode = inpaint.get_eraser(cfg, 25.0)
        self.assertEqual(mode, "batch")
        self.assertEqual(len(eraser._workers), 3)

    def test_get_eraser_default_single_lama(self):
        from core import inpaint
        cfg = AppConfig()
        cfg.erase_engine = "lama"
        self.assertEqual(cfg.erase_lama_workers, 1)   # 默认 1
        fake = self._fake_worker_cls([])
        with mock.patch("importlib.util.find_spec", return_value=None), \
                mock.patch.object(inpaint, "ExternalLamaEraser", fake):
            eraser, mode = inpaint.get_eraser(cfg, 25.0)
        self.assertEqual(mode, "batch")
        self.assertIsInstance(eraser, fake)


class MigrateTests(unittest.TestCase):
    def test_hy_mt_replaced(self):
        data = {"cfg_version": 2, "ep_translate": {"ollama_model": "hy-mt2:latest"}}
        _migrate_old(data)
        self.assertEqual(data["ep_translate"]["ollama_model"], DEFAULT_MT_TAG)
        self.assertEqual(data["cfg_version"], 6)

    def test_pipe_erase_derived_from_placement(self):
        for placement, expected in (("erase", True), ("cover", False), ("below", False)):
            data = {"cfg_version": 2, "placement": placement}
            _migrate_old(data)
            self.assertEqual(data["pipe_erase"], expected, placement)
        data = {"cfg_version": 2, "placement": "cover", "pipe_erase": True}
        _migrate_old(data)
        self.assertTrue(data["pipe_erase"])

    def test_erase_mode_derived_and_autocolor_off(self):
        data = {"cfg_version": 3, "placement": "cover"}
        _migrate_old(data)
        self.assertEqual(data["erase_mode"], "cover")
        self.assertFalse(data["auto_text_color"])

    def test_dml_default_on_after_migration(self):
        # 发行版已内嵌 onnxruntime-directml + 运行时探测可用性: 默认开启;
        # 没有 DX12 显卡的机器探测失败自动走 CPU
        data = {"cfg_version": 4}
        _migrate_old(data)
        self.assertTrue(data["locator_rapid_dml"])
        data = {"cfg_version": 4, "locator_rapid_dml": False}
        _migrate_old(data)
        self.assertFalse(data["locator_rapid_dml"])     # 用户显式关闭的不动

    def test_placement_and_erase_mode_stay_in_sync(self):
        cfg = AppConfig()
        cfg.placement = "cover"
        cfg._normalize()
        self.assertEqual(cfg.erase_mode, "cover")
        cfg.placement, cfg.erase_mode = "below", "erase"
        cfg._normalize()
        self.assertEqual((cfg.placement, cfg.erase_mode), ("below", "erase"))

    def test_legacy_engine_names(self):
        for old, new in (("rapid", "auto"), ("easy", "auto"), ("locate", "auto"), ("paddle", "local")):
            data = {"ocr_engine": old}
            _migrate_old(data)
            self.assertEqual(data["ocr_engine"], new)
        data = {"pipe_ocr": False}
        _migrate_old(data)
        self.assertFalse(data["pipe_locate"])
        self.assertFalse(data["pipe_read"])


class HardwareTests(unittest.TestCase):
    def test_classify_boundaries(self):
        c = hardware.classify
        self.assertEqual(c({"vram_gb": 12, "ram_gb": 24, "cpu": 4}), "high")
        self.assertEqual(c({"vram_gb": 11.9, "ram_gb": 64, "cpu": 32}), "mid")
        self.assertEqual(c({"vram_gb": 0, "ram_gb": 16, "cpu": 8}), "mid")
        self.assertEqual(c({"vram_gb": 0, "ram_gb": 16, "cpu": 6}), "low")
        self.assertEqual(c({"vram_gb": 6, "ram_gb": 8, "cpu": 4}), "mid")

    def test_apply_preset_keeps_alt_model_only_in_high(self):
        cfg = AppConfig()
        cfg.ep_translate = {**cfg.ep_translate, "provider": "ollama",
                            "ollama_url": MANAGED_OLLAMA, "ollama_model": ALT_MT_TAG}
        hardware.apply_preset(cfg, "high")
        self.assertEqual(cfg.ep_translate["ollama_model"], ALT_MT_TAG)
        hardware.apply_preset(cfg, "mid")
        self.assertEqual(cfg.ep_translate["ollama_model"], LIGHT_MT_TAG)
        self.assertEqual(cfg.erase_engine, "lama")
        hardware.apply_preset(cfg, "low")
        self.assertEqual((cfg.erase_engine, cfg.ocr_engine), ("opencv", "local"))

    def test_apply_preset_leaves_api_endpoint_alone(self):
        cfg = AppConfig()
        cfg.ep_translate = {**cfg.ep_translate, "provider": "api", "api_model": "deepseek-flash"}
        hardware.apply_preset(cfg, "low")
        self.assertEqual(cfg.ep_translate["api_model"], "deepseek-flash")


class QueueWorkerTests(unittest.TestCase):
    @staticmethod
    def _quiet(cfg):
        cfg.pipe_translate = cfg.pipe_erase = cfg.pipe_render = False
        return cfg

    def _run(self, cfg, paths, state=None, **patches):
        from core.workers import QueueWorker
        done, stages, msgs = [], [], []
        w = QueueWorker(paths, cfg, state_provider=state)
        w.itemDone.connect(lambda i, ok, msg, res: done.append((i, ok, msg, res)))
        w.itemStage.connect(lambda i, s: stages.append((i, s)))
        w.itemProgress.connect(lambda i, p, m: msgs.append((i, p, m)))
        if patches:
            with mock.patch.multiple("core.workers", **patches):
                w.run()
        else:
            w.run()
        return w, done, stages, msgs

    def test_partial_locate_is_not_treated_as_done(self):
        """定位中途取消(状态里只有候选段、没有 locate_done 标记) → 续跑必须重新定位,
        不能拿着部分候选段直接读字; 有标记时才能直接复用读字。"""
        from core.pipeline import SegRecord
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(b"x")
        tmp.close()
        seg = SegRecord(id="s1", start=0.0, end=1.0, bbox=(0, 0, 10, 10),
                        quad_pts=[[0, 0], [10, 0], [10, 10], [0, 10]],
                        snaps=[(0.5, tmp.name, 0, 0)])
        ev = SubtitleEvent(0.0, 1.0, "字幕")
        cfg = self._quiet(AppConfig())

        def make_state(done):
            return lambda p: {"events": [], "masks": [], "meta": {"w": 640, "h": 360},
                              "segments": [seg.brief()], "erased": None, "locate_done": done}

        ap = mock.Mock(return_value=([ev], {"w": 640, "h": 360, "fps": 25, "duration": 60}, [seg], {}))
        sr = mock.Mock(return_value=([ev], {}, 0))
        _, done, _, _ = self._run(cfg, ["a.mp4"], state=make_state(False),
                                  run_auto_pipeline=ap, stage_read=sr)
        self.assertTrue(ap.called, "无 locate_done 标记时必须重新定位")
        self.assertFalse(sr.called, "无 locate_done 标记时不得直接复用读字")
        self.assertTrue(done[0][3].get("locate_done"), "定位完成后结果应带上 locate_done")

        ap2 = mock.Mock(return_value=([ev], {"w": 640, "h": 360, "fps": 25, "duration": 60}, [seg], {}))
        sr2 = mock.Mock(return_value=([ev], {}, 0))
        _, done2, _, _ = self._run(cfg, ["a.mp4"], state=make_state(True),
                                   run_auto_pipeline=ap2, stage_read=sr2)
        self.assertTrue(sr2.called, "有 locate_done 标记时才能直接复用读字")
        self.assertFalse(ap2.called)
        os.unlink(tmp.name)

    def test_skips_when_no_stage_selected(self):
        cfg = AppConfig()
        cfg.pipe_locate = cfg.pipe_read = False
        _, done, _, _ = self._run(cfg, ["a.mp4"])
        self.assertFalse(done[0][1])
        self.assertIn("未勾选", done[0][2])

    def test_locate_only(self):
        cfg = AppConfig()
        cfg.pipe_read = False
        fake = mock.Mock(return_value=([], {"w": 100, "h": 50}))
        _, done, stages, _ = self._run(cfg, ["a.mp4"], stage_locate=fake)
        self.assertEqual(stages, [(0, "定位")])
        self.assertTrue(done[0][1])
        self.assertIn("定位0个候选段", done[0][2])

    def test_no_events_marks_failure(self):
        cfg = self._quiet(AppConfig())
        fake = mock.Mock(return_value=([], {"w": 100, "h": 50}, [], None))
        _, done, _, _ = self._run(cfg, ["a.mp4"], run_auto_pipeline=fake)
        self.assertFalse(done[0][1])
        self.assertIn("未检测到字幕", done[0][2])

    def test_stop_marks_current_and_remaining(self):
        from core.workers import QueueWorker
        cfg = self._quiet(AppConfig())
        w = QueueWorker(["a.mp4", "b.mp4"], cfg)
        done = []
        w.itemDone.connect(lambda i, ok, msg, res: done.append((i, ok, msg)))

        def fake(path, *a, **k):
            w.stop()
            return [], {"w": 100, "h": 50}, [], None

        with mock.patch("core.workers.run_auto_pipeline", fake):
            w.run()
        self.assertEqual([d[1] for d in done], [False, False])
        self.assertIn("已取消", done[0][2])
        self.assertIn("已停止", done[1][2])

    def test_resume_skips_recognition_and_translation(self):
        cfg = self._quiet(AppConfig())
        cfg.pipe_translate = True
        ev = SubtitleEvent(0, 1, "原文", translated="译文")
        state = lambda p: {"events": [ev], "meta": {"w": 100, "h": 50}}
        auto, trans = mock.Mock(), mock.Mock()
        _, done, stages, _ = self._run(cfg, ["a.mp4"], state=state,
                                       run_auto_pipeline=auto, run_translate_pipeline=trans)
        auto.assert_not_called()
        trans.assert_not_called()
        self.assertEqual([s for _, s in stages], ["复用识别结果"])
        self.assertTrue(done[0][1])

    def test_resume_translates_when_missing(self):
        cfg = self._quiet(AppConfig())
        cfg.pipe_translate = True
        ev = SubtitleEvent(0, 1, "原文")
        state = lambda p: {"events": [ev], "meta": {"w": 100, "h": 50}}
        trans = mock.Mock()
        self._run(cfg, ["a.mp4"], state=state, run_translate_pipeline=trans)
        trans.assert_called_once()

    def test_below_placement_does_not_pretend_to_cover(self):
        """译文放下方 + 勾选 04: 不再报'将按样式遮盖', 而是如实说明原字幕保留。"""
        cfg = self._quiet(AppConfig())
        cfg.pipe_erase, cfg.placement = True, "below"
        ev = SubtitleEvent(0, 1, "原文", translated="译文")
        state = lambda p: {"events": [ev], "meta": {"w": 100, "h": 50}}
        _, done, _, msgs = self._run(cfg, ["a.mp4"], state=state)
        self.assertTrue(done[0][1])
        self.assertTrue(any("原字幕保留" in m for _, _, m in msgs))
        self.assertFalse(any("遮盖" in m for _, _, m in msgs))

    def test_worker_does_not_snapshot_state_at_start(self):
        """状态在轮到该视频时才取: 启动后、处理前对该视频的改动能被看到。"""
        cfg = self._quiet(AppConfig())
        cfg.pipe_translate = True
        store = {"events": [SubtitleEvent(0, 1, "原文", translated="译")], "meta": {"w": 100, "h": 50}}
        from core.workers import QueueWorker
        w = QueueWorker(["a.mp4"], cfg, state_provider=lambda p: store)
        store["events"].append(SubtitleEvent(1, 2, "新加的", translated=""))    # 启动后才加的字幕
        trans = mock.Mock()
        with mock.patch("core.workers.run_translate_pipeline", trans):
            w.run()
        trans.assert_called_once()                       # 新加的那条没译 → 翻译阶段被触发
        self.assertEqual(len(trans.call_args[0][0]), 2)

    def test_stage_progress_is_per_stage_and_overall_monotonic(self):
        from core.workers import _StageProgress
        # 进度条值 = 当前环节的线性进度(每环节 0→100, 换环节重新计)
        seen = []
        sp = _StageProgress(["translate", "export"], lambda p, m: seen.append(p))
        sp.stage("translate")(50, "a")
        self.assertEqual(seen[-1], 50)
        sp.stage("translate")(30, "b")              # 环节内回退如实反映
        self.assertEqual(seen[-1], 30)
        sp.stage("export")(50, "c")                 # 换环节重新从 0 计
        self.assertEqual(seen[-1], 50)
        sp.stage("export")(100, "d")
        self.assertEqual(seen[-1], 100)
        # 加权总值(供队列 ETA)必须只增不减且最终=100
        overalls = []
        sp2 = _StageProgress(["translate", "export"],
                             lambda p, m: overalls.append(sp2.value))
        sp2.stage("translate")(100, "a")
        sp2.stage("translate")(30, "b")
        sp2.stage("export")(50, "c")
        sp2.stage("export")(100, "d")
        self.assertEqual(overalls, sorted(overalls))
        self.assertAlmostEqual(overalls[-1], 100.0)


class OutputPathTests(unittest.TestCase):
    def test_rename_or_overwrite(self):
        from core.workers import output_path
        cfg = AppConfig()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.mp4").write_bytes(b"x")
            self.assertEqual(os.path.basename(output_path(tmp, "a.mp4", cfg)), "a (2).mp4")
            cfg.out_overwrite = "overwrite"
            self.assertEqual(os.path.basename(output_path(tmp, "a.mp4", cfg)), "a.mp4")
            self.assertEqual(os.path.basename(output_path(tmp, "b.mp4", cfg)), "b.mp4")


class GeometryTests(unittest.TestCase):
    def test_slanted_line_is_not_vertical(self):
        from core.geometry import quad_axis, quad_from_rect, quad_is_vertical
        q = quad_from_rect(400, 300, 600, 40, -35)
        self.assertFalse(quad_is_vertical(q))
        angle, long_side, short_side = quad_axis(q)
        self.assertAlmostEqual(abs(angle), 35.0, delta=0.5)
        self.assertAlmostEqual(long_side, 600.0, delta=0.5)
        self.assertAlmostEqual(short_side, 40.0, delta=0.5)

    def test_upright_column_is_vertical(self):
        from core.geometry import quad_is_vertical
        self.assertTrue(quad_is_vertical([[0, 0], [40, 0], [40, 400], [0, 400]]))
        self.assertFalse(quad_is_vertical([[0, 0], [400, 0], [400, 40], [0, 40]]))

    def test_polygon_nms_keeps_slanted_neighbours(self):
        from core.geometry import quad_from_rect
        from core.locator import _nms
        a = quad_from_rect(400, 200, 600, 40, -30)
        b = quad_from_rect(400, 250, 600, 40, -30)
        dets = [{"quad": a, "bbox": (0, 0, 1, 1), "conf": 0.9},
                {"quad": b, "bbox": (0, 0, 1, 1), "conf": 0.8}]
        self.assertEqual(len(_nms(dets)), 2)


class FragmentTests(unittest.TestCase):
    def test_partial_line_is_merged_into_full_line(self):
        from core.pipeline import _postfilter_events
        cfg = AppConfig()
        cfg.filter_llm = False
        full = SubtitleEvent(1.0, 3.0, "バカバカっかたすぎっ", x=300, y=100,
                             geom=[[0, 80], [600, 80], [600, 120], [0, 120]])
        part = SubtitleEvent(1.2, 2.8, "かたすき", x=450, y=100,
                             geom=[[300, 80], [600, 80], [600, 120], [300, 120]])
        events = [full, part]
        dropped = _postfilter_events(events, [], cfg, {"w": 1280, "h": 720}, lambda p, m: None, lambda: False)
        self.assertEqual([e.text for e in events], ["バカバカっかたすぎっ"])
        self.assertEqual(len(dropped), 1)


class CoverRuleTests(unittest.TestCase):
    def test_auto_mask_only_in_cover_mode(self):
        ev = SubtitleEvent(0, 1, "text")
        self.assertFalse(event_covered(ev, "erase", True))
        self.assertTrue(event_covered(ev, "cover", True))
        self.assertFalse(event_covered(ev, "cover", False))
        ev.mask_custom, ev.mask_on = True, True
        self.assertTrue(event_covered(ev, "erase", True))
        ev.mask_on = False
        self.assertFalse(event_covered(ev, "cover", True))

    def test_prepare_cover_masks_counts_only_covered(self):
        """04 遮罩阶段只补几何、只数会被盖住的, 不再写无效开关。"""
        a = SubtitleEvent(0, 1, "a", bbox=(0, 0, 10, 5))
        b = SubtitleEvent(0, 1, "")
        c = SubtitleEvent(0, 1, "c", bbox=(0, 0, 10, 5), mask_custom=True, mask_on=False)
        self.assertEqual(prepare_cover_masks([a, b, c], "cover", True), 1)
        self.assertEqual(len(a.mask), 4)
        self.assertFalse(a.mask_on)
        self.assertEqual(prepare_cover_masks([a, b, c], "below", True), 0)

    def test_user_mask_blocks_are_not_erased(self):
        from core.v6_render import frame_mask
        cfg = AppConfig()
        cfg.erase_time_pad, cfg.erase_dilate = 0, 0
        m = MaskEvent(0, 1, 10, 10, 50, 30)
        self.assertEqual(np.count_nonzero(frame_mask([], [m], 0.5, 100, 100, cfg)), 0)


class EraseSignatureTests(unittest.TestCase):
    def test_signature_tracks_erase_geometry_only(self):
        from core.v6_render import erase_signature
        cfg = AppConfig()
        ev = SubtitleEvent(0, 1, "a", geom=[[0, 0], [10, 0], [10, 5], [0, 5]])
        s1 = erase_signature([ev], cfg)
        ev.translated = "译"
        self.assertEqual(s1, erase_signature([ev], cfg))
        ev.end = 2.0
        self.assertNotEqual(s1, erase_signature([ev], cfg))

    def test_signature_includes_video_identity(self):
        from core.v6_render import erase_signature
        cfg = AppConfig()
        ev = SubtitleEvent(0, 1, "a", geom=[[0, 0], [10, 0], [10, 5], [0, 5]])
        with tempfile.TemporaryDirectory() as tmp:
            v = Path(tmp, "v.mp4")
            v.write_bytes(b"1")
            s1 = erase_signature([ev], cfg, str(v))
            v.write_bytes(b"12")                          # 同路径换了文件 → 旧缓存不能复用
            self.assertNotEqual(s1, erase_signature([ev], cfg, str(v)))

    def test_affected_ranges_only_cover_changed_time(self):
        from core.v6_render import affected_ranges, erase_items
        cfg = AppConfig()
        cfg.erase_time_pad = 0.0
        a = SubtitleEvent(0, 2, "a", geom=[[0, 0], [10, 0], [10, 5], [0, 5]])
        b = SubtitleEvent(5, 6, "b", geom=[[0, 10], [10, 10], [10, 15], [0, 15]])
        old = erase_items([a, b], cfg)
        a.end = 2.5                                       # 只把 a 拉长 0.5s
        ranges = affected_ranges(old, erase_items([a, b], cfg), cfg, fps=25.0)
        self.assertEqual(len(ranges), 1)
        lo, hi = ranges[0]
        self.assertLess(lo, 2.0)
        self.assertGreater(hi, 2.5)
        self.assertLess(hi, 3.0)                          # b 的时间段完全不碰
        self.assertEqual(affected_ranges(old, old, cfg), [])


class WorkspaceTests(unittest.TestCase):
    def test_state_roundtrip(self):
        from core import workspace
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(workspace, "STATE_DIR", Path(tmp)):
                ev = SubtitleEvent(1, 2, "原文", translated="译文", geom=[[0, 0], [10, 0], [10, 5], [0, 5]])
                mk = MaskEvent(0, 1, 1, 2, 3, 4)
                workspace.save_state("v.mp4", [ev], [mk], {"w": 100, "h": 50})
                st = workspace.load_state("v.mp4")
                self.assertEqual(st["events"][0].translated, "译文")
                self.assertEqual(st["masks"][0].x2, 3)
                self.assertEqual(st["meta"]["w"], 100)
                self.assertIsNone(st["erased"])
                self.assertEqual(st["segments"], [])

    def test_segments_persist_when_snapshots_exist(self):
        from core import workspace
        from core.pipeline import SegRecord
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp, "s.jpg")
            snap.write_bytes(b"jpg")
            seg = SegRecord(id="s1", start=0, end=1, bbox=(0, 0, 10, 5), snaps=[[0.0, str(snap), 0, 0]],
                            quad_pts=[[0, 0], [10, 0], [10, 5], [0, 5]])
            with mock.patch.object(workspace, "STATE_DIR", Path(tmp, "state")):
                workspace.save_state("v.mp4", [], [], {}, None, [seg])
                st = workspace.load_state("v.mp4")
                self.assertEqual(len(st["segments"]), 1)
                self.assertIn(str(snap.parent.resolve()), workspace.referenced_locate_folders())
                snap.unlink()                                 # 快照没了 → 定位结果视为失效
                self.assertEqual(workspace.load_state("v.mp4")["segments"], [])


class StabilizeTests(unittest.TestCase):
    @staticmethod
    def _ev(track):
        ev = SubtitleEvent(0, 3, "text")
        ev.geom_track = track
        return ev

    def test_jittery_track_fixed_to_median(self):
        from core.models import stabilize_tracks
        base = [(100, 200), (400, 200), (400, 260), (100, 260)]
        track = []
        for i in range(6):
            dx = (i % 3) - 1
            dy = ((i + 1) % 3) - 1
            track.append([i * 0.1, [[x + dx, y + dy] for x, y in base]])
        ev = self._ev(track)
        self.assertEqual(stabilize_tracks([ev]), 1)
        self.assertEqual(ev.geom_track, [])
        self.assertTrue(ev.geom_anchored)
        self.assertEqual(ev.geom, [[float(x), float(y)] for x, y in base])

    def test_moving_track_is_kept(self):
        from core.models import stabilize_tracks
        base = [(100, 200), (400, 200), (400, 260), (100, 260)]
        track = [[i * 0.1, [[x + i * 10, y] for x, y in base]] for i in range(5)]
        ev = self._ev(track)
        self.assertEqual(stabilize_tracks([ev]), 0)
        self.assertEqual(len(ev.geom_track), 5)

    def test_short_track_untouched_and_idempotent(self):
        from core.models import stabilize_tracks
        ev = self._ev([[0.0, [[0, 0], [10, 0], [10, 5], [0, 5]]]])
        self.assertEqual(stabilize_tracks([ev]), 0)
        self.assertEqual(stabilize_tracks([ev]), 0)


class VerticalSplitTests(unittest.TestCase):
    def test_split_chars_merges_fragments(self):
        from core.pipeline_quality import split_chars
        col = np.full((200, 40, 3), 30, np.uint8)
        white = (230, 230, 230)
        cv2.putText(col, "A", (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, white, 2)
        cv2.putText(col, "B", (4, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.4, white, 2)
        for y in (120, 132, 144):
            cv2.line(col, (7, y), (33, y), white, 2)
        self.assertEqual(len(split_chars(col)), 3)


class ModelDialogTests(unittest.TestCase):
    """模型库只置顶提示, 绝不自己开始下载(要用户点按钮、一键推荐, 或主窗口在用户确认后调 install_pinned)。"""

    def _dialog(self, **kw):
        from gui.model_dialog import ModelManagerDialog
        return ModelManagerDialog(AppConfig(), None, **kw)

    def test_pinned_ids_do_not_autostart(self):
        dlg = self._dialog(pinned_ids=["qwen3_8b", "lama"])
        self.assertIsNone(dlg.worker)
        self.assertEqual([t for t in dlg.findChildren(QThread) if t.isRunning()], [])
        self.assertFalse(dlg.bar.isVisible())
        self.assertEqual(dlg._chain, [])
        self.assertIn("一键安装推荐组合", dlg.lbl.text())
        self.assertIn("GB", dlg.lbl.text())
        self.assertEqual([x["id"] for x in dlg._items[:2]], ["qwen3_8b", "lama"])
        title = dlg.table.cellWidget(0, 0).findChild(QLabel).text()
        self.assertIn("· 默认", title)

    def test_install_queues_without_starting_duplicates(self):
        from core import model_manager as mm
        dlg = self._dialog(pinned_ids=["qwen3_8b"])
        it = dlg._find("qwen3_8b")
        self.assertIsNotNone(it)
        with mock.patch.object(mm, "is_installed", return_value=False), \
                mock.patch.object(mm, "InstallWorker") as worker_cls:
            worker_cls.return_value.start = lambda: None
            dlg._install([it])
            dlg._install([it])
        self.assertEqual(len(dlg._chain), 0)
        self.assertIsNotNone(dlg._current)
        self.assertEqual(dlg._current["id"], "qwen3_8b")

    def test_menu_open_still_manual(self):
        dlg = self._dialog()
        self.assertIsNone(dlg.worker)
        self.assertEqual(dlg.lbl.text(), "")


class ModelManagerTests(unittest.TestCase):
    def test_prepare_endpoint_never_downloads_silently(self):
        from core import model_manager as mm
        ep = {"provider": "ollama", "ollama_url": MANAGED_OLLAMA, "ollama_model": DEFAULT_MT_TAG}
        with mock.patch.object(mm, "ensure_ollama", return_value=MANAGED_OLLAMA), \
                mock.patch.object(mm, "is_installed", return_value=False), \
                mock.patch.object(mm, "install_sync") as install:
            with self.assertRaises(RuntimeError):
                mm.prepare_endpoint(ep)
            install.assert_not_called()

    def test_required_items_follow_actual_route(self):
        from core import model_manager as mm
        cfg = AppConfig()
        cfg.ocr_engine, cfg.locator_vlm_verify = "local", False
        cfg.pipe_translate, cfg.filter_llm = False, False
        cfg.pipe_erase = False
        self.assertEqual(mm.required_items(cfg), [])
        cfg.pipe_translate = True
        self.assertEqual([x["kind"] for x in mm.required_items(cfg)], ["ollama_llm"])

    def test_apply_lama_keeps_below_placement(self):
        from core import model_manager as mm
        cfg = AppConfig()
        cfg.placement = "below"
        item = mm.find_item(item_id="lama")
        with mock.patch.object(AppConfig, "save", lambda self: None):
            mm.apply_to_config(item, cfg)
        self.assertEqual(cfg.placement, "below")
        self.assertEqual(cfg.erase_engine, "lama")


if __name__ == "__main__":
    unittest.main()
