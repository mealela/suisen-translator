import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import copy
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from config import AppConfig
from core.ass_generator import _t, generate_ass, generate_srt, parse_ass
from core.models import MaskEvent, SubtitleEvent, cover_quad, rect_quad
from core.ocr_protocols import BadOCR, validate_ocr
from core.pipeline import SegRecord, _make_event
from core.v6_common import qiou
from core.v6_paint import ass_layout, event_paths, paint_frame
from core.v6_render import frame_mask

app = QApplication.instance() or QApplication([])


class V6Tests(unittest.TestCase):
    def test_repetition_is_rejected(self):
        with self.assertRaises(BadOCR):
            validate_ocr("くう～！入力画面```markdown" + "入力画面```" * 40)

    def test_short_interjection_is_allowed(self):
        self.assertEqual(validate_ocr("くう～！"), "くう～！")

    def test_multiline_is_not_flattened(self):
        with self.assertRaises(BadOCR):
            validate_ocr("第一行\n第二行")

    def test_geometry_is_not_changed_by_reading(self):
        cfg = AppConfig()
        q = [[10, 20], [210, 40], [207, 70], [7, 50]]
        seg = SegRecord(id="test", start=1, end=2, bbox=(7, 20, 210, 70), quad_pts=q)
        original = copy.deepcopy(seg.quad())
        ev = _make_event(seg, "测试", 0.5, cfg)
        self.assertEqual(seg.quad(), original)
        self.assertEqual(ev.geom, original)
        self.assertEqual(ev.mask, list(seg.bbox))
        self.assertFalse(ev.mask_on)          # 自动遮罩不再写无效的 mask_on, 由处理方式+04勾选决定

    def test_parallel_lines_do_not_overlap(self):
        a = [[0, 0], [100, 0], [100, 20], [0, 20]]
        b = [[0, 30], [100, 30], [100, 50], [0, 50]]
        self.assertEqual(qiou(a, b), 0.0)

    def test_dilation_applies_to_polygon(self):
        cfg = AppConfig()
        cfg.erase_time_pad = 0
        ev = SubtitleEvent(0, 1, "test", geom=[[20, 20], [60, 30], [57, 45], [17, 35]])
        cfg.erase_dilate = 0
        cfg.erase_expand_ratio = 0.0
        small = frame_mask([ev], [], 0.5, 100, 100, cfg)
        cfg.erase_dilate = 5
        large = frame_mask([ev], [], 0.5, 100, 100, cfg)
        self.assertGreater(np.count_nonzero(large), np.count_nonzero(small))

    def test_expand_ratio_enlarges_and_clips(self):
        cfg = AppConfig()
        cfg.erase_time_pad = 0
        cfg.erase_dilate = 0
        ev = SubtitleEvent(0, 1, "test", geom=[[20, 20], [60, 20], [60, 40], [20, 40]])
        cfg.erase_expand_ratio = 0.0
        base = frame_mask([ev], [], 0.5, 100, 100, cfg)
        cfg.erase_expand_ratio = 0.5
        big = frame_mask([ev], [], 0.5, 100, 100, cfg)
        self.assertGreater(np.count_nonzero(big), np.count_nonzero(base))
        # 外扩不越界: 输出形状/取值仍合法
        ev2 = SubtitleEvent(0, 1, "test", geom=[[0, 0], [98, 0], [98, 98], [0, 98]])
        cfg.erase_expand_ratio = 0.5
        m2 = frame_mask([ev2], [], 0.5, 100, 100, cfg)
        self.assertEqual(m2.shape, (100, 100))
        self.assertGreater(np.count_nonzero(m2), 0)

    def test_time_carry(self):
        self.assertEqual(_t(59.999), "0:01:00.00")
        ev = SubtitleEvent(59.9999, 61, "test")
        self.assertIn("00:01:00,000", generate_srt([ev]))

    def test_ass_multiline_roundtrip(self):
        ev = SubtitleEvent(0, 2, "第一行\n第二行", x=100, y=100)
        result, _ = parse_ass(generate_ass([ev], [], 640, 360))
        self.assertEqual(result[0].text, "第一行\n第二行")

    def test_ass_playres_scaling(self):
        ass = ("[Script Info]\nPlayResX: 640\nPlayResY: 360\n\n"
               "[Events]\n"
               "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
               r"Dialogue: 1,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\an5\pos(320,180)\fs40\bord2}字幕")
        evs, _ = parse_ass(ass, 1280, 720)
        self.assertEqual(len(evs), 1)
        self.assertAlmostEqual(evs[0].x, 640.0)
        self.assertAlmostEqual(evs[0].y, 360.0)
        self.assertEqual(evs[0].font_size, 80)
        self.assertAlmostEqual(evs[0].outline_width, 4.0)
        evs2, _ = parse_ass(ass)
        self.assertAlmostEqual(evs2[0].x, 320.0)
        self.assertEqual(evs2[0].font_size, 40)
        ev = SubtitleEvent(0, 2, "test", x=100, y=100, font_size=40)
        evs3, _ = parse_ass(generate_ass([ev], [], 1280, 720), 1280, 720)
        self.assertAlmostEqual(evs3[0].x, 100.0)
        self.assertEqual(evs3[0].font_size, 40)

    def test_ass_mask_geometry_matches_cover_quad(self):
        """ASS 里的字幕遮罩与画面合成用同一个四点框(cover_quad), 不再是 bbox 外扩再旋转。"""
        ev = SubtitleEvent(0, 2, "文字", x=100, y=50, geom=[[10, 30], [190, 40], [188, 70], [8, 60]])
        text = generate_ass([ev], [], 640, 360, placement="cover", pipe_erase=True, mask_style="solid")
        mask_line = next(l for l in text.splitlines() if ",mask," in l)
        self.assertIn("m 10 30 l 190 40 188 70 8 60", mask_line)
        self.assertEqual(cover_quad(ev), [[10.0, 30.0], [190.0, 40.0], [188.0, 70.0], [8.0, 60.0]])
        ev.mask_custom, ev.mask = True, [0, 0, 50, 20]
        self.assertEqual(cover_quad(ev), [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]])

    def test_paint_does_not_mutate_geometry(self):
        cfg = AppConfig()
        seg = SegRecord(id="a", start=0, end=1, bbox=(10, 10, 300, 60),
                        quad_pts=[[10, 10], [300, 10], [300, 60], [10, 60]])
        ev = _make_event(seg, "测试字幕", 0.5, cfg)
        before = copy.deepcopy(ev.geom)
        fill, edge = event_paths(ev, "Arial", 0.5)
        self.assertEqual(ev.geom, before)
        self.assertFalse(edge.isEmpty())
        self.assertLessEqual(edge.boundingRect().height(), 50.1)

    def test_ass_layout_follows_paint(self):
        """ASS 的有效字号/中心取自与画面合成同一套排布: 中心落在原字框中心, 字号按格高换算且 >0。"""
        cfg = AppConfig()
        seg = SegRecord(id="b", start=0, end=1, bbox=(100, 500, 700, 560),
                        quad_pts=[[100, 500], [700, 500], [700, 560], [100, 560]])
        ev = _make_event(seg, "一句不长的字幕", 0.9, cfg)
        lay = ass_layout(ev, "Arial")
        self.assertIsNotNone(lay)
        self.assertGreater(lay["fs"], 0)
        cx, cy = lay["center"]
        self.assertAlmostEqual(cx, 400.0, delta=2.0)
        self.assertAlmostEqual(cy, 530.0, delta=2.0)
        text = generate_ass([ev], [], 1280, 720, layout=lambda e: ass_layout(e, "Arial"))
        self.assertIn(f"\\fs{lay['fs']}", text)

    def test_font_fits_box_height_only(self):
        """字号只由框高决定: 文字(含描边)高 ≈ 框高×0.98, 宽度不受框宽限制。"""
        cfg = AppConfig()
        seg = SegRecord(id="c", start=0, end=1, bbox=(100, 500, 300, 560),
                        quad_pts=[[100, 500], [300, 500], [300, 560], [100, 560]])
        ev = _make_event(seg, "一句明显比框宽得多的长字幕文字", 0.9, cfg)
        _, edge = event_paths(ev, "Arial", 0.5)
        b = edge.boundingRect()
        self.assertAlmostEqual(b.height(), 60 * 0.98, delta=1.5)
        self.assertGreater(b.width(), 200)
        _, edge_l = event_paths(ev, "Arial", 0.5, align="left")
        self.assertAlmostEqual(edge_l.boundingRect().left(), 100.0, delta=1.5)

    def test_ass_roundtrip_keeps_box(self):
        ev = SubtitleEvent(0, 2, "文字", x=200, y=50, geom=[[100, 30], [300, 30], [300, 70], [100, 70]],
                           fit_mode="contain")
        evs, _ = parse_ass(generate_ass([ev], [], 640, 360, layout=lambda e: ass_layout(e, "Arial")))
        self.assertEqual(evs[0].geom, [[100.0, 30.0], [300.0, 30.0], [300.0, 70.0], [100.0, 70.0]])
        self.assertEqual(evs[0].fit_mode, "contain")

    def test_vlm_schema_echo_is_rejected(self):
        """模型把提示词字段名/占位符当内容输出(schema 回显 {"text":"text"}/"可见文字") → 视为无结果, 不落成字幕。"""
        import numpy as np
        from core.pipeline import read_region
        cfg = AppConfig()
        cfg.ocr_engine = "vlm"

        class EchoClient:
            def ask(self, img, prompt, max_tokens=0):
                return '{"text": "text"}'

        class EchoClient2:
            def ask(self, img, prompt, max_tokens=0):
                return '{"text": "可见文字"}'

        crop1 = np.zeros((40, 200, 3), np.uint8)
        crop2 = np.full((40, 200, 3), 255, np.uint8)      # 不同画面 → 不同缓存键
        stub_local = lambda: type("L", (), {"read": lambda self, c, v: ("", 0.0)})()
        text1, _, _ = read_region(crop1, False, cfg, EchoClient(), stub_local, [], [0, 100], "a")
        text2, _, _ = read_region(crop2, False, cfg, EchoClient2(), stub_local, [], [0, 100], "b")
        self.assertEqual(text1, "")
        self.assertEqual(text2, "")

    def test_vlm_cached_schema_echo_is_rejected(self):
        """缓存里残留的 schema 回显结果同样要被拦下(命中缓存也走同一条拒绝规则)。"""
        import hashlib
        import numpy as np
        from core.pipeline import read_region
        from core.v6_common import cache_key, cache_put, endpoint_identity
        cfg = AppConfig()
        cfg.ocr_engine = "vlm"
        crop = np.full((40, 200, 3), 7, np.uint8)
        key = cache_key("ocr-v7", hashlib.sha256(crop.tobytes()).hexdigest(),
                        endpoint_identity(cfg.ep("vlm")))
        cache_put(key, "text")

        class NoCall:
            def ask(self, *a, **k):
                raise AssertionError("命中缓存时不应再调用模型")

        stub_local = lambda: type("L", (), {"read": lambda self, c, v: ("", 0.0)})()
        text, _, _ = read_region(crop, False, cfg, NoCall(), stub_local, [], [0, 100], "c")
        self.assertEqual(text, "")

    def test_flag_growing_tracks_marks_drift(self):
        """字幕框随时间渐进变大(疑似特效字幕定位漂移) → 标为待复核; 稳定轨迹不标。"""
        from core.models import flag_growing_tracks
        grow = [(0.1 * i, [[100, 100], [300, 100], [300, 100 + int(50 + 2 * i)],
                           [100, 100 + int(50 + 2 * i)]]) for i in range(10)]
        ev = SubtitleEvent(0, 2, "文字", geom=rect_quad((100, 100, 300, 150)))
        ev.geom_track = grow
        self.assertEqual(flag_growing_tracks([ev]), 1)
        self.assertTrue(ev.needs_review())
        self.assertIn("定位漂移", ev.review_reason)

        stable = [(0.1 * i, [[100, 100], [300, 100], [300, 150], [100, 150]]) for i in range(10)]
        ev2 = SubtitleEvent(0, 2, "文字", geom=rect_quad((100, 100, 300, 150)))
        ev2.geom_track = stable
        self.assertEqual(flag_growing_tracks([ev2]), 0)
        self.assertFalse(ev2.needs_review())

    def test_paint_frame_is_noop_without_activity(self):
        cfg = AppConfig()
        frame = np.zeros((40, 60, 3), np.uint8)
        ev = SubtitleEvent(5, 6, "later", x=30, y=20)
        self.assertIs(paint_frame(frame, [ev], [], 1.0, cfg), frame)
        m = MaskEvent(0, 1, 5, 5, 30, 20)
        out = paint_frame(frame, [], [m], 0.5, cfg)
        self.assertIsNot(out, frame)
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
