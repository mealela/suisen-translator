# -*- coding: utf-8 -*-
"""字幕定位检测层(纯 ONNX Runtime)。
RapidOCR 加载 PP-OCRv5/v6 检测模型 → 解析四点框 → 几何卫生过滤 → NMS。
定位框是全流程唯一位置权威, 本层不做任何几何修正。
置信度分两档: 达到 locator_min_conf 的为正式候选;
在其 0.6 倍以上的为弱检测(weak=True), 只用于已存在轨迹的续命/关联,
从不新建轨迹——提高召回同时不引入新误检。
方向判定用四点框自身的长轴(core.geometry.quad_is_vertical): 斜排横向字幕不会被当成竖排;
NMS 与面积上限用多边形: 斜排相邻行的外接矩形重叠很大, 用外接矩形会把真字幕压掉。"""
import logging
import math
import os

import cv2
import numpy as np

from core.geometry import quad_area, quad_axis, quad_is_vertical

log = logging.getLogger("suisen.locator")
_LOCATOR_CACHE = {}
_CV_THREADS_SET = False


def worker_threads(cfg=None):
    """检测/识别 intra-op 线程数: 默认留 2 个核给 GUI 与视频解码"""
    n = int(getattr(cfg, "locator_threads", 0) or 0)
    return n if n > 0 else max(1, (os.cpu_count() or 4) - 2)


def limit_cv_threads(cfg=None):
    global _CV_THREADS_SET
    if _CV_THREADS_SET:
        return
    try:
        cv2.setNumThreads(worker_threads(cfg))
    except Exception:
        pass
    _CV_THREADS_SET = True


def _quad_to_loc(poly, conf=0.82, source="rapid"):
    """模型四点框 → 统一结构; 方向与角度取自框自身长轴"""
    pts = [(float(p[0]), float(p[1])) for p in poly[:4]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    angle = 0.0
    if len(pts) == 4:
        vertical = quad_is_vertical(pts)
        if not vertical:
            val, _, _ = quad_axis(pts)
            if 2.0 <= abs(val) <= 60.0:
                angle = round(val, 1)
    else:
        vertical = (y2 - y1) > (x2 - x1) * 1.6
    return {"bbox": (x1, y1, x2, y2), "quad": pts, "conf": float(conf),
            "angle": angle, "vertical": vertical, "source": source,
            "weak": False}


# ---------- RapidOCR 输出解析(兼容 1.x list / 2.x-3.x 对象) ----------

def _item_quad_conf(item):
    candidate, conf = item, None
    if isinstance(item, (list, tuple)) and len(item) in (2, 3):
        first = item[0]
        try:
            if len(first) >= 4 and not isinstance(
                    first[0], (int, float, np.number)):
                candidate = first
                if len(item) == 3 and isinstance(
                        item[2], (int, float, np.number)):
                    conf = float(item[2])
                elif (len(item) == 2 and isinstance(item[1], (list, tuple))
                      and len(item[1]) == 2):
                    conf = float(item[1][1])
        except Exception:
            pass
    try:
        pts = [(float(p[0]), float(p[1])) for p in candidate]
    except Exception:
        return None, None
    return (pts[:4], conf) if len(pts) >= 4 else (None, None)


def _extract_rapid_quads(result):
    if result is None:
        return []
    if isinstance(result, tuple):
        result = result[0]
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        scores = getattr(result, "scores", None)
        try:
            scores = list(scores) if scores is not None else []
        except TypeError:
            scores = []
        out = []
        for i, poly in enumerate(boxes):
            try:
                pts = [(float(p[0]), float(p[1])) for p in poly]
            except Exception:
                continue
            if len(pts) >= 4:
                conf = 0.82
                if i < len(scores):
                    try:
                        conf = float(scores[i])
                    except (TypeError, ValueError):
                        pass
                out.append((pts[:4], conf))
        return out
    try:
        items = list(result)
    except Exception:
        return []
    out = []
    for item in items:
        pts, conf = _item_quad_conf(item)
        if pts:
            out.append((pts, conf if conf is not None else 0.82))
    return out


# ---------- RapidOCR 参数组(逐级回退) ----------

def _enum_member(enum_cls, *names):
    for n in names:
        v = getattr(enum_cls, n, None)
        if v is not None:
            return v
    return None


def _dml_available():
    """onnxruntime-directml 装了才允许 use_dml, 否则 CPU 版 onnxruntime 会因为未知 provider 报错/回退。"""
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _engine_common(cfg):
    from core.paths import base_path
    p = {"EngineConfig.onnxruntime.intra_op_num_threads": worker_threads(cfg),
         "Global.model_root_dir": str(base_path() / "models" / "rapidocr")}
    if bool(getattr(cfg, "locator_rapid_dml", False)) and _dml_available():
        p["EngineConfig.onnxruntime.use_dml"] = True
    return p


def _versions(cfg):
    """按配置排出 (OCRVersion, is_v6) 候选: 要 v6 先试 v6, 再退 v5/v4"""
    from rapidocr import OCRVersion
    want6 = "6" in str(getattr(cfg, "locator_rapid_version", "PP-OCRv5") or "")
    out = []
    for name, is6 in (("PPOCRV6", True), ("PPOCRV5", False), ("PPOCRV4", False)):
        v = getattr(OCRVersion, name, None)
        if v is None or (is6 and not want6):
            continue
        out.append((v, is6))
    return out


def _model_type(is6, big):
    from rapidocr import ModelType
    if is6:
        names = ("MEDIUM", "SERVER") if big else ("SMALL", "MOBILE")
    else:
        names = ("SERVER",) if big else ("MOBILE",)
    return _enum_member(ModelType, *names) or ModelType.MOBILE


def det_param_sets(cfg):
    try:
        from rapidocr import EngineType
        versions = _versions(cfg)
    except Exception:
        return [{}]
    big = str(getattr(cfg, "locator_rapid_type", "server") or "server").lower() == "server"
    common = _engine_common(cfg)
    common.update({"Global.use_cls": False, "Global.use_rec": False})
    extra = {}
    try:
        side = int(getattr(cfg, "locator_det_limit_side_len", 0) or 0)
        if side > 0:
            extra.update({"Det.limit_side_len": side, "Det.limit_type": "max"})
    except (TypeError, ValueError):
        pass
    try:
        bt = float(getattr(cfg, "locator_det_box_thresh", 0) or 0)
        if bt:
            extra["Det.box_thresh"] = bt
    except (TypeError, ValueError):
        pass
    sets = []
    for ver, is6 in versions:
        base = dict(common)
        base.update({"Det.engine_type": EngineType.ONNXRUNTIME,
                     "Det.ocr_version": ver,
                     "Det.model_type": _model_type(is6, big)})
        sets.append({**base, **extra})
        sets.append(base)
    sets.append({**common, **extra})
    sets.append(dict(common))
    sets.append({})
    return sets


def rec_param_sets(cfg, lang):
    try:
        from rapidocr import EngineType, LangRec
        versions = _versions(cfg)
    except Exception:
        return [{}]
    # PP-OCRv5 的 CH 识别模型原生覆盖日文假名/汉字, japan 退到 CH 也是正确的
    names = {"ch": ("CH",), "en": ("EN", "CH"), "japan": ("JAPAN", "CH"),
             "korean": ("KOREAN", "CH")}.get(lang, ("CH",))
    common = _engine_common(cfg)
    common.update({"Global.use_cls": False, "Global.use_det": False})
    sets = []
    for ver, is6 in versions:
        for n in names:
            lr = getattr(LangRec, n, None)
            if lr is None:
                continue
            p = dict(common)
            p.update({"Rec.engine_type": EngineType.ONNXRUNTIME,
                      "Rec.lang_type": lr,
                      "Rec.ocr_version": ver,
                      "Rec.model_type": _model_type(is6, False)})
            sets.append(p)
    sets.append(dict(common))
    sets.append({})
    return sets


def _construct(param_sets, tag):
    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise RuntimeError("未安装 rapidocr: pip install rapidocr onnxruntime")
    err = None
    for params in param_sets:
        try:
            return (RapidOCR(params=params) if params else RapidOCR()), params
        except Exception as exc:
            err = exc
            log.debug("RapidOCR %s 参数组不可用 %s: %s", tag,
                      {k: str(v) for k, v in params.items()}, exc)
    raise RuntimeError(f"RapidOCR {tag} 初始化失败: {err}")


class RapidBackend:
    name = "rapid"

    def __init__(self, cfg=None):
        limit_cv_threads(cfg)
        self.ocr, self.params = _construct(det_param_sets(cfg), "检测")
        self.mode = None
        probe = np.full((64, 240, 3), 255, np.uint8)
        cv2.putText(probe, "Ab12", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        for mode in ("det_only", "plain"):
            try:
                _extract_rapid_quads(self._call(probe, mode))
                self.mode = mode
                break
            except Exception:
                continue
        if self.mode is None:
            raise RuntimeError("RapidOCR 接口探测失败")
        log.info("RapidOCR 检测就绪(%s): %s", self.mode,
                 {k: str(v) for k, v in self.params.items()})

    def _call(self, image, mode):
        if mode == "det_only":
            return self.ocr(image, use_det=True, use_cls=False, use_rec=False)
        return self.ocr(image)

    def detect(self, image):
        try:
            pairs = _extract_rapid_quads(self._call(image, self.mode))
        except Exception:
            log.exception("RapidOCR 定位失败")
            return []
        return [_quad_to_loc(p, c, self.name) for p, c in pairs]


class RapidLineReader:
    """RapidOCR 单行识别(本地快路读字)"""

    def __init__(self, lang="ch", cfg=None):
        limit_cv_threads(cfg)
        self.ocr, self.params = _construct(rec_param_sets(cfg, lang), f"识别[{lang}]")
        log.info("RapidOCR 识别就绪: %s", {k: str(v) for k, v in self.params.items()})

    def read_line(self, crop):
        res = self.ocr(crop, use_det=False, use_cls=False, use_rec=True)
        if isinstance(res, tuple):
            res = res[0]
        txts = getattr(res, "txts", None)
        scores = getattr(res, "scores", None)
        if txts:
            text = str(txts[0]).strip()
            try:
                score = float(scores[0]) if scores is not None else 0.6
            except (TypeError, ValueError, IndexError):
                score = 0.6
            return text, score
        return "", 0.0


# ---------- 卫生钳制 ----------

def _sanitize(image, dets, cfg):
    from core.v6_common import ordered_quad, qbox
    h, w = image.shape[:2]
    min_conf = float(getattr(cfg, "locator_min_conf", 0.45))
    weak_floor = min_conf * 0.6
    min_side = max(4.0, float(getattr(cfg, "locator_min_height_ratio", 0.010)) * h)
    max_area = float(getattr(cfg, "locator_max_area_ratio", 0.35)) * w * h
    vertical_ok = bool(getattr(cfg, "vertical_support", True))
    out = []
    for original in dets:
        try:
            det = dict(original)
            conf = float(det.get("conf", 0))
            if not math.isfinite(conf) or conf < weak_floor:
                continue
            q = ordered_quad(det["quad"], w, h)
            x1, y1, x2, y2 = qbox(q)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                continue
            _, _, short_side = quad_axis(q)
            vertical = quad_is_vertical(q)
            if vertical and not vertical_ok:
                continue            # 未开启竖排: 竖长框多为纹理/特效
            if short_side < min_side:
                continue            # 字高(竖排为列宽)太小: 细线/噪点(按框自身短边, 斜排不受外接矩形影响)
            if quad_area(q) > max_area:
                continue            # 整屏大块(按多边形面积, 斜排长行不会被外接矩形误伤)
            det.update(quad=q.tolist(), bbox=(x1, y1, x2, y2),
                       quality=conf, vertical=vertical,
                       weak=conf < min_conf)
            out.append(det)
        except (ValueError, TypeError, KeyError):
            continue
    return out


def _nms(dets, threshold=0.55):
    """多边形 IoU 的 NMS: 斜排相邻行的外接矩形会大面积重叠, 用外接矩形会把真字幕压掉。"""
    from core.v6_common import qiou
    dets = sorted(dets, key=lambda d: (bool(d.get("weak")),
                                       -float(d.get("conf", 0.0))))
    kept = []
    for det in dets:
        if all(qiou(det["quad"], old["quad"]) <= threshold for old in kept):
            kept.append(det)
    return kept


class Locator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.note = ""
        self.backend = RapidBackend(cfg)

    def detect(self, image):
        dets = _sanitize(image, self.backend.detect(image), self.cfg)
        dets = _nms(dets)
        limit = max(1, int(getattr(self.cfg, "locator_max_frame_candidates", 16)))
        strong = [d for d in dets if not d.get("weak")][:limit]
        weak = [d for d in dets if d.get("weak")][:max(4, limit // 2)]
        return strong + weak


_KEY_FIELDS = (
    "locator_rapid_version", "locator_rapid_type", "locator_rapid_dml",
    "locator_threads", "locator_det_limit_side_len", "locator_det_box_thresh",
    "locator_min_conf", "locator_min_height_ratio", "locator_max_area_ratio",
    "locator_max_frame_candidates", "vertical_support",
)


def get_locator(cfg=None):
    from copy import deepcopy
    from config import AppConfig
    cfg = deepcopy(cfg if cfg is not None else AppConfig())
    key = tuple(getattr(cfg, name, None) for name in _KEY_FIELDS)
    if key not in _LOCATOR_CACHE:
        _LOCATOR_CACHE[key] = Locator(cfg)
    return _LOCATOR_CACHE[key]
