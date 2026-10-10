import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from core.paths import base_dir

BASE_DIR = base_dir()
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

APP_VERSION = "1.0.0"

PLACEMENTS = ("cover", "below", "erase")
ERASE_MODES = ("erase", "cover")
THEMES = ("dark", "light", "blue")
HW_TIERS = ("auto", "high", "mid", "low")
OVERWRITE_MODES = ("rename", "overwrite")
SUBTITLE_FORMATS = ("ass", "srt")
ALIGNS = ("left", "center", "right")
MANAGED_OLLAMA = "http://127.0.0.1:11435"      # 模型库托管的本地服务

# 本地模型标签: 模型库 REGISTRY / 硬件预设 / 向导 / 设置预设 的唯一来源
DEFAULT_VLM_TAG = "qwen3-vl:8b-instruct"       # 默认读字: 阿里 Qwen3-VL 8B
LIGHT_VLM_TAG = "glm-ocr:q8_0"                 # 低配读字: 智谱 GLM-OCR Q8_0(1.6GB; :latest 为 F16 2.2GB)
BIG_VLM_TAG = "qwen3.8:27b"                    # 高配读字: 阿里 Qwen3.8 27B(原生多模态, Q4 约18GB)
DEFAULT_MT_TAG = "qwen3:8b"                    # 默认翻译: 阿里 Qwen3 8B
ALT_MT_TAG = "deepseek-r1:8b"                  # 备选翻译: DeepSeek-R1-0528-Qwen3-8B
LIGHT_MT_TAG = "qwen3:4b"                      # 低配翻译

KEYRING_SERVICE = "suisen-translator"
_SECRET_EPS = ("ep_translate", "ep_vlm")


def _keyring():
    try:
        import keyring
        return keyring
    except Exception:
        return None


def resolve_secret(value):
    v = str(value or "").strip()
    if v.startswith("env:"):
        return os.environ.get(v[4:].strip(), "")
    if v.startswith("keyring:"):
        kr = _keyring()
        try:
            return (kr.get_password(KEYRING_SERVICE, v[8:].strip()) if kr else "") or ""
        except Exception:
            return ""
    return v


def stash_secret(name, value):
    """明文 → 凭据库, 返回 'keyring:名称'; 不可用/失败时原样返回。"""
    v = str(value or "").strip()
    if not v or v.startswith(("env:", "keyring:")):
        return v
    kr = _keyring()
    if kr is None:
        return v
    try:
        kr.set_password(KEYRING_SERVICE, name, v)
        return "keyring:" + name
    except Exception:
        return v


def make_ep(provider="api", **kw):
    ep = {
        "provider": provider,
        "local_model": "", "local_mmproj": "", "local_ctx": 4096,
        "ollama_url": MANAGED_OLLAMA, "ollama_model": "",
        "api_base": "", "api_key": "", "api_model": "",
    }
    ep.update(kw)
    return ep


def _ep_translate():
    return make_ep("ollama", ollama_model=DEFAULT_MT_TAG,
                   api_base="https://api.deepseek.com/v1",
                   api_model="deepseek-flash")


def _ep_vlm():
    return make_ep("ollama", ollama_model=DEFAULT_VLM_TAG,
                   api_base="https://open.bigmodel.cn/api/paas/v4",
                   api_model="glm-4v-flash")


@dataclass
class AppConfig:
    cfg_version: int = 6

    # ---- 定位追踪 ----
    precise_timing: bool = True
    track_similarity: float = 0.50
    layout_fit: str = "contain"          # contain: 字号按框高自适应, 宽度不限 / stretch: 宽高都贴合框
    subtitle_align: str = "center"       # 译文在定位框内的对齐: left/center/right(竖排: 上/中/下)
    ocr_max_tokens: int = 192
    ocr_request_budget: int = 1000
    ocr_local_fallback: bool = True
    ep_translate: dict = field(default_factory=_ep_translate)
    ep_vlm: dict = field(default_factory=_ep_vlm)
    request_timeout: int = 300           # 单次 AI 请求读超时(秒): 本地 CPU 跑批量翻译可能很慢

    # ---- 翻译 ----
    target_lang: str = "中文(简体)"
    batch_size: int = 8
    ctx_translate: bool = True

    # ---- 读字 OCR ----
    ocr_engine: str = "vlm"              # vlm(默认, 质量优先) | auto | local
    ocr_lang: str = "ch"
    conf_threshold: float = 0.6
    vlm_detail_width: int = 960
    vertical_support: bool = True
    auto_text_color: bool = False
    subtitle_lock_position: bool = True  # 锁定字幕位置(强制把轨迹收敛为中位框, 防止字幕乱动; 关掉则保留移动字幕的轨迹)

    # ---- 定位检测器 ----
    locator_rapid_version: str = "PP-OCRv5"
    locator_rapid_type: str = "server"
    locator_rapid_dml: bool = True       # DirectML 加速检测: 发行版已内嵌 onnxruntime-directml, 有 DX12 显卡即生效; 运行时探测不可用会自动回退 CPU
    locator_threads: int = 0
    locator_det_limit_side_len: int = 1280
    locator_det_box_thresh: float = 0.5
    locator_min_conf: float = 0.45
    locator_min_height_ratio: float = 0.010
    locator_max_area_ratio: float = 0.35
    locator_max_frame_candidates: int = 16
    locator_max_active: int = 40

    sample_interval: float = 0.30
    locator_gap_tolerance: float = 0.35
    locator_flash_probe: bool = True
    locator_cut_threshold: float = 22.0
    min_seg_dur: float = 0.25
    locator_min_hits: int = 2
    static_text_max_dur: float = 60.0

    locator_vlm_verify: bool = True
    locator_verify_mode: str = "ambiguous"
    locator_verify_budget: int = 100
    locator_verify_threshold: float = 0.82

    seg_bridge_gap: float = 1.2
    event_merge_gap: float = 1.6

    filter_symbol_only: bool = True
    filter_llm: bool = True
    filter_llm_batch: int = 40

    # ---- 自动样式 ----
    subtitle_safe_color: str = "#F4F4F4"
    subtitle_safe_outline: str = "#101010"
    font_fallback: bool = True

    # ---- 原字处理 / 遮罩 ----
    # placement: erase(AI无痕擦除, 译文原位) | cover(遮罩盖住, 译文原位) | below(保留原字, 译文放下方)
    # erase_mode 为 placement 的派生字段(兼容旧配置), 不再单独提供界面
    placement: str = "erase"
    erase_mode: str = "erase"
    mask_style: str = "bar"
    mask_bar_alpha: int = 55
    mask_blur: int = 25
    erase_engine: str = "lama"
    erase_python: str = ""
    erase_api_url: str = ""
    erase_dilate: int = 8
    erase_expand_ratio: float = 0.05   # 擦除范围在定位框基础上绕框心外扩的比例, 防止原字边缘漏擦
    erase_chunk: int = 60
    erase_lama_workers: int = 1        # LAMA 并行子进程数(1=不变; 每+1 约多占 1.5~2GB 显存, GPU 有余量才建议开)
    erase_lama_batch: int = 1          # LAMA 批量推理(每批帧数; 1=逐帧不变, >1 时同尺寸帧合成一张量一次推理)
    erase_lama_max_side: int = 1920    # LAMA 输入 ROI 长边上限(px, 0=不限): 4K 字幕条降到 1080p 推理再回贴, 约 3~4 倍提速
    erase_crf: int = 16
    erase_time_pad: float = 0.04
    erase_min_change: float = 0.45
    erase_cmd: dict = field(default_factory=dict)
    erase_cache_days: int = 3            # 工作区擦除缓存保留天数

    # ---- 杂项 ----
    ui_lang: str = "zh"
    ui_theme: str = "dark"
    debug_mode: bool = False
    logs_max_mb: int = 512
    tutorial_done: bool = False
    hw_tier: str = "auto"
    hw_applied: bool = False
    hw_summary: str = ""

    ffmpeg_path: str = ""
    hw_decode: bool = True             # 硬件解码(仅加速悬停截图等单发寻址取帧; 顺序取帧走 PyAV 软解更快, 不用于流水线)
    libmpv_path: str = ""              # 预览播放器 libmpv-2.dll 路径; 留空=models/bin → 程序目录 → PATH, 没有则首次启动下载
    libmpv_url: str = ""               # 自定义 libmpv 下载地址(zip/7z, 内含 libmpv-2.dll); 留空=mpv 官方 Windows 构建最新版

    # ---- 顶部流水线(01 定位 / 02 识别 / 03 翻译 / 04 原字处理 / 05 导出) ----
    pipe_locate: bool = True
    pipe_read: bool = True
    pipe_translate: bool = True
    pipe_erase: bool = True              # 04 原字处理: 勾选=按 placement 处理原字幕; 不勾=保留
    pipe_render: bool = True
    pipe_subs: bool = True               # 05 导出时同时生成字幕文件(格式见 subtitle_format)

    # ---- 输出 ----
    out_dir: str = ""
    out_suffix: str = "_translated"
    out_overwrite: str = "rename"        # 同名输出: rename=自动改名 / overwrite=覆盖
    export_direct: bool = False          # 队列: 擦除+嵌字一次完成(单次编码, 不写擦除缓存)
    subtitle_format: str = "ass"         # 字幕文件格式: ass(与画面排布一致, 默认) / srt(纯文本)
    subtitle_font: str = "Microsoft YaHei"
    keep_audio: bool = True
    out_bilingual: bool = False
    video_encoder: str = "auto"        # 成品编码器: auto(有 N 卡用 NVENC) / libx264 / h264_nvenc / h264_qsv / h264_amf

    recent: list = field(default_factory=list)
    prompts: dict = field(default_factory=dict)
    first_run: bool = False

    def ep(self, role):
        src = self.ep_translate if role == "translate" else self.ep_vlm
        ep = deepcopy(src)
        ep["api_key"] = resolve_secret(ep.get("api_key", ""))
        return ep

    def add_recent(self, path):
        r = [p for p in self.recent if p != path]
        r.insert(0, path)
        self.recent = r[:10]
        self.save()

    def _normalize(self):
        if self.video_encoder not in ("auto", "libx264", "h264_nvenc", "h264_qsv", "h264_amf"):
            self.video_encoder = "auto"
        if self.subtitle_align not in ALIGNS:
            self.subtitle_align = "center"
        if self.layout_fit not in ("contain", "stretch"):
            self.layout_fit = "contain"
        if self.placement not in PLACEMENTS:
            self.placement = "erase"
        if self.erase_mode not in ERASE_MODES:
            self.erase_mode = "erase"
        if self.placement in ERASE_MODES:
            self.erase_mode = self.placement
        if self.out_overwrite not in OVERWRITE_MODES:
            self.out_overwrite = "rename"
        self.request_timeout = int(max(30, min(3600, int(self.request_timeout or 300))))
        self.erase_lama_workers = int(max(1, min(4, int(self.erase_lama_workers or 1))))
        self.erase_lama_batch = int(max(1, min(8, int(self.erase_lama_batch or 1))))
        if self.subtitle_format not in SUBTITLE_FORMATS:
            self.subtitle_format = "ass"

    def save(self):
        import tempfile
        self._normalize()
        for attr in _SECRET_EPS:
            ep = getattr(self, attr)
            if isinstance(ep, dict):
                ep["api_key"] = stash_secret(attr, ep.get("api_key", ""))
        data = asdict(self)
        data.pop("first_run", None)
        fd, temporary = tempfile.mkstemp(prefix="config-", suffix=".tmp", dir=BASE_DIR)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, CONFIG_PATH)
        except Exception:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls):
        c = cls()
        try:
            with open(CONFIG_PATH, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            c.first_run = True
            return c
        _migrate_old(data)
        for k, v in data.items():
            if not hasattr(c, k) or k == "first_run":
                continue
            cur = getattr(c, k)
            if isinstance(cur, dict) and isinstance(v, dict):
                merged = deepcopy(cur)
                merged.update(v)
                setattr(c, k, merged)
            else:
                setattr(c, k, v)
        c._normalize()
        plain = any(
            str((getattr(c, a) or {}).get("api_key", "")).strip()
            and not str(getattr(c, a)["api_key"]).startswith(("env:", "keyring:"))
            for a in _SECRET_EPS)
        if plain and _keyring():
            try:
                c.save()
            except Exception:
                pass
        return c


def _migrate_old(data):
    def equals(key, old):
        try:
            return abs(float(data.get(key, old)) - old) < 1e-9
        except (TypeError, ValueError):
            return False

    version = int(data.get("cfg_version", 1) or 1)

    if data.get("ocr_engine") in ("rapid", "easy", "locate"):
        data["ocr_engine"] = "auto"
    if data.get("ocr_engine") == "paddle":
        data["ocr_engine"] = "local"
    if version >= 3 and data.get("ocr_engine") == "auto":
        data["ocr_engine"] = "vlm"
    old_pipe = bool(data.get("pipe_ocr", True))
    data.setdefault("pipe_locate", old_pipe)
    data.setdefault("pipe_read", old_pipe)

    if equals("locator_min_conf", 0.35):
        data["locator_min_conf"] = 0.5
    if equals("locator_min_conf", 0.5):
        data["locator_min_conf"] = 0.45
    if equals("locator_det_box_thresh", 0.6):
        data["locator_det_box_thresh"] = 0.5
    if equals("locator_min_height_ratio", 0.012):
        data["locator_min_height_ratio"] = 0.010
    if equals("track_similarity", 0.86):
        data["track_similarity"] = 0.5
    if equals("erase_time_pad", 0.12):
        data["erase_time_pad"] = 0.04
    if equals("sample_interval", 0.35):
        data["sample_interval"] = 0.30
    if equals("min_seg_dur", 0.22):
        data["min_seg_dur"] = 0.25
    if equals("locator_max_frame_candidates", 48):
        data["locator_max_frame_candidates"] = 12
    if equals("locator_max_frame_candidates", 12):
        data["locator_max_frame_candidates"] = 16
    if data.get("placement") not in PLACEMENTS:
        data["placement"] = "erase"

    if "keep_debug_images" in data and "debug_mode" not in data:
        data["debug_mode"] = bool(data.pop("keep_debug_images"))
    if version < 2:
        data["keep_audio"] = True
    if version < 3:
        ep = data.get("ep_translate")
        if isinstance(ep, dict) and "hy-mt" in str(ep.get("ollama_model", "")).lower():
            ep["ollama_model"] = DEFAULT_MT_TAG
        data.setdefault("pipe_erase", data.get("placement", "erase") == "erase")
    if version < 4:
        data["auto_text_color"] = False
    if version < 5:
        # 发行版已内嵌 onnxruntime-directml 且运行时探测可用性(_dml_available):
        # 默认开启, 没有 DX12 显卡的机器探测失败自动走 CPU
        data.setdefault("locator_rapid_dml", True)
    if version < 6:
        data.setdefault("pipe_subs", bool(data.get("pipe_ass", True)))   # pipe_ass → 字幕文件统一开关
    if data.get("subtitle_format") not in SUBTITLE_FORMATS:
        data["subtitle_format"] = "ass"
    if data.get("erase_mode") not in ERASE_MODES:
        data["erase_mode"] = "cover" if data.get("placement") == "cover" else "erase"
    if data.get("ui_theme") not in THEMES:
        data["ui_theme"] = "dark"
    if data.get("hw_tier") not in HW_TIERS:
        data["hw_tier"] = "auto"
    if data.get("out_overwrite") not in OVERWRITE_MODES:
        data["out_overwrite"] = "rename"
    data["cfg_version"] = 6
