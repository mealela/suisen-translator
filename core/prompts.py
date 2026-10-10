"""所有AI提示词默认值；设置页可覆盖。"""

DEFAULTS = {
    "translate_system": (
        "你是专业的字幕翻译引擎。把用户输入的字幕翻译成{target}，"
        "口语化、简洁自然，符合目标语言的字幕习惯。"
        "只输出译文本身，不要解释、引号或前缀。"
    ),

    "translate_batch": (
        "以下是同一视频按时间顺序排列的字幕。{context}"
        "请把JSON数组中的每条翻译成{target}，保持前后人称、语气和专有名词一致。"
        "译文必须简洁，适合在原字幕区域显示。\n"
        "{payload}\n"
        "严格输出与输入等长的JSON字符串数组，不要markdown，不要解释。"
        "只输出译文数组; 任何情况下不要输出原文, 不要输出\"原文→译文\"映射。"
    ),

    "vlm_read": (
        "只转写本图中叠加/烧录的字幕文字，保持原语言、原标点与原有假名，"
        "不要归一化或改写。"
        "画面角落的水印、台标、频道名、用户名、UI按钮、时间码不是字幕，不要转写。"
        "不要翻译、猜测、补全看不清的字，不要描述画面或输出坐标。"
        "禁止使用任何工具、插件或联网搜索，直接根据图片内容回答，不要额外消耗。"
        "如果有多行，按从上到下顺序用换行分隔(竖排则按从右到左的列顺序)。"
        "看不清或没有字幕时 text 为空字符串。"
        '只输出JSON对象：{"text":"可见文字"}。'
    ),

    "locator_verify": (
        "这是同一个视频候选区域在不同时间点的截图拼图。候选可能完全不是字幕。"
        "判断其中是否存在叠加或烧录在视频上的字幕、歌词、对白或说明性花字。"
        "场景内自然文字、招牌、路牌、包装、衣服文字、UI、用户名、水印、"
        "频道台标、时间码均为非字幕。"
        "除非文字轮廓清楚且在多帧中具有一致的屏幕位置，否则判为false。"
        "禁止使用任何工具、插件或联网搜索，直接根据图片内容判断，不要额外消耗。"
        '只输出JSON：{"subtitle":true或false,"confidence":0到1,'
        '"line_count":整数,"vertical":true或false,"reason":"短原因"}。'
    ),

    "subtitle_filter": (
        "以下JSON数组来自视频OCR。字段：t=文本，sec=持续秒数，"
        "y/x=画面相对位置(0~1)，hr=行高/画面高，c=识别置信度，src=来源。"
        "逐项判断是否为需要翻译的台词、歌词、旁白或说明字幕。"
        "以下情况输出false：水印、用户名、频道名、台标、UI/按钮、时间码、乱码、纯符号、"
        "场景内招牌/包装/屏幕上的文字，以及一闪而过的效果音/拟声词特效字"
        "（如 ドクン、ズブッ、ゴゴゴ、♥连打 这类非对白的拟声/效果文字）。"
        "简短语气词和叹词(如 嗯、啊、はい、いや、ダメ)属于对白，输出true，"
        "不要因文本短或口语化而判false。拿不准的一律输出true。\n"
        "{payload}\n"
        "严格输出与输入等长的JSON布尔数组，不要解释。"
    ),
}

TITLES = {
    "translate_system": "翻译 · 系统提示词",
    "translate_batch": "翻译 · 批量用户提示词",
    "vlm_read": "读字 · 候选区域转写(通用VLM)",
    "locator_verify": "定位 · 多帧候选核验",
    "subtitle_filter": "过滤 · LLM语义过滤(效果音/水印/UI)",
}

HINTS = {
    "translate_system": "占位符：{target}",
    "translate_batch": "占位符：{target}、{payload}、{context}",
    "vlm_read": "无占位符。GLM-OCR 走固定指令不用此提示词。必须要求输出 {\"text\":...} JSON。",
    "locator_verify": "无占位符。输入为同一候选区域的多帧拼图。需要通用VLM(GLM-OCR不支持)。",
    "subtitle_filter": "占位符：{payload}。使用翻译端点(文本LLM)。是否启用见[设置→定位与时间→质量过滤]。",
}


def get_prompt(cfg, key):
    try:
        text = (cfg.prompts or {}).get(key, "")
    except Exception:
        text = ""
    text = (text or "").strip()
    return text if text else DEFAULTS[key]


def render(tpl, **kw):
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl
