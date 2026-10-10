import os
import re

_BREAKS = re.compile(r"\\N|\r?\n")
_CJK = re.compile(r"[\u2e80-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def is_glm_ocr(ep):
    provider = ep.get("provider", "api")
    key = {"api": "api_model", "ollama": "ollama_model",
           "local": "local_model"}.get(provider, "api_model")
    name = os.path.basename(str(ep.get(key, "")))
    name = name.lower().replace("_", "-")
    return "glm-ocr" in name


def normalize_one_line(text):
    """仅用于已经确认是一个字幕行/列的自动识别、翻译结果。"""
    rows = [" ".join(row.split()) for row in _BREAKS.split(str(text or ""))]
    rows = [row for row in rows if row]
    if not rows:
        return ""
    if len(rows) == 1:
        return rows[0]
    if all(len(row) == 1 for row in rows):
        return "".join(rows)
    out = rows[0]
    for row in rows[1:]:
        boundary = out[-1:] + row[:1]
        separator = "" if _CJK.search(boundary) else " "
        out += separator + row
    return out


class BadOCR(ValueError):
    pass


def validate_ocr(text, max_chars=240, multiline=False, max_lines=4):
    """清洗/校验模型读字输出。multiline=True 时允许多行(检测器把多行合成一框的情况),
    以 \\n 连接返回; 否则多行视为异常(保持旧行为)。"""
    text = str(text or "").strip()
    if len(text) > 10000:
        raise BadOCR("输出异常过长")
    match = re.fullmatch(
        r"```(?:text|markdown|plaintext)?\s*\n(.*?)\n```",
        text, flags=re.S | re.I,
    )
    if match:
        text = match.group(1).strip()
    if "```" in text:
        raise BadOCR("输出包含内嵌或重复Markdown围栏")
    if re.search(r"<\|[^>]+\|>|</?think>", text, re.I):
        raise BadOCR("输出包含生成模板标记")
    rows = [x.strip() for x in _BREAKS.split(text) if x.strip()]
    limit = max_chars * (max_lines if multiline else 1)
    if len(text) > limit:
        raise BadOCR("输出超过该字幕区域的合理字符上限")
    compact = re.sub(r"\s+", "", text)
    if re.search(r"(.{2,24})\1{5,}", compact, re.S):
        raise BadOCR("检测到重复生成循环")
    if len(compact) > 80:
        grams = [compact[i:i + 4] for i in range(len(compact) - 3)]
        if len(set(grams)) / max(1, len(grams)) < 0.18:
            raise BadOCR("长输出重复度异常")
    if len(rows) > 1:
        if not multiline:
            raise BadOCR("读出了多行：需要在定位阶段获得独立行框")
        if len(rows) > max_lines:
            raise BadOCR(f"行数异常({len(rows)}>{max_lines})")
        return "\n".join(rows)
    return rows[0] if rows else ""
