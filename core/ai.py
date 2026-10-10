import base64
import json
import logging
import os
import re
import threading
import time
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests

from core.v6_common import fmt_mmss

from config import MANAGED_OLLAMA

log = logging.getLogger("suisen.ai")


class AIError(Exception):
    pass


_LLAMA_CACHE = {}
_RETRY_STATUS = {429, 500, 502, 503, 504}

# ---- 接入预设: 设置页 / 向导 / 端点编辑器 共用的唯一数据源(只用官方 API, 不含第三方聚合) ----
API_PRESETS_TEXT = [
    ("DeepSeek 官方", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("阿里百炼 · Qwen-Flash", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-flash"),
    ("Google · Gemini 2.5-Flash", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"),
    ("智谱 GLM-4-Flash", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("Moonshot Kimi", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("本地 OpenAI 兼容服务(LM Studio 等)", "http://127.0.0.1:1234/v1", ""),
]
API_PRESETS_VLM = [
    ("阿里百炼 · Qwen3.8-Max(旗舰多模态)", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.8-max"),
    ("阿里百炼 · Qwen-VL-Plus(多模态读字)", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-plus"),
    ("Google · Gemini 2.5-Flash-Lite(读字)", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash-lite"),
    ("Google · Gemini 2.5-Pro(读字)", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-pro"),
    ("本地 vLLM / LM Studio", "http://127.0.0.1:1234/v1", ""),
]


def api_presets(vision):
    return API_PRESETS_VLM if vision else API_PRESETS_TEXT


def local_endpoint(tag):
    return {"provider": "ollama", "ollama_url": MANAGED_OLLAMA, "ollama_model": tag}


def _no_think(model):
    m = str(model or "").lower()
    return "qwen3" in m or "deepseek-r1" in m or "deepseek-v3" in m


def friendly_http_error(status, body=""):
    body = (body or "")[:160]
    if status == 401:
        return "API Key 无效或未授权(401)。请到[设置]检查 Key 是否正确/过期。"
    if status == 403:
        return "无权限或余额不足(403)。请检查账号套餐/余额。"
    if status == 404:
        return ("接口路径或模型名不存在(404)。确认 Base URL 结尾(通常为 /v1), 并用[获取模型]校验模型名。" + body)
    if status == 429:
        return "请求过于频繁被限流(429), 已自动重试仍失败。稍后再试或调小批量条数。"
    if status >= 500:
        return f"服务端错误({status}), 已自动重试仍失败, 稍后再试。{body}"
    return f"API 请求失败({status}): {body}"


def _post_json(url, headers, payload, timeout, tag="API", abort=None):
    """请求在辅助线程里发出, 主线程每 0.2s 看一次取消标志: 本地模型慢慢生成时点[取消]能立刻返回。
    timeout 为读超时(秒), 不再有 90s 硬上限(CPU 跑本地大模型一批可能要几分钟)。"""
    timeout = max(30.0, float(timeout))

    def cancelled():
        return bool(abort and abort())

    for attempt in range(3):
        if cancelled():
            raise AIError("已取消")
        box = {}

        def work():
            try:
                with requests.post(url, headers=headers, json=payload, timeout=(10, timeout)) as response:
                    box["status"] = response.status_code
                    box["text"] = response.text
                    if response.status_code == 200:
                        box["data"] = response.json()
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(0.2)
            if cancelled():
                raise AIError("已取消")
        if "error" in box:
            exc = box["error"]
            if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
                if attempt == 2:
                    raise AIError(f"[{tag}] 网络超时或连接失败: {exc}") from exc
                continue
            raise AIError(f"[{tag}] 请求异常: {exc}") from exc
        status = box.get("status", 0)
        if status in _RETRY_STATUS and attempt < 2:
            for _ in range(10 * (2 ** attempt)):
                if cancelled():
                    raise AIError("已取消")
                time.sleep(0.1)
            continue
        if status != 200:
            raise AIError(f"[{tag}] " + friendly_http_error(status, box.get("text", "")))
        return box["data"]
    raise AIError(f"[{tag}] 请求失败")


def clean_think(raw: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()


def coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v or "").strip().lower() in ("true", "1", "yes", "y", "竖排", "是")


def coerce_num(v, default=None):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        return float(m.group()) if m else default
    except Exception:
        return default


def extract_json(raw):
    text = clean_think(raw)
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not starts:
        raise AIError("模型未返回 JSON 对象或数组")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError as exc:
        raise AIError(f"模型返回的 JSON 无效或被截断: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise AIError("模型返回的 JSON 根节点不是对象或数组")
    return value


def b64_jpeg(frame, max_w=960, quality=80):
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, max(2, int(h * max_w / w))))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise AIError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def ep_summary(ep):
    p = ep.get("provider", "api")
    if p == "local":
        return "本地·" + (os.path.basename(ep.get("local_model", "")) or "未选模型")
    if p == "ollama":
        return "Ollama·" + (ep.get("ollama_model") or "?")
    return "API·" + (ep.get("api_model") or "?")


def _conn_error_hint(e):
    """把底层网络错误翻译成可读提示; SSL 证书校验失败(常见于杀毒软件/代理拦截 HTTPS)单独说明。"""
    s = str(e)
    if "CERTIFICATE_VERIFY_FAILED" in s or "SSLCertVerificationError" in s or "certificate verify failed" in s:
        return ("SSL 证书校验失败: 多半是杀毒软件(如卡巴斯基)或公司代理拦截了 HTTPS。\n"
                "本程序已默认信任 Windows 系统证书库; 若仍失败, 请在杀毒软件里把该 API 域名加入"
                "[不扫描/排除]列表, 或暂时关闭其「HTTPS/加密连接扫描」。\n原始错误: " + s)
    return s


def list_models(ep, timeout=10):
    ep = resolved_endpoint(ep)
    from core.model_manager import ensure_ollama, is_managed
    if is_managed(ep):
        ensure_ollama()
    p = ep.get("provider")
    if p == "ollama":
        url = (ep.get("ollama_url") or "http://localhost:11434").rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            raise AIError(friendly_http_error(r.status_code, r.text))
        return sorted(m.get("name", "") for m in r.json().get("models", []))
    if p == "api":
        base = (ep.get("api_base") or "").rstrip("/")
        if not base:
            raise AIError("请先填写 Base URL")
        headers = {}
        if ep.get("api_key", "").strip():
            headers["Authorization"] = f"Bearer {ep['api_key'].strip()}"
        try:
            r = requests.get(base + "/models", headers=headers, timeout=timeout)
            if r.status_code == 200:
                return sorted(str(m.get("id", "")) for m in r.json().get("data", []))
            err = friendly_http_error(r.status_code, r.text)
        except Exception as e:
            err = _conn_error_hint(e)
        try:
            root = base[:-3] if base.endswith("/v1") else base
            r2 = requests.get(root + "/api/v0/models", headers=headers, timeout=timeout)
            if r2.status_code == 200:
                return sorted(str(m.get("id", "")) for m in r2.json().get("data", []))
        except Exception:
            pass
        raise AIError(err)
    raise AIError("本地模型请直接选择 GGUF 文件")


def _load_llama(model_path, n_ctx):
    key = (model_path, int(n_ctx))
    if key in _LLAMA_CACHE:
        return _LLAMA_CACHE[key]
    if not model_path or not os.path.isfile(model_path):
        raise AIError("未找到 GGUF 模型文件, 请在设置中选择")
    try:
        from llama_cpp import Llama
    except ImportError:
        raise AIError("未安装 llama-cpp-python: pip install llama-cpp-python")
    llm = Llama(model_path=model_path, n_ctx=max(2048, int(n_ctx)), n_gpu_layers=-1, verbose=False)
    _LLAMA_CACHE[key] = llm
    return llm


class TextClient:
    def __init__(self, ep, abort=None, timeout=300):
        self.abort = abort
        self.timeout = float(timeout or 300)
        self.ep = resolved_endpoint(ep)
        self.p = self.ep.get("provider", "api")
        from core.model_manager import prepare_endpoint
        prepare_endpoint(self.ep)
        if self.p == "local":
            self.llm = _load_llama(self.ep.get("local_model", ""), int(self.ep.get("local_ctx", 4096)))
        elif self.p not in ("api", "ollama"):
            raise AIError(f"未知接入类型: {self.p}")

    def chat(self, system, user, temperature=0.2, max_tokens=1024):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.p == "local":
            return _finish_text(self.llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens))
        if self.p == "ollama":
            model = self.ep.get("ollama_model", "")
            options = dict(self.ep.get("ollama_options") or {})
            options.update(temperature=temperature, num_predict=max_tokens,
                           num_ctx=int(self.ep.get("local_ctx", 4096)))
            payload = {"model": model, "messages": messages, "stream": False, "options": options}
            if _no_think(model):
                payload["think"] = False
            return _finish_text(_post_json(
                self.ep.get("ollama_url", "http://localhost:11434").rstrip("/") + "/api/chat",
                {}, payload, self.timeout, "Ollama翻译", abort=self.abort), True)
        key = str(self.ep.get("api_key") or "").strip()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        return _finish_text(_post_json(
            self.ep.get("api_base", "").rstrip("/") + "/chat/completions", headers,
            _api_payload(self.ep, messages, temperature, max_tokens),
            min(self.timeout, 180.0), "翻译API", abort=self.abort))


class VLMClient:
    def __init__(self, ep, detail_width=960, abort=None, timeout=300):
        self.abort = abort
        self.timeout = float(timeout or 300)
        self.ep = resolved_endpoint(ep)
        self.p = self.ep.get("provider", "api")
        self.width = max(320, int(detail_width))
        from core.model_manager import prepare_endpoint
        prepare_endpoint(self.ep)
        if self.p == "local":
            raise AIError("本地 GGUF 视觉模型不再支持; 请在模型库启用托管 Ollama 模型, 或使用视觉 API。")
        if self.p not in ("api", "ollama"):
            raise AIError(f"未知视觉接入类型: {self.p}")

    @staticmethod
    def _msgs(b64, prompt, system):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]})
        return msgs

    def ask(self, frame_bgr, prompt, system="", max_tokens=192):
        from core.ocr_protocols import is_glm_ocr
        b64 = b64_jpeg(frame_bgr, self.width, quality=95)
        if self.p == "ollama":
            options = dict(self.ep.get("ollama_options") or {})
            options.update(temperature=0.0, num_predict=int(max_tokens),
                           num_ctx=max(4096, int(self.ep.get("local_ctx", 8192))))
            base = self.ep.get("ollama_url", "http://localhost:11434").rstrip("/")
            model = self.ep.get("ollama_model", "")
            if is_glm_ocr(self.ep):
                payload = {"model": model, "prompt": "Text Recognition:", "images": [b64],
                           "stream": False, "options": options}
                data = _post_json(base + "/api/generate", {}, payload, self.timeout, "GLM-OCR", abort=self.abort)
            else:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt, "images": [b64]})
                payload = {"model": model, "messages": messages, "stream": False, "options": options}
                if _no_think(model):
                    payload["think"] = False
                data = _post_json(base + "/api/chat", {}, payload, self.timeout, "视觉读字", abort=self.abort)
            return _finish_text(data, True)
        key = str(self.ep.get("api_key") or "").strip()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        return _finish_text(_post_json(
            self.ep.get("api_base", "").rstrip("/") + "/chat/completions", headers,
            _api_payload(self.ep, self._msgs(b64, prompt, system), 0.0, max_tokens),
            min(self.timeout, 180.0), "视觉API", abort=self.abort))


def test_endpoint(ep, vision=False, detail_width=640, abort=None, timeout=120):
    import secrets
    started = time.monotonic()
    if vision:
        token = "T" + secrets.token_hex(2).upper()
        image = np.full((110, 360, 3), 245, np.uint8)
        cv2.putText(image, token, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 10, 10), 3)
        result = clean_think(VLMClient(ep, detail_width, abort=abort, timeout=timeout).ask(
            image, "只输出图片中的验证码, 不要解释。", max_tokens=40))
        if token not in result:
            raise AIError("接口有响应, 但未正确读出测试图验证码; 不能判定视觉测试通过。")
    else:
        result = TextClient(ep, abort=abort, timeout=timeout).chat(
            "Translate the user's text into Chinese.", "Hello.", temperature=0, max_tokens=80)
        if not clean_think(result).strip():
            raise AIError("接口返回空文本")
    return f"测试通过, 用时 {fmt_mmss(time.monotonic() - started)}"


# ---------------- 翻译(带上下文) ----------------

def _ctx_block(ctx_pairs):
    if not ctx_pairs:
        return ""
    seg = " / ".join(d[:40] for _, d in ctx_pairs[-2:])
    return "[上文译文, 仅供保持人称语气衔接, 禁止输出或复述]: " + seg + "\n"


_ARROW_SPLIT_RE = re.compile(r"\s*(?:→|->|=>)\s*")
_PREFIX_RE = re.compile(r"^\s*(?:译文|翻译|中文|结果|输出|translation)\s*[:：]\s*", re.I)


def _norm_cmp(s):
    return re.sub(r"[\s\W_]+", "", s or "").lower()


def clean_translation(src, out):
    t = clean_think(out or "").strip().strip('"“”「」').strip()
    t = _PREFIX_RE.sub("", t)
    if _ARROW_SPLIT_RE.search(t):
        parts = _ARROW_SPLIT_RE.split(t)
        if len(parts) >= 2:
            left, sn = _norm_cmp(parts[0]), _norm_cmp(src)
            if left and sn and (left in sn or sn in left
                                or SequenceMatcher(None, left, sn).ratio() > 0.6):
                t = parts[-1].strip()
    if _norm_cmp(t) == _norm_cmp(src) and re.search(r"[ぁ-んァ-ヶー]", src or ""):
        return ""
    if not re.search(r"\\N|\r?\n", src or ""):
        from core.ocr_protocols import normalize_one_line
        t = normalize_one_line(t)
    return t.strip().strip('"').strip()


def translate_one(client, text, target, sys_tpl, ctx_pairs=None):
    system = sys_tpl.replace("{target}", target)
    user = (_ctx_block(ctx_pairs) + "请翻译下一句(只输出这一句的译文):\n" + text) if ctx_pairs else text
    return clean_translation(text, client.chat(system, user))


def translate_batch(client, texts, target, sys_tpl, batch_tpl, ctx_pairs=None):
    if len(texts) == 1:
        return [translate_one(client, texts[0], target, sys_tpl, ctx_pairs)]
    user = (batch_tpl.replace("{target}", target)
            .replace("{context}", _ctx_block(ctx_pairs))
            .replace("{payload}", json.dumps(texts, ensure_ascii=False)))
    system = sys_tpl.replace("{target}", target) + "\n本次输入为数组, 严格输出等长JSON字符串数组。"
    arr = extract_json(client.chat(system, user, max_tokens=min(4096, 256 * len(texts))))
    if not isinstance(arr, list) or len(arr) != len(texts) or not all(isinstance(x, str) for x in arr):
        raise AIError("批量翻译未返回等长字符串数组")
    return [clean_translation(s, x) for s, x in zip(texts, arr)]


def resolved_endpoint(ep):
    from config import resolve_secret
    ep = dict(ep or {})
    ep["api_key"] = resolve_secret(ep.get("api_key"))
    return ep


def _finish_text(data, ollama=False):
    if ollama:
        reason = str(data.get("done_reason") or "")
        if data.get("done") is False or reason in ("length", "limit"):
            raise AIError("模型输出被截断或未完成")
        return (data.get("response") if "response" in data
                else data.get("message", {}).get("content", "")) or ""
    choice = data["choices"][0]
    if choice.get("finish_reason") in ("length", "content_filter"):
        raise AIError("模型输出被截断或被服务端过滤")
    content = choice.get("message", {}).get("content", "")
    if not isinstance(content, str):
        raise AIError("服务端没有返回文本 content")
    return content


def _api_payload(ep, messages, temperature, max_tokens):
    payload = {"model": ep.get("api_model", ""), "messages": messages,
               "temperature": temperature, "max_tokens": int(max_tokens), "stream": False}
    base = str(ep.get("api_base") or "")
    if "api.deepseek.com" in base:
        payload["thinking"] = {"type": "disabled"}
    if "dashscope" in base:
        payload["enable_thinking"] = False
    extra = ep.get("api_extra") or {}
    if not isinstance(extra, dict):
        raise AIError("api_extra 必须是 JSON 对象")
    for key, value in extra.items():
        if key not in ("messages", "model", "stream"):
            payload[key] = value
    return payload
