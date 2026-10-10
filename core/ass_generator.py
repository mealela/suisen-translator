import math
import re

from core.models import FIT_MODES, MaskEvent, SubtitleEvent, cover_quad, event_covered

HEADER = """[Script Info]
Title: Suisen Translator
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H7F000000,-1,0,0,0,100,100,0,0,1,2,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _t(sec: float) -> str:
    cs = max(0, int(math.floor(float(sec) * 100 + 0.5)))
    seconds, fraction = divmod(cs, 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _pt(s: str) -> float:
    h, m, rest = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _c(hex_color: str) -> str:
    hx = hex_color.lstrip("#").upper()
    if len(hx) != 6:
        hx = "FFFFFF"
    return f"&H00{hx[4:6]}{hx[2:4]}{hx[0:2]}&"


def _hex_from_ass(s: str) -> str:
    hx = re.sub(r"[^0-9A-Fa-f]", "", s)[-6:].upper().zfill(6)
    return f"#{hx[4:6]}{hx[2:4]}{hx[0:2]}"


def _esc(t):
    return t.replace("{", "(").replace("}", ")")


def _alpha_tag(alpha):
    a = int(round((1.0 - max(0.0, min(1.0, float(alpha)))) * 255))
    return f"\\1a&H{a:02X}&" if a else ""


def _poly_line(points, color, start, end, alpha=1.0):
    """任意四点框遮罩(绝对坐标), 与画面合成用的几何完全一致。"""
    pts = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in points]
    d = f"m {pts[0][0]} {pts[0][1]} l " + " ".join(f"{x} {y}" for x, y in pts[1:])
    tag = f"{{\\an7\\pos(0,0)\\1c{_c(color)}{_alpha_tag(alpha)}\\bord0\\shad0\\p1}}{d}{{\\p0}}"
    return f"Dialogue: 0,{_t(start)},{_t(end)},Default,,0,0,0,mask,{tag}"


def _style(ev, fs, bord, fscx=100.0, rotation=None):
    tag = (f"\\fs{int(fs)}\\c{_c(ev.color)}\\3c{_c(ev.outline_color)}"
           f"\\bord{bord:.1f}\\shad0\\b{1 if ev.bold else 0}")
    if ev.font.strip():
        tag += f"\\fn{ev.font.strip()}"
    if abs(fscx - 100.0) > 0.5:
        tag += f"\\fscx{fscx:.0f}"
    rot = ev.rotation if rotation is None else rotation
    if abs(rot) > 0.05:
        tag += f"\\frz{-rot:.1f}"
    return tag


def _effect(ev, vertical=False):
    """Effect 字段(播放器忽略)记录定位框与排布方式, 供 parse_ass 无损回读:
    q:x y ×4 = 四点框 / f:排布方式 / g:条目id(竖排多列合并) / v 竖排"""
    parts = []
    if vertical:
        parts.append("v")
    q = ev.geom if (ev.geom is not None and len(ev.geom) == 4) else None
    if q is not None:
        parts.append("q:" + " ".join(f"{float(v):.1f}" for p in q for v in p))
        parts.append("f:" + (ev.fit_mode if ev.fit_mode in FIT_MODES else "contain"))
    parts.append("g:" + ev.id)
    return ";".join(parts)


def generate_ass(events, masks, width, height, font="Microsoft YaHei",
                 include_masks=True, bilingual=False, t_off=0.0,
                 placement="erase", pipe_erase=True,
                 mask_style="solid", mask_alpha=1.0, layout=None, align="center"):
    """include_masks: 右键遮挡块永远写; 字幕自身遮罩按 event_covered 判定, 几何取 cover_quad。
    layout: ev → core.v6_paint.ass_layout(...) 的结果; 给出时字号/拉伸/对齐锚点/竖排多列与画面合成一致,
    不给则按原始字段输出(纯函数, 不依赖 Qt)。"""
    lines = [HEADER.format(w=width, h=height, font=font)]

    def paint(color):
        if mask_style in ("bar", "blur"):
            return "#000000", float(mask_alpha)
        return color, 1.0

    if include_masks:
        for m in masks or []:
            c, a = paint(m.color)
            lines.append(_poly_line([(m.x1, m.y1), (m.x2, m.y1), (m.x2, m.y2), (m.x1, m.y2)],
                                    c, m.start - t_off, m.end - t_off, a))
    for ev in sorted(events, key=lambda e: (e.start, e.y)):
        text = ev.display_text()
        if not text:
            continue
        st, en = ev.start - t_off, ev.end - t_off
        if en <= 0:
            continue
        st = max(0.0, st)
        if include_masks and event_covered(ev, placement, pipe_erase):
            c, a = paint(ev.mask_color)
            lines.append(_poly_line(cover_quad(ev), c, st, en, a))

        lay = layout(ev) if layout else None
        if lay:
            fs = lay["fs"]
            sx, sy = lay["scale"]
            fscx = 100.0 * sx / sy if sy else 100.0
            bord = ev.outline_w() * (sy or 1.0)
            eff = _effect(ev, lay["vertical"])
            if lay["vertical"]:
                for cx, cy, column, _scale in lay["items"]:
                    body = "\\N".join(_esc(ch) for ch in column if not ch.isspace())
                    tag = "{" + f"\\an5\\pos({cx:.0f},{cy:.0f})" + _style(ev, fs, bord, fscx, lay["rotation"]) + "}"
                    lines.append(f"Dialogue: 1,{_t(st)},{_t(en)},Default,,0,0,0,{eff},{tag}{body}")
            else:
                rows = []
                for _, _, row, scale in lay["items"]:
                    body = _esc(" ".join(row.split()))
                    if abs(scale - 1.0) > 1e-3:
                        body = r"{\fs%d}" % max(10, int(round(fs * scale))) + body
                    rows.append(body)
                cx, cy = lay["center"]
                an = int(lay.get("an", 5))
                tag = "{" + f"\\an{an}\\pos({cx:.0f},{cy:.0f})" + _style(ev, fs, bord, fscx, lay["rotation"]) + "}"
                lines.append(f"Dialogue: 1,{_t(st)},{_t(en)},Default,,0,0,0,{eff},{tag}" + "\\N".join(rows))
            continue

        # 无排布回调: 按原始字段输出(居中锚点)
        if ev.vertical:
            body = "\\N".join(c for c in _esc(text) if c != " ")
        else:
            body = "\\N".join(_esc(line) for line in text.splitlines() if line.strip())
            if bilingual and ev.translated.strip() and ev.text.strip():
                small = max(10, int(ev.font_size * 0.55))
                body += r"\N{\fs%d}" % small + _esc(" ".join(ev.text.split()))
        tag = ("{" + f"\\an5\\pos({int(round(ev.x))},{int(round(ev.y))})"
               + _style(ev, ev.font_size, ev.outline_w()) + "}")
        lines.append(f"Dialogue: 1,{_t(st)},{_t(en)},Default,,0,0,0,{_effect(ev, ev.vertical)},{tag}{body}")
    return "\n".join(lines) + "\n"


def generate_srt(events, bilingual=False):
    def stamp(value):
        milliseconds = max(0, int(math.floor(float(value) * 1000 + 0.5)))
        seconds, fraction = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{fraction:03d}"

    output = []
    for ev in sorted(events, key=lambda e: (e.start, e.y)):
        text = ev.display_text()
        if not text or ev.end <= ev.start:
            continue
        if bilingual and ev.translated.strip() and ev.text.strip():
            text += "\n" + ev.text.strip()
        output.append(f"{len(output) + 1}\n{stamp(ev.start)} --> {stamp(ev.end)}\n{text}\n")
    return "\n".join(output) + "\n"


def _parse_styles(content):
    """解析 [V4+ Styles] 段 → {样式名: {align, ml, mr, mv, fs, font, bold, outline, color, ocolor}}。
    标准 ASS 字幕(无 \\pos)靠样式的对齐+边距定位, 导入时要按样式还原位置。"""
    styles, fmt, in_styles = {}, None, False
    for line in content.splitlines():
        ls = line.strip()
        if ls.startswith("["):
            low = ls.lower()
            in_styles = low.startswith("[v4+ styles") or low.startswith("[v4 styles")
            continue
        if not in_styles:
            continue
        if ls.lower().startswith("format:"):
            fmt = [f.strip().lower() for f in ls.split(":", 1)[1].split(",")]
            continue
        if ls.lower().startswith("style:") and fmt:
            vals = [v.strip() for v in ls.split(":", 1)[1].split(",")]
            if len(vals) < len(fmt):
                continue
            d = dict(zip(fmt, vals))

            def num(k, default=0.0):
                try:
                    return float(d.get(k, default))
                except (ValueError, TypeError):
                    return default
            styles[d.get("name", "")] = {
                "align": int(num("alignment", 2)), "ml": num("marginl"), "mr": num("marginr"),
                "mv": num("marginv"), "fs": num("fontsize", 40), "font": d.get("fontname", ""),
                "bold": d.get("bold", "0") in ("1", "-1"), "outline": num("outline"),
                "color": d.get("primarycolour", ""), "ocolor": d.get("outlinecolour", ""),
            }
    return styles


def parse_ass(content: str, video_w=0, video_h=0, fs_ratio=1.0):
    """导入 ASS。坐标/字号/描边相对 PlayResX/Y, 与视频分辩率不一致时按比例换算;
    fs_ratio: \\fs(格高) → 本软件字号(em 高)的换算比例, GUI 传 v6_paint.ass_font_ratio()。
    本软件导出的 ASS 在 Effect 里带定位框(q:)与排布方式(f:): 回读后定位框、字号自适应与导出前完全一致;
    竖排多列按 g: 合并回一条。"""
    rx = ry = 0.0
    for line in content.splitlines():
        ls = line.strip()
        if ls.startswith("PlayResX:"):
            try:
                rx = float(ls.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif ls.startswith("PlayResY:"):
            try:
                ry = float(ls.split(":", 1)[1].strip())
            except ValueError:
                pass
    if rx <= 0:                     # ASS 规范: PlayResX/Y 省略时默认 384×288
        rx = 384.0
    if ry <= 0:
        ry = 288.0
    sx = (video_w / rx) if video_w else 1.0
    sy = (video_h / ry) if video_h else 1.0
    ratio = float(fs_ratio) if fs_ratio and fs_ratio > 0 else 1.0
    styles = _parse_styles(content)
    events, masks, groups = [], [], {}
    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        try:
            fields = line[len("Dialogue:"):].split(",", 9)
            if len(fields) < 10:
                continue
            start, end = _pt(fields[1]), _pt(fields[2])
            style = dict(styles.get(fields[3].strip(), {}))
            parts = [p.strip() for p in fields[8].split(";") if p.strip()]
            body = fields[9]
            tags = "".join(re.findall(r"\{([^}]*)\}", body))
            plain = re.sub(r"\{[^}]*\}", "", body)
            # ASS 规则: 行内 MarginL/R/V(第5,6,7字段) 非零时覆盖样式边距; 行内 \an 覆盖样式对齐
            for fi, key in ((5, "ml"), (6, "mr"), (7, "mv")):
                try:
                    v = float(fields[fi])
                    if v:
                        style[key] = v
                except (ValueError, IndexError):
                    pass
            man = re.search(r"\\an(\d)", tags)
            if man:
                style["align"] = int(man.group(1))
            mpos = re.search(r"\\pos\(([\-\d.]+),([\-\d.]+)\)", tags)
            if mpos:
                ax = float(mpos.group(1)) * sx
                ay = float(mpos.group(2)) * sy
                # \pos 是按 \an 对齐的锚点(不是文字中心); 换算成 manual 布局用的"文字中心"
                fs_eff = float(style.get("fs", 40.0) or 40.0) * sy
                lines = [x for x in re.split(r"\\N|\r?\n", plain) if x.strip()] or [plain]
                wmax = max((_text_w(x, fs_eff) for x in lines), default=0.0)
                align = int(style.get("align", 2) or 2)
                col = (align - 1) % 3
                row = (align - 1) // 3
                if col == 0:
                    ax += wmax / 2.0
                elif col == 2:
                    ax -= wmax / 2.0
                half = fs_eff * 0.6 * max(1, len(lines))
                if row == 0:
                    ay -= half
                elif row == 2:
                    ay += half
                px, py = ax, ay
            else:
                px, py = _style_anchor(style, rx, ry, sx, sy)
            if "mask" in parts or "\\p1" in tags:
                mc = re.search(r"\\1c(&H[0-9A-Fa-f]+&)", tags)
                color = _hex_from_ass(mc.group(1)) if mc else "#101010"
                md = re.search(r"m\s+0\s+0\s+l\s+([\d.]+)\s+0\s+[\d.]+\s+([\d.]+)", body)
                if md and "\\pos(0,0)" not in tags:
                    w, h = float(md.group(1)) * sx, float(md.group(2)) * sy
                    masks.append(MaskEvent(start, end, px, py, px + w, py + h, color=color))
                    continue
                mp = re.search(
                    r"m\s+([\-\d.]+)\s+([\-\d.]+)\s+l\s+([\-\d.]+)\s+([\-\d.]+)\s+"
                    r"([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)", body)
                if not mp:
                    continue
                v = [float(mp.group(i)) for i in range(1, 9)]
                qx = [v[i] * sx for i in (0, 2, 4, 6)]
                qy = [v[i] * sy for i in (1, 3, 5, 7)]
                masks.append(MaskEvent(start, end, px + min(qx), py + min(qy),
                                       px + max(qx), py + max(qy), color=color))
                continue
            vertical = "v" in parts
            quad = fit = group = None
            for p in parts:
                if p.startswith("q:"):
                    vals = [float(x) for x in p[2:].split()]
                    if len(vals) == 8:
                        quad = [[vals[i] * sx, vals[i + 1] * sy] for i in range(0, 8, 2)]
                elif p.startswith("f:"):
                    fit = p[2:]
                elif p.startswith("g:"):
                    group = p[2:]
            text = plain.replace("\\N", "" if vertical else "\n").strip()
            if not text:
                continue
            if group and group in groups:                 # 竖排第 2..n 列: 并回同一条
                prev = groups[group]
                prev.text += "\n" + text
                prev.end = max(prev.end, end)
                continue

            def g(pat, default):
                m = re.search(pat, tags)
                return m.group(1) if m else default
            ev = SubtitleEvent(
                start=start, end=end, text=text, x=px, y=py,
                font_size=max(1, int(float(g(r"\\fs([\d.]+)", float(style.get("fs", 40.0) or 40.0))) * sy / ratio)),
                rotation=-float(g(r"\\frz([\-\d.]+)", 0.0)),
                bold=g(r"\\b(\d)", "1" if style.get("bold", True) else "0") == "1",
                outline_width=float(g(r"\\bord([\d.]+)", float(style.get("outline", 0.0) or 0.0))) * sy,
                font=(g(r"\\fn([^\\}]+)", "") or str(style.get("font", ""))).strip(), vertical=vertical)
            mc = re.search(r"\\1?c(&H[0-9A-Fa-f]+&)", tags)
            if mc:
                ev.color = _hex_from_ass(mc.group(1))
            elif style.get("color"):
                try:
                    ev.color = _hex_from_ass(style["color"])
                except Exception:
                    pass
            m3 = re.search(r"\\3c(&H[0-9A-Fa-f]+&)", tags)
            if m3:
                ev.outline_color = _hex_from_ass(m3.group(1))
            elif style.get("ocolor"):
                try:
                    ev.outline_color = _hex_from_ass(style["ocolor"])
                except Exception:
                    pass
            if quad is not None:
                ev.set_quad(quad)                         # 定位框回来了 → 字号按框自适应, 与导出前一致
                ev.fit_mode = fit if fit in FIT_MODES else "contain"
                ev.geom_anchored = True
            if group:
                groups[group] = ev
            events.append(ev)
        except Exception:
            continue
    return events, masks


def _text_w(text, fs):
    """估算文字宽度(像素): 全角字符≈fs, 半角字符≈0.55fs; 用于把 \an 锚点换算成文字中心。"""
    w = 0.0
    for ch in str(text):
        w += 1.0 if ord(ch) >= 0x2E80 else 0.55
    return w * fs


def _style_anchor(style, rx, ry, sx, sy):
    """无 \\pos 的样式定位: 按 ASS 对齐(小键盘 1-9) + 边距算出锚点, 再换算成我们 manual 布局用的"文字中心"。
    默认(无样式信息)按底部居中(标准字幕位置)。"""
    align = int(style.get("align", 2) or 2)
    ml = float(style.get("ml", 0.0) or 0.0)
    mr = float(style.get("mr", 0.0) or 0.0)
    mv = float(style.get("mv", 0.0) or 0.0)
    fs = float(style.get("fs", 40.0) or 40.0)
    col = (align - 1) % 3          # 0=左, 1=居中, 2=右
    row = (align - 1) // 3         # 0=底, 1=中, 2=顶
    if col == 0:
        x = ml
    elif col == 1:
        x = rx / 2.0
    else:
        x = rx - mr
    half = fs * 0.6                # 锚点是行底/顶, 中心向文字方向偏移半行高
    if row == 0:
        y = ry - mv - half         # 底对齐: 中心在锚点上方
    elif row == 1:
        y = ry / 2.0
    else:
        y = mv + half              # 顶对齐: 中心在锚点下方
    return x * sx, y * sy
