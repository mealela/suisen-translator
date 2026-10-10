"""字体处理: 自动回退(防豆腐块) 与 烧录前按需下载。"""
_FALLBACKS = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
              "PingFang SC", "SimHei", "Yu Gothic", "Malgun Gothic", "Arial"]

NOTO_FAMILY = "Noto Sans CJK SC"
_NOTO_URL = ("https://raw.githubusercontent.com/notofonts/noto-cjk/"
             "main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-{style}.otf")


def ensure_font(family, progress=lambda p, m: None, abort=None):
    """烧录/合成前调用: 所选字体支持按需下载时, 缺失则现场下载并注册。
    目前支持 Noto Sans CJK SC; 其他字体依赖系统已安装。"""
    if str(family or "").strip() != NOTO_FAMILY:
        return
    try:
        from core.paths import base_path
        folder = base_path() / "models" / "fonts"
        need = [(s, folder / f"NotoSansCJKsc-{s}.otf")
                for s in ("Regular", "Bold")
                if not (folder / f"NotoSansCJKsc-{s}.otf").is_file()]
        if need:
            from core.model_manager import download
            for style, path in need:
                progress(0, f"下载字体 {path.name}(约33MB)…")
                download(_NOTO_URL.format(style=style), path, progress, abort)
        from PySide6.QtGui import QFontDatabase
        for path in folder.glob("*.otf"):
            QFontDatabase.addApplicationFont(str(path))
    except Exception:
        pass                                        # 字体获取失败时走回退族


def pick_font(family, target_lang):
    try:
        from PySide6.QtGui import QFontDatabase
        ws_map = {"中文(简体)": QFontDatabase.WritingSystem.SimplifiedChinese,
                  "中文(繁体)": QFontDatabase.WritingSystem.TraditionalChinese,
                  "日本語": QFontDatabase.WritingSystem.Japanese,
                  "한국어": QFontDatabase.WritingSystem.Korean}
        ws = None
        for k, v in ws_map.items():
            if target_lang == k:
                ws = v
                break
        fams = set(QFontDatabase.families())

        def ok(f):
            if f not in fams:
                return False
            if ws is None:
                return True
            try:
                return ws in QFontDatabase.writingSystems(f)
            except Exception:
                return True

        if ok(family):
            return family
        for f in _FALLBACKS:
            if ok(f):
                return f
    except Exception:
        pass
    return family or "Microsoft YaHei"
