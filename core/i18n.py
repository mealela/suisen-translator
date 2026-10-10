"""界面文案层占位: 1.1 起只提供中文界面(英文翻译覆盖不全, 设置项已撤下)。
tr() 原样返回, 保留调用点以便日后补全完整译文后恢复选项。"""

def set_lang(lang):
    return None


def current():
    return "zh"


def tr(text):
    return text
