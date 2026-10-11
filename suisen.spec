# -*- mode: python ; coding: utf-8 -*-
# 打包: python -m pip install pyinstaller && pyinstaller suisen.spec
# 产出 dist\suisen\suisen.exe (onedir 单目录; 首次运行需联网下载模型)
from PyInstaller.utils.hooks import collect_all

datas = [("assets", "assets")]
binaries = []
hiddenimports = []
hiddenimports += ["mpv"]
hiddenimports += ["py7zr"]          # 新机器下载 libmpv 后解 7z 用(ensure_libmpv)

# 这几个包自带配置/数据文件, 必须收进发行目录
for pkg in ("rapidocr", "onnxruntime", "av"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# torch 系不进包: LAMA 由模型库在用户机器上安装独立运行时(见 model_manager._install_lama)
# simple_lama_inpainting 也必须排除: 它只有几十KB, 一旦被打进包,
# importlib.util.find_spec() 就会误判"进程内依赖已就绪", 导致
# 既跳过共享运行时安装, 又不会回退到 ExternalLamaEraser 子进程路径。
_EXCLUDE = ["paddle", "paddleocr", "paddlex",
            "simple_lama_inpainting",
            "torch", "torchvision", "torchgen", "functorch",
            "transformers", "bitsandbytes", "accelerate", "diffusers",
            "scipy", "triton", "sympy", "networkx", "mpmath"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDE,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="suisen",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="assets/logo.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="suisen",
)
