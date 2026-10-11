# 打包与发布说明

## 打包成 exe (PyInstaller, onedir)

```powershell
python -m pip install pyinstaller
pyinstaller suisen.spec
```

产出 `dist\suisen\suisen.exe`(单目录)。

要点:
- **onedir 而非 onefile**: onnxruntime 等体积大, onefile 启动极慢且杀软误报多。
- spec 里已 `collect_all rapidocr / onnxruntime / av`; 已 `excludes` paddle 系与 **torch 全家桶**。
- **torch 不进包**: LAMA 在发行版走独立运行时——模型库一键安装 python embeddable + torch + simple-lama-inpainting + big-lama 权重。
  torch 构建按用户机器选择: 有 NVIDIA 驱动装 `2.6.0+cu124`(约2.6GB), 否则 `2.6.0+cpu`(约200MB), 见 `model_manager._install_lama`;
  进程内能 import simple_lama_inpainting 时(开发环境)自动走进程内路径。
- **预览伴音用 QtMultimedia**(PySide6 自带, PyInstaller 的 PySide6 hook 会收集 multimedia 插件); 运行环境没有音频后端时自动静音, 不影响画面。
- 图标用 `assets/logo.ico`。
- **不要装到 Program Files**: 程序会在自身目录写 config.json / models / logs / .work。
- 首次运行需联网下载 RapidOCR 检测模型(约80MB)与所选本地模型(推荐组合约14GB, 模型库下载前会给出总体积), 在安装页写明。
- FFmpeg 按需下载(约110MB), 只在用户走到「原字处理/导出」时触发。

## FFmpeg 不要打进包里

FFmpeg 是 GPL 组件。当前实现: 程序在用户机器上按需自动下载(`core/inpaint.py: ensure_ffmpeg`)——
本地找不到 FFmpeg 时下载 gyan.dev 的 essentials 包(断点续传), 只解出 `bin/ffmpeg.exe` 到 `models/bin/`, zip 随即删除。
- 优先级: `config.json: ffmpeg_path`(设置→杂项) → `models/bin/ffmpeg.exe` → 数据根 `ffmpeg.exe` → PATH
- 若日后改为随包分发, 必须同时附 FFmpeg 的 LICENSE/GPL 文本与源码获取方式。

## 发布前检查

- [ ] 仓库里**没有** config.json(在 .gitignore)
- [ ] `git status` 里不出现 models/ logs/ .work/ ffmpeg.exe
- [ ] `installer.iss` 与 `config.APP_VERSION` 版本号一致(当前 1.0.0)
- [ ] `python -m unittest test_v6_core test_state_machines` 通过

## 依赖许可证速查

| 组件 | 许可证 | 说明 |
|---|---|---|
| PySide6(含 QtMultimedia) | LGPL-3.0 | 动态链接, 附 LICENSE 文本即可 |
| RapidOCR / onnxruntime | Apache-2.0 | 模型 ONNX 自带 PP-OCR 许可 |
| torch(CPU/CUDA 版) | BSD-3 | 不打包; 模型库安装 LAMA 时从 PyTorch 官方源下载 |
| simple-lama-inpainting / LAMA 权重 | Apache-2.0 | 权重由模型库下载 |
| OpenCV | Apache-2.0 | |
| FFmpeg | GPL/LGPL | 不打包, 用户机器上按需自动下载 |
| libmpv | GPL-2.0+ | **随安装包分发**(独立 DLL, 运行时动态加载, 新机器免下载); 已附 GPL-2.0 全文与源码链接(`THIRD-PARTY-NOTICES.txt` + `third_party_licenses/`, 安装时写入安装目录) |
| ProPainter | S-Lab 非商用 | **不入库**, 用户自行下载; 下载确认框已提示 |
| DiffuEraser | 看其仓库 | 同上 |
| GLM-OCR / Qwen3 / Qwen3-VL / Qwen3.8 / DeepSeek-R1 模型 | 各自许可 | 用户经 Ollama 自行拉取 |
