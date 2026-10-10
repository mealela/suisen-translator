# -*- coding: utf-8 -*-
"""LAMA 逐帧擦除子进程入口。

由主程序以独立 Python 运行时调用(发行版不内置 torch):
    单次: python lama_runner.py --frames <dir> --masks <dir> --out <dir>
    常驻: python lama_runner.py --serve <jobdir> [--batch N]
      循环处理 <jobdir>/task-*/(出现 ready 文件即开始), 结果写入
      task-*/out/frames/ 并写 done 标记; 出错写 error 文件; 出现
      <jobdir>/_stop 即退出。常驻模式让 torch 与模型在整个擦除过程
      只加载一次, 不再每块重启进程。

--batch N>1 时把同尺寸的帧按 N 个一批合成张量一次推理(降每帧开销;
GPU 被其他程序占用时收益不稳定, 主程序默认 1, 可在设置里开)。

读取 frames/masks 下同名 PNG 序列, 修复结果写入 <out>/frames/。
权重文件 big-lama.pt 与本脚本位于同一目录。
"""
import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image


def _load_model():
    weights = Path(__file__).resolve().parent / "big-lama.pt"
    if not weights.is_file():
        print(f"缺少权重文件: {weights}", file=sys.stderr)
        sys.exit(2)
    os.environ.setdefault("LAMA_MODEL", str(weights))
    from simple_lama_inpainting import SimpleLama
    return SimpleLama()


def _ceil_modulo(x, mod):
    return x if x % mod == 0 else (x // mod + 1) * mod


def _prep_batch(frame_paths, masks_dir, device):
    """按 prepare_img_and_mask 复刻: [0,1] 浮点 + symmetric 补边;
    一批内统一补到组内最大尺寸(8 的倍数), 合成 (N,3,H,W)/(N,1,H,W) 张量。返回 (imgs, masks, 原尺寸列表)。"""
    import numpy as np
    import torch
    imgs, msks, orig = [], [], []
    mh = mw = 0
    for fp in frame_paths:
        a = np.asarray(Image.open(fp).convert("RGB")).astype(np.float32) / 255.0
        a = np.transpose(a, (2, 0, 1))                       # (3,H,W)
        m = np.asarray(Image.open(Path(masks_dir) / fp.name).convert("L")).astype(np.float32) / 255.0
        orig.append(a.shape[1:])                             # (H, W)
        imgs.append(a)
        msks.append(m)
        mh = max(mh, a.shape[1])
        mw = max(mw, a.shape[2])
    mh, mw = _ceil_modulo(mh, 8), _ceil_modulo(mw, 8)
    ti = torch.zeros(len(imgs), 3, mh, mw)
    tm = torch.zeros(len(msks), 1, mh, mw)
    for i, (a, m) in enumerate(zip(imgs, msks)):
        h, w = a.shape[1], a.shape[2]
        a = np.pad(a, ((0, 0), (0, mh - h), (0, mw - w)), mode="symmetric")
        m = np.pad(m, ((0, mh - h), (0, mw - w)), mode="symmetric")
        ti[i] = torch.from_numpy(a)
        tm[i, 0] = torch.from_numpy((m > 0).astype(np.float32))
    return ti.to(device), tm.to(device), orig


def _forward_batch(lam, frame_paths, masks_dir, outdir):
    """同一尺寸的一批帧合成一个张量一次推理, 裁回各自原尺寸后逐帧落盘。"""
    import numpy as np
    import torch
    ti, tm, orig = _prep_batch(frame_paths, masks_dir, lam.device)
    with torch.inference_mode():
        out = lam.model(ti, tm).detach().cpu().numpy()       # (N,3,H,W)
    for i, fp in enumerate(frame_paths):
        h, w = orig[i]
        r = np.clip(out[i].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)[:h, :w]
        Image.fromarray(r).save(Path(outdir) / fp.name)


def _run_task(model, frames_dir, masks_dir, out_dir, batch=1):
    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        raise RuntimeError("帧目录为空")
    outdir = Path(out_dir) / "frames"
    outdir.mkdir(parents=True, exist_ok=True)
    if batch <= 1:
        for i, fp in enumerate(frames):
            mp = Path(masks_dir) / fp.name
            if not mp.is_file():
                raise RuntimeError(f"缺少遮罩: {mp.name}")
            img = Image.open(fp).convert("RGB")
            mask = Image.open(mp).convert("L")
            model(img, mask).save(outdir / fp.name)
            print(f"[lama] {i + 1}/{len(frames)}", flush=True)
        return
    # 批量: 同尺寸才能进同一 batch 张量; 组内按 batch 切批
    by_size = OrderedDict()
    for fp in frames:
        with Image.open(fp) as im:
            by_size.setdefault(im.size, []).append(fp)       # im.size = (W, H)
    done = 0
    for fps_ in by_size.values():
        for j in range(0, len(fps_), batch):
            _forward_batch(model, fps_[j:j + batch], masks_dir, outdir)
            done += len(fps_[j:j + batch])
            print(f"[lama] {done}/{len(frames)}", flush=True)


def serve(jobdir, batch=1):
    root = Path(jobdir)
    stop = root / "_stop"
    model = None
    print("[lama] serve 就绪", flush=True)
    while not stop.exists():
        ready = sorted(root.glob("task-*/ready"))
        if not ready:
            time.sleep(0.05)
            continue
        task = ready[0].parent
        try:
            if model is None:
                model = _load_model()
            _run_task(model, task / "frames", task / "masks", task / "out", batch=batch)
            (task / "done").write_text("ok", encoding="utf-8")
        except Exception as exc:              # 任务级失败不拖垮常驻进程
            (task / "error").write_text(str(exc), encoding="utf-8")
        try:
            (task / "ready").unlink()
        except OSError:
            pass
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames")
    parser.add_argument("--masks")
    parser.add_argument("--out")
    parser.add_argument("--serve")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if args.serve:
        sys.exit(serve(args.serve, batch=max(1, args.batch)))

    if not (args.frames and args.masks and args.out):
        parser.error("需要 --frames/--masks/--out 或 --serve")
    model = _load_model()
    try:
        _run_task(model, args.frames, args.masks, args.out, batch=max(1, args.batch))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
