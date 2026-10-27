# -*- coding: utf-8 -*-
"""
把不满足上传要求（宽>=700, 高>=32, 0.5<=宽/高<=32, <=2MB）的详情页图片修到合规。

两种模式：
  split  （默认，推荐）：在"接近纯色/背景"的水平行处把超高图切成多张。
                         宽度不变、内容零丢失、上下拼接后视觉连续。
                         合规的图原样拷贝（不重编码）。
  pad    ：把同目录所有图左右补边到统一宽度（= ceil(高度/2) 的最大需求值）。
                         图片张数不变，但整组内容会略微缩小、两侧有延伸色带。

用法：
  python fix_aspect_ratio.py --src "商详翻译_马来文" \
                             --out "商详翻译_马来文_合规" --mode split
  加 --dry-run 只出报告不动文件；--only "E1605" 只处理名字含该串的目录。
"""
import argparse
import csv
import io
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageFilter

MIN_W, MIN_H = 700, 32
RATIO_MIN, RATIO_MAX = 0.5, 32.0
MAX_BYTES = int(2.0 * 1024 * 1024)
SAFE = 0.98          # 切分后留 2% 余量：每片高 <= 2*w*SAFE
JPEG_Q = 95
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def natural_key(name):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def is_ok(w, h, size):
    return (w >= MIN_W and h >= MIN_H and RATIO_MIN <= w / h <= RATIO_MAX
            and size <= MAX_BYTES)


def flatness(gray):
    """每行的灰度标准差，越小越像纯色背景行。"""
    s = gray.std(axis=1)
    k = np.ones(3) / 3.0
    return np.convolve(s, k, mode="same")


def plan_cuts(im, cap, min_piece=140, search=260):
    """返回 [0, ..., h]，保证每片高度 <= cap 且 >= min_piece。"""
    w, h = im.size
    k = int(math.ceil(h / float(cap)))
    if k <= 1:
        return [0, h]
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    f = flatness(gray)

    cuts = [0]
    for i in range(1, k):
        prev = cuts[-1]
        remaining = k - i
        target = int(round(h * i / float(k)))
        lo = max(prev + min_piece, h - remaining * cap, target - search)
        hi = min(prev + cap, h - remaining * min_piece, target + search)
        if hi < lo:  # 极端尺寸回退
            lo = max(prev + min_piece, h - remaining * cap)
            hi = min(prev + cap, h - remaining * min_piece)
        hi = max(lo, hi)
        cuts.append(lo + int(np.argmin(f[lo:hi + 1])))
    cuts.append(h)
    return cuts


def save_jpeg(im, path):
    q = JPEG_Q
    while True:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, subsampling=0, optimize=True)
        if buf.tell() <= MAX_BYTES - 4096 or q <= 60:
            with open(path, "wb") as fh:
                fh.write(buf.getvalue())
            return buf.tell(), q
        q -= 5


def pad_to_width(im, target_w):
    w, h = im.size
    if w >= target_w:
        return im
    pad = target_w - w
    left = pad // 2
    right = pad - left
    canvas = Image.new("RGB", (target_w, h))

    def strip(pixels, width):
        s = im.crop((0 if pixels == "left" else w - 1, 0,
                     1 if pixels == "left" else w, h)).resize((width, h))
        r = max(2, width // 6)
        return s.filter(ImageFilter.GaussianBlur(r))

    if left:
        canvas.paste(strip("left", left), (0, 0))
    if right:
        canvas.paste(strip("right", right), (target_w - right, 0))
    canvas.paste(im.convert("RGB"), (left, 0))
    return canvas


def run(src_root, out_root, mode, only, dry):
    rows, n_copy, n_split, n_pad, n_skip = [], 0, 0, 0, 0
    done = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        files = sorted((f for f in filenames if f.lower().endswith(EXTS)),
                       key=natural_key)
        if not files:
            continue
        rel = os.path.relpath(dirpath, src_root)
        if only and only.lower() not in rel.lower():
            continue
        out_dir = os.path.join(out_root, rel)

        # pad 模式：先算出本目录统一目标宽度
        target_w = 0
        if mode == "pad":
            for f in files:
                with Image.open(os.path.join(dirpath, f)) as im:
                    w, h = im.size
                if w / h < RATIO_MIN:
                    target_w = max(target_w, int(math.ceil(h / RATIO_MIN * 1.03)))

        seq = 0
        folder_name = os.path.basename(os.path.normpath(dirpath))
        base = folder_name if folder_name not in (".", "") else os.path.splitext(files[0])[0]

        for f in files:
            path = os.path.join(dirpath, f)
            size = os.path.getsize(path)
            with Image.open(path) as im:
                im.load()
                w, h = im.size
                rgb = im.convert("RGB")

                if is_ok(w, h, size):
                    seq += 1
                    if not dry:
                        os.makedirs(out_dir, exist_ok=True)
                        ext = os.path.splitext(f)[1].lower()
                        if ext in (".jpeg",):
                            ext = ".jpg"
                        shutil.copy2(path, os.path.join(out_dir, "%s_%02d%s" % (base, seq, ext)))
                    n_copy += 1
                    continue

                if mode == "pad" and target_w > w and h >= MIN_H:
                    seq += 1
                    out = pad_to_width(rgb, target_w)
                    if not dry:
                        os.makedirs(out_dir, exist_ok=True)
                        save_jpeg(out, os.path.join(out_dir, "%s_%02d.jpg" % (base, seq)))
                    n_pad += 1
                    rows.append([rel, f, "%dx%d" % (w, h), "pad->%d" % target_w])
                    continue

                # split
                cap = int(2 * w * SAFE)
                if h <= cap:
                    seq += 1
                    if not dry:
                        os.makedirs(out_dir, exist_ok=True)
                        save_jpeg(rgb, os.path.join(out_dir, "%s_%02d.jpg" % (base, seq)))
                    n_skip += 1
                    rows.append([rel, f, "%dx%d" % (w, h), "recode-only"])
                    continue

                cuts = plan_cuts(rgb, cap)
                pieces = []
                for a, b in zip(cuts[:-1], cuts[1:]):
                    seq += 1
                    piece = rgb.crop((0, a, w, b))
                    pieces.append("%d-%d(%d)" % (a, b, b - a))
                    if not dry:
                        os.makedirs(out_dir, exist_ok=True)
                        save_jpeg(piece, os.path.join(out_dir, "%s_%02d.jpg" % (base, seq)))
                n_split += 1
                rows.append([rel, f, "%dx%d" % (w, h),
                             "split->%d pieces %s" % (len(pieces), " ".join(pieces))])

        if mode == "pad" and target_w:
            rows.append([rel, "-", "uniform width", str(target_w)])
        done += 1
        if done % 20 == 0:
            print("[%d] %s copy=%d split=%d pad=%d" % (done, rel, n_copy, n_split, n_pad), flush=True)

    return rows, (n_copy, n_split, n_pad, n_skip)


def verify(out_root):
    bad = []
    total = 0
    for dp, dn, fn in os.walk(out_root):
        for f in fn:
            if not f.lower().endswith(EXTS):
                continue
            p = os.path.join(dp, f)
            total += 1
            with Image.open(p) as im:
                w, h = im.size
            size = os.path.getsize(p)
            if not is_ok(w, h, size):
                bad.append((p, w, h, size, round(w / h, 4)))
    return total, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["split", "pad"], default="split")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows, (c, s, p, k) = run(a.src, a.out, a.mode, a.only, a.dry_run)
    print("copied=%d split=%d padded=%d recoded=%d" % (c, s, p, k))
    if not a.dry_run:
        rep = os.path.join(a.out, "_fix_report.csv")
        os.makedirs(a.out, exist_ok=True)
        with open(rep, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["folder", "file", "orig_size", "action"])
            wr.writerows(rows)
        total, bad = verify(a.out)
        print("verify: total=%d bad=%d" % (total, len(bad)))
        for b in bad[:20]:
            print("  BAD", b)
        print("report:", rep)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
