"""把“双色胶囊标签”（左半浅色写“S小号”、右半深色写“30只装/盒”）里被 OCR 连成一行的文字拆成两行，
各自翻译、各自写回自己那一半。会备份 ocr.json 为 ocr.before_split.json（只备份一次）。

用法：python split_pills.py 分图根目录 工作根目录 [改动清单输出.txt] [--dry 只列出不改]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CJK = re.compile(r'[一-鿿]')


def split_x(arr, box):
    """框上下边附近的底色若是左右两段明显不同的纯色，返回分界 x；否则 None。"""
    x0, y0, x1, y1 = box
    h = y1 - y0
    if x1 - x0 < 2.5 * h:
        return None
    band = np.concatenate([arr[y0 + 1:y0 + 4, x0:x1], arr[y1 - 4:y1 - 1, x0:x1]]).astype(np.int16)
    col = np.median(band, axis=0)                           # 每列的底色
    n = len(col)
    best = None
    for s in range(int(0.2 * n), int(0.8 * n)):
        a, b = col[:s], col[s:]
        ma, mb = np.median(a, axis=0), np.median(b, axis=0)
        if np.abs(ma - mb).max() < 60:
            continue
        ua = (np.abs(a - ma).max(axis=1) < 19).mean()
        ub = (np.abs(b - mb).max(axis=1) < 19).mean()
        score = ua + ub
        if ua > 0.85 and ub > 0.85 and (best is None or score > best[0]):
            best = (score, s)
    return x0 + best[1] if best else None


def split_text(text, frac):
    """按字宽比例在 frac 处切文字，吸附到最近的“字母/汉字 ↔ 数字/空格”边界。"""
    t = text.strip()
    widths = [1.0 if CJK.match(c) else 0.55 for c in t]
    total = sum(widths)
    cuts = [k for k in range(1, len(t))
            if t[k - 1] == ' ' or t[k] == ' ' or (CJK.match(t[k - 1]) and not CJK.match(t[k]))
            or (not CJK.match(t[k - 1]) and CJK.match(t[k]))]
    if not cuts:
        return None
    acc = np.cumsum(widths) / total
    k = min(cuts, key=lambda c: abs(acc[c - 1] - frac))
    a, b = t[:k].strip(), t[k:].strip()
    # 双色胶囊是短标签：两段都短、切口不在标点处（长句压在两块颜色的照片上不算）
    if not a or not b or len(a) > 6 or len(b) > 10 or re.search(r'[，。、！？|,.!?;；：:“”]', a[-1] + b[0]):
        return None
    return a, b


def main():
    src_root, work_root = Path(sys.argv[1]), Path(sys.argv[2])
    rest = [a for a in sys.argv[3:] if a != '--dry']
    out = Path(rest[0]) if rest else None
    changed = []
    for ocr_p in sorted(work_root.glob('*/ocr.json')):
        prod = ocr_p.parent.name
        ocr = json.loads(ocr_p.read_text(encoding='utf-8'))
        dirty = False
        for name, lines in ocr.items():
            arr = None
            new = []
            for l in lines:
                xs = [p[0] for p in l['poly']]
                ys = [p[1] for p in l['poly']]
                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                parts = None
                if CJK.search(l['text']) and 14 <= box[3] - box[1] <= 80:
                    if arr is None:
                        arr = np.asarray(Image.open(src_root / prod / name).convert('RGB'))
                    sx = split_x(arr, box)
                    if sx:
                        parts = split_text(l['text'], (sx - box[0]) / (box[2] - box[0]))
                if parts:
                    (a, b), (x0, y0, x1, y1) = parts, box
                    new.append(dict(l, text=a, seg=True, poly=[[x0, y0], [sx - 2, y0], [sx - 2, y1], [x0, y1]]))
                    new.append(dict(l, text=b, seg=True, poly=[[sx + 2, y0], [x1, y0], [x1, y1], [sx + 2, y1]]))
                    changed.append((prod, name, l['text'], a, b))
                    dirty = True
                else:
                    new.append(l)
            ocr[name] = new
        if dirty and '--dry' not in sys.argv:
            bak = ocr_p.with_name('ocr.before_split.json')
            if not bak.exists():
                bak.write_text(ocr_p.read_text(encoding='utf-8'), encoding='utf-8')
            ocr_p.write_text(json.dumps(ocr, ensure_ascii=False, indent=1), encoding='utf-8')
    for c in changed:
        print('\t'.join(c))
    print(f'拆分 {len(changed)} 行', file=sys.stderr)
    if out:
        out.write_text('\n'.join(f'{p}\t{n}' for p, n in sorted({(c[0], c[1]) for c in changed})), encoding='utf-8')


if __name__ == '__main__':
    main()
