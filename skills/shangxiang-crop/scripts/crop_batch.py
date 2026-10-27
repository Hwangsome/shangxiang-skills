"""把一个目录下的商详图按版块边界批量裁成完整的分图（不补边、不缩放），每张商详图一个子目录。

用法：python crop_batch.py 源目录 输出目录 [进程数]
已裁过的（子目录里已有图）会跳过；结果汇总在 输出目录/裁剪清单.csv。
切点算法见 cut_v2.py。
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from cut_v2 import find_cuts  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def crop_one(args):
    src, out_root = args
    out_dir = Path(out_root) / src.stem
    if out_dir.is_dir() and any(out_dir.glob('*.jpg')):
        return src.name, 'skip', '', ''
    try:
        im = Image.open(src).convert('RGB')
        arr = np.asarray(im)
        cuts = find_cuts(arr.astype(np.float32))
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (a, b) in enumerate(zip(cuts, cuts[1:]), 1):
            Image.fromarray(arr[a:b]).save(out_dir / f'{src.stem}_{i:02d}.jpg', quality=95)
        return src.name, f'{im.width}x{im.height}', len(cuts) - 1, ' '.join(map(str, cuts))
    except Exception as e:  # 单张失败不影响其它
        return src.name, 'error', '', repr(e)


if __name__ == '__main__':
    src_dir, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, row in enumerate(ex.map(crop_one, [(f, str(out_root)) for f in files]), 1):
            rows.append(row)
            print(f'[{k}/{len(files)}]', *row[:3], flush=True)
    with open(out_root / '裁剪清单.csv', 'a', newline='', encoding='utf-8-sig') as fp:
        w = csv.writer(fp)
        if fp.tell() == 0:
            w.writerow(['文件', '原图尺寸', '分图数', '切点/错误'])
        w.writerows(rows)
    errors = [r for r in rows if r[1] == 'error']
    print('完成', len(rows), '张，失败', len(errors), '张')
    for r in errors:
        print(' ', r[0], r[3])
