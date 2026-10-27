"""批量第 3 步：对每个商品执行擦字 + 写马来文（render_step.process），已有输出的分图跳过。

用法：python render_batch.py 分图根目录 工作根目录 输出根目录 [--force] [--skip 商品名] [只处理的商品名...]
  --force：覆盖已有输出（并重写 render_report.txt）；--skip：排除某个商品，可多次使用
"""
import json
import sys
import traceback
from pathlib import Path

from render_step import process

src_root, work_root, out_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
args = sys.argv[4:]
force = '--force' in args
skip = {args[k + 1] for k, a in enumerate(args) if a == '--skip'}
only = {a for k, a in enumerate(args) if not a.startswith('--') and (k == 0 or args[k - 1] != '--skip')}
products = sorted(p.parent.name for p in work_root.glob('*/translations.json'))
if only:
    products = [p for p in products if p in only]
products = [p for p in products if p not in skip]
done = failed = 0
for k, prod in enumerate(products, 1):
    work = work_root / prod
    ocr = json.loads((work / 'ocr.json').read_text(encoding='utf-8'))
    trans = json.loads((work / 'translations.json').read_text(encoding='utf-8'))
    out_dir = out_root / prod
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for name in sorted(ocr):
        out = out_dir / name
        if out.exists() and not force:
            continue
        try:
            canvas, rep = process(src_root / prod / name, ocr[name], trans.get(name, {}))
            canvas.save(out, quality=95, subsampling=0)
            report += [f'== {name}'] + ['   ' + r for r in rep]
            done += 1
        except Exception:
            failed += 1
            report.append(f'!! {name} 失败\n{traceback.format_exc()}')
    if report:
        with open(work / 'render_report.txt', 'w' if force else 'a', encoding='utf-8') as f:
            f.write('\n'.join(report) + '\n')
    print(f'[{k}/{len(products)}] {prod}', flush=True)
print(f'完成：新出图 {done} 张，失败 {failed} 张')
