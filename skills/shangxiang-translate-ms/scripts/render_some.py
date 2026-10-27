"""只重画指定的分图（对症修复用，不必整批重跑），并替换 render_report.txt 里这些分图的段落。

用法：python render_some.py 分图根目录 工作根目录 输出根目录 清单.txt
  清单每行：商品名<TAB>分图文件名
"""
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from render_step import process

src_root, work_root, out_root, listfile = map(Path, sys.argv[1:5])
todo = defaultdict(list)
for line in listfile.read_text(encoding='utf-8').splitlines():
    if '\t' in line:
        prod, name = line.split('\t')[:2]
        todo[prod].append(name.strip())

done = failed = 0
for k, (prod, names) in enumerate(sorted(todo.items()), 1):
    work = work_root / prod
    ocr = json.loads((work / 'ocr.json').read_text(encoding='utf-8'))
    trans = json.loads((work / 'translations.json').read_text(encoding='utf-8'))
    sections = {}
    for name in sorted(set(names)):
        try:
            canvas, rep = process(src_root / prod / name, ocr[name], trans.get(name, {}))
            canvas.save(out_root / prod / name, quality=95, subsampling=0)
            sections[name] = [f'== {name}'] + ['   ' + r for r in rep]
            done += 1
        except Exception:
            failed += 1
            sections[name] = [f'!! {name} 失败', traceback.format_exc()]
    # 报告里按分图分段，替换重画过的段落，其余保持不变
    rp = work / 'render_report.txt'
    old = rp.read_text(encoding='utf-8').splitlines() if rp.exists() else []
    blocks, cur = {}, None
    order = []
    for line in old:
        m = re.match(r'(==|!!) (\S+\.jpg)', line)
        if m:
            cur = m.group(2)
            order.append(cur)
            blocks[cur] = [line]
        elif cur:
            blocks[cur].append(line)
    for name, sec in sections.items():
        if name not in blocks:
            order.append(name)
        blocks[name] = sec
    rp.write_text('\n'.join(l for n in sorted(set(order)) for l in blocks[n]) + '\n', encoding='utf-8')
    print(f'[{k}/{len(todo)}] {prod}  {len(names)} 张', flush=True)
print(f'完成：重画 {done} 张，失败 {failed} 张')
