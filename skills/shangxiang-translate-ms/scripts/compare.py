"""原图 / 译图左右并排，便于检查。用法：python compare.py 原图目录 译图目录 输出.jpg [文件名...]"""
import sys
from pathlib import Path

from PIL import Image

src, dst, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
names = sys.argv[4:] or sorted(p.name for p in dst.glob('*.jpg'))
pairs = []
for n in names:
    a, b = Image.open(src / n), Image.open(dst / n)
    pair = Image.new('RGB', (a.width * 2 + 10, a.height), 'white')
    pair.paste(a, (0, 0))
    pair.paste(b, (a.width + 10, 0))
    pairs.append(pair)
W = max(p.width for p in pairs)
sheet = Image.new('RGB', (W, sum(p.height + 10 for p in pairs)), 'white')
y = 0
for p in pairs:
    sheet.paste(p, (0, y))
    y += p.height + 10
sheet.save(out, quality=88)
