"""批量第 2 步（翻译）的辅助脚本。词表 glossary_ms.json：{"中文原行": "马来文"}，所有商品共用，保证同一句译法一致。

python glossary_step.py collect 工作根目录
    汇总所有 ocr.json 里含中文、置信度 ≥0.6 的行，去重后按出现次数排序，
    （品牌 logo 保持原样：用环境变量 SX_BRAND_REGEX 指定品牌名正则；检测报告扫描件里的小字——同一分图 ≥12 行高 <16px 的中文——也不翻译）
    把词表里还没有的写到 工作根目录/待翻译.txt（每行：序号<TAB>次数<TAB>中文）。
python glossary_step.py merge 工作根目录 译文.tsv
    把译文（每行：中文<TAB>马来文）并入词表。
python glossary_step.py apply 工作根目录
    按词表给每个商品生成 translations.json（{分图: {行号: 马来文}}），词表里没有的行不写（保持原样）。
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

CJK = re.compile(r'[一-鿿]')
MIN_SCORE = 0.6
BRAND = re.compile(os.environ.get('SX_BRAND_REGEX', r'(?!)'), re.I)  # 默认不匹配任何行
DOC_PX, DOC_LINES = 16, 12


def norm(s):
    return s.strip()


def load_glossary(root):
    p = root / 'glossary_ms.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def save_glossary(root, g):
    (root / 'glossary_ms.json').write_text(json.dumps(g, ensure_ascii=False, indent=0), encoding='utf-8')


def height(l):
    ys = [p[1] for p in l['poly']]
    return max(ys) - min(ys)


def iter_lines(root):
    for ocr_path in sorted(root.glob('*/ocr.json')):
        ocr = json.loads(ocr_path.read_text(encoding='utf-8'))
        for name, lines in ocr.items():
            cjk = [(i, l) for i, l in enumerate(lines) if CJK.search(l['text']) and l['score'] >= MIN_SCORE]
            doc = sum(height(l) < DOC_PX for _, l in cjk) >= DOC_LINES
            for i, l in cjk:
                t = norm(l['text'])
                if BRAND.match(t) or (doc and height(l) < DOC_PX):
                    continue
                yield ocr_path.parent, name, i, t


def collect(root):
    g = load_glossary(root)
    cnt = Counter(t for *_x, t in iter_lines(root))
    todo = [(t, c) for t, c in cnt.most_common() if t not in g]
    with open(root / '待翻译.txt', 'w', encoding='utf-8') as f:
        for k, (t, c) in enumerate(todo, 1):
            f.write(f'{k}\t{c}\t{t}\n')
    print(f'不同的中文行 {len(cnt)}，词表已有 {len(cnt) - len(todo)}，待翻译 {len(todo)}')


def merge(root, tsv):
    g = load_glossary(root)
    n = 0
    for line in Path(tsv).read_text(encoding='utf-8').splitlines():
        if '\t' not in line:
            continue
        zh, ms = line.split('\t', 1)
        if zh.strip() and ms.strip():
            g[norm(zh)] = ms.strip()
            n += 1
    save_glossary(root, g)
    print(f'并入 {n} 条，词表共 {len(g)} 条')


def apply(root):
    g = load_glossary(root)
    per = {}
    for prod, name, i, t in iter_lines(root):
        if t in g:
            per.setdefault(prod, {}).setdefault(name, {})[str(i)] = g[t]
    for prod in sorted(root.glob('*/ocr.json')):
        d = prod.parent
        (d / 'translations.json').write_text(json.dumps(per.get(d, {}), ensure_ascii=False, indent=1),
                                             encoding='utf-8')
    print(f'生成 {len(per)} 个商品的 translations.json')


if __name__ == '__main__':
    cmd, root = sys.argv[1], Path(sys.argv[2])
    {'collect': lambda: collect(root), 'merge': lambda: merge(root, sys.argv[3]),
     'apply': lambda: apply(root)}[cmd]()
