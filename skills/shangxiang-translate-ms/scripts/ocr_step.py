"""第 1 步：用 RapidOCR（PP-OCR 中文模型，onnxruntime）识别分图里的中文文字行，输出 ocr.json。

每行记录文字、置信度、四边形框。用法：python ocr_step.py 分图目录 输出目录
（本机 paddlepaddle-gpu 与 torch 的 CUDA 库冲突，所以不用 PaddleOCR。）
"""
import json
import sys
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
engine = RapidOCR()
result = {}
for f in sorted(src_dir.glob('*.jpg')):
    res, _ = engine(str(f))
    lines = []
    for poly, text, score in res or []:
        if text.strip():
            lines.append({'text': text, 'score': round(float(score), 3),
                          'poly': [[int(x), int(y)] for x, y in poly]})
    result[f.name] = lines
    print(f.name, len(lines))
(out_dir / 'ocr.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
