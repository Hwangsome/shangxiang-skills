"""批量第 1 步：对 分图根目录 下每个商品子目录做 OCR，写到 工作根目录/<商品>/ocr.json（已有的跳过）。

用法：python ocr_batch.py 分图根目录 工作根目录 [进程数]
      .venv_ortgpu\Scripts\python ocr_batch.py 分图根目录 工作根目录 --gpu
      （--gpu：单进程用显卡，需 onnxruntime-gpu；CUDA/cuDNN 借用 torch 自带的 DLL）
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_engine = None
USE_GPU = '--gpu' in sys.argv


def ocr_file(path):
    global _engine
    if _engine is None:
        if USE_GPU:
            import torch  # noqa: F401  先加载 torch 自带的 CUDA 12 / cuDNN 9
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR(det_use_cuda=USE_GPU, cls_use_cuda=USE_GPU, rec_use_cuda=USE_GPU)
    res, _ = _engine(str(path))
    return [{'text': t, 'score': round(float(s), 3), 'poly': [[int(x), int(y)] for x, y in p]}
            for p, t, s in (res or []) if t.strip()]


def do_product(args):
    src, work = Path(args[0]), Path(args[1])
    out = work / 'ocr.json'
    if out.exists():
        return src.name, 'skip'
    work.mkdir(parents=True, exist_ok=True)
    result = {f.name: ocr_file(f) for f in sorted(src.glob('*.jpg'))}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return src.name, sum(len(v) for v in result.values())


if __name__ == '__main__':
    src_root, work_root = Path(sys.argv[1]), Path(sys.argv[2])
    products = sorted(p for p in src_root.iterdir() if p.is_dir())
    jobs = [(str(p), str(work_root / p.name)) for p in products]
    if USE_GPU:
        results = map(do_product, jobs)
    else:
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        results = ProcessPoolExecutor(n).map(do_product, jobs)
    for k, (name, r) in enumerate(results, 1):
        print(f'[{k}/{len(jobs)}] {name} {r}', flush=True)
    print('完成')
