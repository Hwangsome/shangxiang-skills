---
name: shangxiang-translate-ms
description: 把电商商品详情分图上的中文翻译成马来文（Bahasa Melayu），在本机直接改图——OCR 识别中文、按共用词表翻译、擦掉原字、按原位置/颜色/字号/粗细写回马来文，产品照片和排版不动。用户说“商详翻译成马来文 / 详情图翻译 / 马来西亚站详情页 / 把图上的中文换成马来文 / Shopee 马来详情图”，或者要把裁剪好的商详分图本地化时，都应使用本 Skill。也可作为翻译成其它语言的起点。
---

# 商详翻译（马来文）

全程在本机运行，不调用云端图像服务。保留原图背景，只替换含中文的文字行；英文、型号、数字原样保留。输出与原分图同名、同尺寸。

## 准备

- Python 依赖见 `requirements.txt`：`rapidocr_onnxruntime`、`opencv-python`、`numpy`、`Pillow`、`torch`、`simple-lama-inpainting`（照片纹理底的擦字修补，建议有 NVIDIA 显卡）。
- 字体：`render_step.py` 顶部的 `FONTS` 默认指向 Windows 的微软雅黑（`C:\Windows\Fonts\msyh*.ttc`）。在别的系统上请改成本机可用的字体。
- 显卡加速 OCR（可选）：装 `onnxruntime-gpu`，运行时加 `--gpu`。脚本会先 `import torch`，借用 torch 自带的 CUDA/cuDNN。不要和 paddlepaddle-gpu 装在同一个环境里，两者的 CUDA 库会冲突。
- 品牌名保持原样：设置环境变量 `SX_BRAND_REGEX`（品牌名的正则表达式），匹配到的行不翻译。

下面的 `<分图根目录>` 是裁剪结果（每个商品一个子目录，可以用 `shangxiang-crop` 生成），`<工作目录>` 存放中间文件，`<输出目录>` 存放译图。

## 流程

### 1. 识别中文

```bash
python scripts/ocr_batch.py <分图根目录> <工作目录> [进程数]      # CPU
python scripts/ocr_batch.py <分图根目录> <工作目录> --gpu          # 显卡，单进程
```

每个商品得到 `<工作目录>/<商品>/ocr.json`，已存在的会跳过。

### 2. 翻译（这一步由你来做）

```bash
python scripts/glossary_step.py collect <工作目录>
```

脚本会把所有商品里还没进词表的中文行去重，写到 `<工作目录>/待翻译.txt`（每行：序号、出现次数、中文）。品牌 logo，以及检测报告扫描件里的成片小字，会自动跳过。

**读这个文件，把每一行翻成马来文**，写成 TSV（每行：`中文<TAB>马来文`）。翻译要求：

- 用马来西亚电商常见、自然的说法，不要逐字硬翻。卖点标题要短，因为图上的位置有限，译文越长字就会缩得越小。
- 数字、单位、型号原样保留（5kg、2.5L、E1601）。
- 同一个词在所有商品里用同一个译法，词表就是为了保证这一点。
- 单字碎片、产品包装上的第三方品牌、报关单/检测报告正文：不要写进 TSV，留空就会保持原样。

然后合并进词表，并生成每个商品的 `translations.json`：

```bash
python scripts/glossary_step.py merge <工作目录> 译文.tsv
python scripts/glossary_step.py apply <工作目录>
```

词表 `<工作目录>/glossary_ms.json` 由所有商品共用，以后有新商品，只需翻译新增的行。

### 3. 擦字 + 写马来文

```bash
python scripts/render_batch.py <分图根目录> <工作目录> <输出目录> [--force] [--skip 商品名] [只处理的商品名...]
```

已有输出的分图会跳过；`--force` 会覆盖重画。每个商品的排版记录在 `render_report.txt` 里。

双色胶囊标签（左半一种颜色写“S小号”、右半另一种颜色写“30只装/盒”）如果被 OCR 连成了一行，先运行 `python scripts/split_pills.py <分图根目录> <工作目录> [清单.txt] [--dry]` 把它拆成两段（会备份 `ocr.before_split.json`），再做第 2、3 步。

### 4. 检查

```bash
python scripts/compare.py <原分图目录> <译图目录> 对比.jpg [文件名...]
```

这会生成左原图、右译图的对比，拿给用户抽查。重点看：字有没有擦干净、字号是否小到看不清、文字有没有压到照片上。

### 5. 对症修复

发现某一类问题时（某个词译得太长、某种版式排坏了），先改词表或规则，然后**只重画受影响的分图**，不要整批重跑：

```bash
python scripts/render_some.py <分图根目录> <工作目录> <输出目录> 清单.txt   # 清单每行：商品名<TAB>分图文件名
```

改词表之后，要先 `glossary_step.py apply`，再重画。

## 交给下一步

译图准备上传前，用 `shangxiang-upload-check` 检查尺寸、比例和文件大小。

## 排版规则与已知不足

擦字和排版的详细规则（纯色底、渐变底和照片底分别怎么擦，字号、粗细、对齐怎么定，马来文较长时如何加宽、折行、缩小）见 `references/rendering-rules.md`。调整 `render_step.py` 之前先读它；向用户解释“为什么这张图这样排”时也可以参考。
