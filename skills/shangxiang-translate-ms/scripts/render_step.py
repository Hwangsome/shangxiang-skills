"""第 2 步：擦掉中文、按原位置/颜色/字号/粗细/对齐写上马来文（本机运行，不调用任何云服务）。

输入：分图目录、工作目录（含 ocr.json、translations.json）、输出目录。
- 只处理 translations.json 里有译文、且原文含中文的行；英文、型号等原样保留，擦字时也不会碰到它们。
- 擦字：只擦文字笔画所在的行；纯色底直接用底色填，有纹理/照片的底用 LaMa 修补。
- 底色取框内边缘的主色（文字压在色块/标签上时也对），字色取笔画颜色，粗细按笔画宽度/字高判断。
- 段落：一句话被拆成几行的合成一段；同一列、同样式的几行（如编号注意事项）作为一叠统一字号、
  依次往下排，多出来的行占用下方空白。
- 宽度只向空白背景扩展；字号不低于原来的 60%（且 ≥12px），实在放不下宁可越过细线也不缩到看不清。
用法：python render_step.py 分图目录 工作目录 输出目录
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

FONTS = {'regular': r'C:\Windows\Fonts\msyh.ttc', 'bold': r'C:\Windows\Fonts\msyhbd.ttc',
         'light': r'C:\Windows\Fonts\msyhl.ttc'}
CJK = re.compile(r'[\u4e00-\u9fff]')
MARGIN = 14
MIN_SCALE, MIN_PX = 0.6, 12
PLATE_PAD = 10
DEBUG = False
DEBUG_DIR = '.'
SCALES = (1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.65, 0.6)

_lama = None


def lama():
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
    return _lama


def box_of(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


# ---------- 分析 ----------

def dominant(px):
    q = (px // 24).astype(np.int64)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    sel = keys == vals[counts.argmax()]
    return np.median(px[sel], axis=0), counts.max() / len(keys), px[sel]


def box_background(arr, box):
    """底色取框内最外 3px 一圈的主色；返回 (底色, 是否纯色)。"""
    x0, y0, x1, y1 = box
    inner = arr[y0:y1, x0:x1].astype(np.int16)
    e = 3
    ring = np.concatenate([inner[:e].reshape(-1, 3), inner[-e:].reshape(-1, 3),
                           inner[:, :e].reshape(-1, 3), inner[:, -e:].reshape(-1, 3)])
    bg, share, px = dominant(ring)
    return bg, share > 0.6 and float(px.std(axis=0).mean()) < 6


def ink_rows(dist_mask):
    """框内真正有字的行段（最大的一段，允许 2 行以内的断开），返回 (起, 止) 相对框顶。"""
    rows = dist_mask.mean(axis=1) > 0.01
    best, cur = (0, 0, 0), None
    for y, r in enumerate(list(rows) + [False]):
        if r:
            cur = [y, y + 1, cur[2] if cur else 0] if cur is None else [cur[0], y + 1, cur[2]]
            cur[2] += dist_mask[y].sum()
            gap = 0
        elif cur is not None:
            gap = locals().get('gap', 0) + 1
            if gap > 2 or y == len(rows):
                if cur[2] > best[2]:
                    best = tuple(cur)
                cur = None
    return best[0], best[1]


def glyph_mask(arr, box, bg=None):
    """框内文字笔画（只取主要字行），整图大小的布尔图。"""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    if bg is None:
        bg, _ = box_background(arr, box)
    dist = np.abs(arr[y0:y1, x0:x1].astype(np.int16) - bg).max(axis=2)
    m = dist > 45
    if m.sum() < 10:
        m = dist > 25
    a, b = ink_rows(m)
    out = np.zeros((h, w), bool)
    if b > a:
        out[y0 + a:y0 + b, x0:x1] = m[a:b]
    return out


def plate_extent(arr, box, bg):
    """文字压在色块（标签/按钮）上时，返回色块的 [x0, y0, x1, y1] 和色块外的底色；不是色块返回 None。"""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - 6), max(0, y0 - 6), min(w, x1 + 6), min(h, y1 + 6)
    region = arr[Y0:Y1, X0:X1].astype(np.int16)
    ring = np.ones(region.shape[:2], bool)
    ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
    outer, share, _ = dominant(region[ring])
    sides = np.concatenate([region[:, :x0 - X0].reshape(-1, 3), region[:, x1 - X0:].reshape(-1, 3)])
    inside_plate = len(sides) > 0 and np.abs(np.median(sides, axis=0) - bg).max() < 60   # 渐变色块两侧颜色可差不少
    if share < 0.5 and not inside_plate:
        return None

    def same(px):
        return np.abs(px.astype(np.int16) - bg).max(axis=-1) < 30

    if inside_plate:
        # 文字框整个落在色块里（如药丸形标签）：沿中线向外找色块边，色块不大、外面颜色明显不同才算
        # 色块常是渐变（左深右浅的金色标签）：逐像素看颜色是否连续，不和框内底色硬比
        def walk(line, i, step):
            # 渐变是缓慢变化；色块边缘（即使有抗锯齿）是 4px 内变化很大，遇到就停
            line = line.astype(np.int16)
            i0 = i
            while 0 <= i + step < len(line) and np.abs(line[i + step] - bg).max() < 90:
                back = line[max(0, min(len(line) - 1, i - 3 * step))] if abs(i - i0) >= 3 else line[i0]
                if np.abs(line[i + step] - back).max() > 25:
                    break
                i += step
            return i

        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        px0, px1 = walk(arr[cy], x0, -1), walk(arr[cy], x1 - 1, 1) + 1
        # 竖向在文字左侧（仍在色块内）那一列从中线往上下找，避开笔画
        xc = max(0, x0 - 4)
        py0, py1 = min(y0, walk(arr[:, xc], cy, -1)), max(y1, walk(arr[:, xc], cy, 1) + 1)
        cx = xc
        if DEBUG:
            print('   plate?', box, px0, px1, py0, py1)
        if px0 == 0 or px1 == w or py0 == 0 or py1 == h or py1 - py0 > 1.9 * (y1 - y0) or \
                px1 - px0 > 0.6 * w:
            return None
        out_px = np.array([arr[cy, max(0, px0 - 3)], arr[cy, min(w - 1, px1 + 2)],
                           arr[max(0, py0 - 3), cx], arr[min(h - 1, py1 + 2), cx]]).astype(np.int16)
        if (np.abs(out_px - bg).max(axis=1) >= 25).sum() < 3:     # 四周至少三面颜色明显不同
            return None
        o = np.median(out_px, axis=0)
        if np.abs(out_px[0] - out_px[1]).max() > 40:
            # 左右两端外面颜色差很多：多半一端连着图标（✓、×），找到的不是真正的边，只缩字不拉长色块
            return [px0, py0, px1, py1], None
        return [px0, py0, px1, py1], tuple(int(c) for c in o)

    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    py0, py1 = y0, y1
    while py0 > 0 and same(arr[py0 - 1, x0 + 2]) and same(arr[py0 - 1, x1 - 3]):
        py0 -= 1
    while py1 < h and same(arr[py1, x0 + 2]) and same(arr[py1, x1 - 3]):
        py1 += 1
    band = arr[max(py0, cy - 2):min(py1, cy + 3)]
    col_same = same(band).mean(axis=0) > 0.5
    px0, px1 = x0, x1
    while px0 > 0 and col_same[px0 - 1]:
        px0 -= 1
    while px1 < w and col_same[px1]:
        px1 += 1
    return [px0, py0, px1, py1], tuple(int(c) for c in outer)


def analyse(arr, box):
    x0, y0, x1, y1 = box
    bg, flat = box_background(arr, box)
    mask = glyph_mask(arr, box, bg)
    inner = arr[y0:y1, x0:x1].astype(np.int16)
    m = mask[y0:y1, x0:x1]
    px = inner[m]
    if len(px):
        d = np.abs(px - bg).max(axis=1)
        fg = np.median(px[d >= np.percentile(d, 50)], axis=0)
    else:
        fg = np.zeros(3)
    rows = np.flatnonzero(m.any(axis=1))
    ink_h = (rows[-1] - rows[0] + 1) if len(rows) else (y1 - y0)
    return dict(bg=tuple(int(c) for c in bg), fg=tuple(int(c) for c in fg), mask=mask,
                weight=weight_of(arr, box, bg, mask), flat=flat, ink_h=int(ink_h),
                plate=plate_extent(arr, box, bg))


def weight_of(arr, box, bg, mask):
    """粗细：笔画宽度 / 字高，配合笔画密度。笔画按“字色与底色的中点”取，金色、浅色字也不会被算细。"""
    x0, y0, x1, y1 = box
    inner = arr[y0:y1, x0:x1].astype(np.int16)
    m = mask[y0:y1, x0:x1]
    rows = np.flatnonzero(m.any(axis=1))
    if not len(rows):
        return 'regular'
    bh = max((rows[-1] - rows[0] + 1) * 1.15, 1)   # 用真实字高（含少量留白）
    d = np.abs(inner - np.array(bg)).max(axis=2)
    fg_d = np.percentile(d[m], 90) if m.any() else 120
    m8 = ((d > 0.5 * fg_d) & m).astype(np.uint8)
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    peaks = (dt >= cv2.dilate(dt, np.ones((3, 3)))) & (dt > 0)
    ratio = 2 * float(np.median(dt[peaks])) / bh if peaks.any() else 0
    # 另一种笔画宽度估计：2×面积 / 周长；两者取平均，单个估计受像素量化影响较大
    cs, _ = cv2.findContours(m8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cs)
    ratio = (ratio + 2 * float(m8.sum()) / max(perim, 1) / bh) / 2
    dens = float(m8[rows[0]:rows[-1] + 1].mean())
    # 小字的笔画至少 1~2 像素，比例天然偏大：阈值随字高放宽
    if ratio >= 0.068 + 0.5 / bh or (dens >= 0.38 and bh >= 38):
        return 'bold'
    if ratio < 0.05 and dens < 0.15:
        return 'light'
    return 'regular'


# ---------- 分组 ----------

def same_style(p, it):
    ph, h = p['box'][3] - p['box'][1], it['box'][3] - it['box'][1]
    return 0.8 <= h / ph <= 1.25 and p['weight'] == it['weight'] and \
        max(abs(a - b) for a, b in zip(p['fg'], it['fg'])) < 60


def group_lines(items):
    """一句话被拆成多行的合成一段：上一行译文以逗号/分号结尾，或下一行以小写开头；编号行不合并。"""
    groups = []
    for it in items:
        g = groups[-1] if groups else None
        if g:
            p = g[-1]
            gap = it['box'][1] - p['box'][3]
            ov = min(p['box'][2], it['box'][2]) - max(p['box'][0], it['box'][0])
            narrow = min(p['box'][2] - p['box'][0], it['box'][2] - it['box'][0])
            same_col = ov > 0.5 * narrow or abs(p['box'][0] - it['box'][0]) <= 8
            continues = p['text'].rstrip().endswith((',', ';', '，')) or it['text'][:1].islower()
            numbered = re.match(r'\s*\d+[.、]', it['text']) is not None
            if continues and not numbered and same_col and same_style(p, it) and \
                    -2 <= gap < 0.9 * (p['box'][3] - p['box'][1]):
                g.append(it)
                continue
        groups.append([it])
    return groups


def stack_groups(groups):
    """同一列（左对齐 x 相同）、同样式、上下紧挨的几段作为一叠，统一字号、依次往下排。"""
    stacks = []
    for g in groups:
        # 不只看上一叠：同一行旁边另有文字时，排序会把它插在这一列中间
        for s in reversed(stacks):
            p, q = s[-1][-1], g[0]
            lh = p['box'][3] - p['box'][1]
            # 段落里有彩色高亮的行（颜色、粗细不同）也算同一叠：只看左边对齐和字高
            # 但颜色相同、粗细不同的是“标题 + 说明”，不算；色块标签各自单独排
            ph, qh = p['box'][3] - p['box'][1], q['box'][3] - q['box'][1]
            same_color = max(abs(a - b) for a, b in zip(p['fg'], q['fg'])) < 60
            # （已有 3 行以上的段落里，个别行粗细判得不同不算）
            # 图标格/多栏卖点那一行不和上面的副标题连成一叠
            if p['plate'] or q['plate'] or bool(p.get('grid')) != bool(q.get('grid')) or \
                    (same_color and p['weight'] != q['weight'] and sum(len(x) for x in s) < 3):
                continue
            if abs(p['box'][0] - q['box'][0]) <= 8 and 0.8 <= qh / ph <= 1.25 and \
                    0 <= q['box'][1] - p['box'][3] < 1.6 * lh:
                s.append(g)
                break
        else:
            stacks.append([g])
    return stacks


# ---------- 排版 ----------

def font_at(weight, size):
    return ImageFont.truetype(FONTS[weight], max(8, int(round(size))))


def base_size(weight, ink_h):
    """字号：让 'Ag' 的墨迹高度约等于原中文字的墨迹高度。"""
    f = font_at(weight, 100)
    l, t, r, b = f.getbbox('Ag')
    return 100 * ink_h * 0.95 / (b - t)


def tw(font, s):
    l, t, r, b = font.getbbox(s)
    return r - l


def wrap(font, text, maxw):
    lines, cur = [], ''
    for wd in text.split(' '):
        trial = (cur + ' ' + wd).strip()
        if cur and tw(font, trial) > maxw:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def content_mask(px, bg):
    """与底色明显不同的像素（别的文字、图形、产品）。照片/纹理底（大理石墙面等）按自身起伏放宽阈值，
    纹理本身不算内容；返回 (掩码, 是否纹理底)。"""
    d = np.abs(px.astype(np.int16) - np.array(bg)).max(axis=-1)
    noise = float(np.percentile(d, 40))
    textured = noise > 6
    return d > (max(35, 2.5 * noise + 20) if textured else 35), textured


def busy_map(arr, bg, y0, y1):
    m, textured = content_mask(arr[y0:y1], bg)
    col = m.mean(axis=0)
    if not textured:                                      # 平滑底上零星的噪点/斑点不算挡住
        return np.convolve(col, np.ones(5) / 5, mode='same') > 0.05
    col = np.convolve(col, np.ones(7) / 7, mode='same')    # 纹理里零星的亮点、细纹不算挡住
    return col > 0.08


def free_span(arr, bg, band, x0, x1, blockers):
    h, w = arr.shape[:2]
    busy = busy_map(arr, bg, *band)
    for bx0, by0, bx1, by1 in blockers:
        if min(bx1, x1) - max(bx0, x0) > 0.3 * min(bx1 - bx0, x1 - x0):
            # 和自己重叠的（压在下面的装饰英文等）不算挡在旁边，原图的字本来就压在它上面
            if min(by1, band[1]) > max(by0, band[0]):
                busy[max(0, bx0):min(w, bx1)] = False
            continue
        overlap = min(by1, band[1]) - max(by0, band[0])
        if overlap > 0.5 * min(by1 - by0, band[1] - band[0]):   # 只有同一行的文字才算挡在旁边
            # 同一行的另一段也会被翻译、也可能变长。字高相近（并列标签）：空隙两边各占一半；
            # 大标题旁的小字说明：标题占整个空隙（只留 12px），小字不往标题那边扩
            ratio = (band[1] - band[0]) / max(1, by1 - by0)
            if ratio > 1.6:                                   # （band 是 OCR 框、blocker 是墨迹框，后者天然偏矮）
                bx0, bx1 = bx0 - 8, bx1 + 8
            elif ratio < 0.62:
                bx0, bx1 = (min(bx0, x1) if bx0 >= x1 else bx0), (max(bx1, x0) if bx1 <= x0 else bx1)
            elif bx0 >= x1:
                bx0 = min(bx0, (x1 + bx0) // 2 + 6)
            elif bx1 <= x0:
                bx1 = max(bx1, (bx1 + x0) // 2 - 6)
            busy[max(0, bx0 - 4):min(w, bx1 + 4)] = True
    left = 0
    while x0 - left - 1 >= 0 and not busy[x0 - left - 1]:
        left += 1
    right = 0
    while x1 + right < w and not busy[x1 + right]:
        right += 1
    # 空到图边时也留出页边距，文字不贴边
    return max(0, min(left, x0 - MARGIN)), max(0, min(right, w - MARGIN - x1))


def free_below(arr, bg, y_start, lo, hi, limit):
    """从 y_start 往下，横向 [lo,hi) 范围内保持同一底色的行数（遇到别的内容就停）。"""
    h, w = arr.shape[:2]
    lo, hi = int(max(0, lo)), int(min(w, hi))
    y_end = int(min(h, limit))
    if y_end <= y_start or hi <= lo:
        return 0
    m, textured = content_mask(arr[y_start:y_end, lo:hi], bg)
    rows = m.mean(axis=1)
    if textured:
        rows = np.convolve(rows, np.ones(3) / 3, mode='same')
    hit = np.flatnonzero(rows > (0.06 if textured else 0.02))
    return int(hit[0]) if len(hit) else y_end - y_start


def neighbor_align(box, others):
    x0, y0, x1, y1 = box
    h = y1 - y0
    found = set()
    for o in others:
        if abs((o[1] + o[3]) / 2 - (y0 + y1) / 2) > 2.5 * h:
            continue
        if abs(o[0] - x0) <= 6:
            found.add('left')
        if abs((o[0] + o[2]) - (x0 + x1)) <= 12:
            found.add('center')
        if abs(o[2] - x1) <= 6:
            found.add('right')
    return found


def plan_group(arr, g, others, W, stack_mode):
    """算出一段的对齐方式、可用宽度、字号基准等（还不定字号）。"""
    gx0, gy0 = min(it['box'][0] for it in g), min(it['box'][1] for it in g)
    gx1, gy1 = max(it['box'][2] for it in g), max(it['box'][3] for it in g)
    first = g[0]
    left, right = free_span(arr, first['bg'], (gy0, gy1), gx0, gx1, others)
    found = set()
    for it in g:
        found |= neighbor_align(it['box'], others)
    if len(g) > 1:
        if max(abs(it['box'][0] - gx0) for it in g) <= 6:
            found.add('left')
        if max(abs((it['box'][0] + it['box'][2]) / 2 - (gx0 + gx1) / 2) for it in g) <= 8:
            found.add('center')
    if stack_mode and gx1 > 0.8 * W and gx0 > 0.25 * W and right < 0.1 * W:
        found = {'right'}                                 # 贴着右页边排的一叠标题：靠右
    elif stack_mode or re.match(r'\s*[·•●]', g[0]['text']):   # 项目符号列表总是靠左
        found = {'left'}
    elif len(g) == 1 and g[0].get('grid') and not ({'left', 'right'} <= found) and             ('center' in found or not ({'left', 'right'} & found)):
        found = {'center'}                                # 图标格里等间距的一排标签：居中（和上下行左/右对齐的除外）
    cx = (gx0 + gx1) / 2
    bw = gx1 - gx0
    # 中心和某一边同时对得上（宽度相近的上下两行）且两侧都有空：多半是居中排版（图标下的标签）
    both_free = min(left, right) > 0.3 * bw
    # 左、中、右同时对得上（上下两行等宽，如多栏卖点）看不出来，按靠左（文案大多靠左）
    short = len(g) == 1 and len(CJK.findall(g[0].get('zh', ''))) <= 4
    if short and not stack_mode and under_picture(arr, g[0]['box'], g[0]['bg']):
        found = {'center'}                                # 图片正下方的名称标签：跟图片居中
    if 'center' in found and (len(found) == 1 or (both_free and not {'left', 'right'} <= found)
                              or (short and found == {'left', 'center'})):
        align = 'center'                                  # 短标签（图片下的名称）和上面的标签中心、左边都对得上：按居中
    elif {'left', 'right'} <= found:
        align = 'left'
    elif found - {'center'}:
        align = (found - {'center'}).pop()
    elif abs(cx - W / 2) < W * 0.06 or \
            ((gx0 - left > 2) == (gx1 + right < W - 2) and
             abs(cx - ((gx0 - left) + (gx1 + right)) / 2) < 0.08 * (left + right + gx1 - gx0)):
        # 空白区间的中点：只有两侧都被挡住（或都通到图边）时才看；左边通到图边、右边被挡住的是靠左排的
        align = 'center'
    elif gx1 + right >= W - 2 and W - gx1 < 0.05 * W and not (gx0 - left <= 2 and gx0 < 0.12 * W):
        align = 'right'                                   # 贴着右页边、右侧一直空到图边：靠右排
    else:
        align = 'left'                                    # 其余都按靠左排（文案大多左对齐，行首常有图标）
    maxw = max(bw, {'left': bw + right, 'right': bw + left, 'center': bw + 2 * min(left, right)}[align] - 4)
    plate = first['plate'] if len(g) == 1 else None
    tight = None
    if plate:
        # 色块标签：文字放不下时把色块往两边（空白处）拉长，文字在色块里居中
        (px0, py0, px1, py1), outer = plate
        trim = max(1, (py1 - py0) // 5)                   # 色块上下边常连着分隔线，扫描时去掉边缘几行
        l_free, r_free = free_span(arr, outer, (py0 + trim, py1 - trim), px0, px1, others) if outer else (8, 8)
        l_free, r_free = max(0, l_free - 8), max(0, r_free - 8)
        maxw = (px1 - px0) + l_free + r_free - 2 * PLATE_PAD
        align = 'center'
        plate = plate + ((l_free, r_free),)
        tight = (px1 - px0) - 2 * PLATE_PAD                # 不拉长色块时的宽度：略缩小能放下就不拉
    if len(g) == 1 and g[0].get('seg'):
        align, maxw = 'center', bw - 6                    # 双色胶囊的一半：在本段内居中，放不下就缩小
    pitch = (g[-1]['box'][1] - g[0]['box'][1]) / (len(g) - 1) if len(g) > 1 else None
    return dict(g=g, box=[gx0, gy0, gx1, gy1], align=align, maxw=maxw, cx=cx, plate=plate,
                maxw_tight=tight if plate else maxw,
                lh=float(np.median([it['box'][3] - it['box'][1] for it in g])),
                ink_h=float(np.median([it['ink_h'] for it in g])),
                weight=first['weight'], fg=first['fg'], bg=first['bg'], pitch=pitch,
                text=' '.join(it['text'] for it in g), left=left, right=right)


def layout_stack(arr, plans, others, H, W):
    """一叠段落统一字号依次往下排；返回每段的 (字号, 行列表, 行距, 顶部 y)。"""
    p0 = plans[0]
    top0 = p0['box'][1]
    last_bottom = plans[-1]['box'][3]
    lo = min(p['box'][0] - (p['left'] if p['align'] != 'left' else 0) for p in plans)
    hi = max(p['box'][0] + p['maxw'] for p in plans)
    below = [b[1] for b in others if b[1] >= last_bottom - 2 and b[0] < hi and b[2] > lo]
    limit = min(below) if below else H - MARGIN
    room_extra = free_below(arr, p0['bg'], last_bottom, lo, hi, limit)
    bottom_limit = last_bottom + max(0, room_extra - 2)   # 行数不变时总能放回原位置
    gaps = [plans[k + 1]['box'][1] - plans[k]['box'][3] for k in range(len(plans) - 1)]
    if all(p['align'] == 'left' for p in plans):
        # 靠左的一叠标题：下一行能写到的宽度，上一行也能用（右边多是淡淡的装饰底纹，不是真正的障碍）
        wmax = max(p['maxw'] for p in plans)
        for p in plans:
            x0, y0, x1, y1 = p['box']
            lo_x, hi_x = x0 + p['maxw'], x0 + wmax
            # 只有加宽的那段同一行上没有别的文字才加宽（右边常有小字说明）
            clash = any(b[0] < hi_x + 8 and b[2] > lo_x and min(b[3], y1 + 4) > max(b[1], y0 - 4) for b in others)
            if not clash:
                p['maxw'] = max(p['maxw'], wmax)
    ws = [p['weight'] for p in plans]                     # 同一叠的粗细按多数统一（小字容易判错）
    for p in plans:
        p['weight'] = max(set(ws), key=ws.count)
    size0 = base_size(p0['weight'], p0['ink_h'])

    def row_mate(p):
        y0, y1 = p['box'][1], p['box'][3]
        return any(min(b[3], y1) - max(b[1], y0) > 0.5 * (y1 - y0) and (b[0] >= p['box'][2] or b[2] <= p['box'][0])
                   for b in others)
    # 表格的一列（每行旁边都有对应的值）：折行会让行和值错开，只缩小、不折行
    table = all(row_mate(p) for p in plans)
    # 先试缩小（最小到 60%）、每段保持原行数：比把一段标题拆成“一词一行”更接近原版式
    for strict in ((True,) if table else (True, False)):
        for scale in SCALES + ((0.5, 0.45, 0.4) if table else ()):
            if strict and scale < MIN_SCALE and not table:
                break
            out = try_stack(plans, p0, size0, scale, top0, gaps, strict, table)
            if out and out[-1] <= bottom_limit:
                return out[0]
    for p in plans:                                       # 下面的兜底按原行框排
        p.pop('lh_eff', None)
    if table:                                             # 仍放不下：最小字号、各行不折，宽度允许略超
        f = font_at(p0['weight'], max(MIN_PX, size0 * 0.4))
        return [(font_at(p['weight'], f.size), [p['text']], f.size * 1.3, p['box'][1]) for p in plans]
    # 最小字号仍放不下：用最小字号照排（可能越过下方空白）
    return stack_fallback(plans, p0, size0, top0, gaps)


def try_stack(plans, p0, size0, scale, top0, gaps, strict, table=False):
    f = font_at(p0['weight'], max(MIN_PX, size0 * scale))
    step = max((p0['pitch'] or p0['lh'] * 1.3) * scale, f.size * 1.3)
    # 字缩小了，行框和段间距也按比例收紧（表格除外：要和旁边的值对齐），整块再在原位置里垂直居中
    r = 1.0 if table else min(1.0, f.size / max(size0, 1))
    y, out = top0, []
    for k, p in enumerate(plans):
        f = font_at(p['weight'], f.size)                  # 统一字号，各段保留自己的粗细
        lines = wrap(f, p['text'], p['maxw'])
        if any(tw(f, s) > p['maxw'] for s in lines) or (strict and len(lines) > len(p['g'])):
            return None
        p['lh_eff'] = p['lh'] * r
        out.append((f, lines, step, y))
        y += step * (len(lines) - 1) + p['lh_eff']
        if k < len(plans) - 1:
            y += gaps[k] * r
    orig = plans[-1]['box'][3] - top0
    if r < 1 and y - top0 < orig:
        shift = (orig - (y - top0)) / 2
        out = [(f, l, st, yy + shift) for f, l, st, yy in out]
        y += shift
    return out, y


def stack_fallback(plans, p0, size0, top0, gaps):
    f = font_at(p0['weight'], max(MIN_PX, size0 * MIN_SCALE))
    step = f.size * 1.3
    y, out = top0, []
    for k, p in enumerate(plans):
        f = font_at(p['weight'], f.size)
        lines = wrap(f, p['text'], p['maxw'])
        out.append((f, lines, step, y))
        y += step * (len(lines) - 1) + p['lh'] + (gaps[k] if k < len(plans) - 1 else 0)
    return out


def layout_single(arr, p, others, H):
    gx0, gy0, gx1, gy1 = p['box']
    n_orig = len(p['g'])
    lo, hi = (gx0 - p['left'], gx1 + p['right']) if p['align'] != 'center' else \
        (p['cx'] - p['maxw'] / 2, p['cx'] + p['maxw'] / 2)
    below = [b[1] for b in others if b[1] > gy0 + 2 and b[0] < hi and b[2] > lo]
    limit = min(below) if below else H - MARGIN
    room = gy1 - gy0 + free_below(arr, p['bg'], gy1, max(gx0, lo), min(gx1, hi), limit)
    size0 = base_size(p['weight'], p['ink_h'])
    # 单行时字的上下不能压到正上方/正下方的文字（如中文标签下紧挨的英文小字）
    yc = (gy0 + gy1) / 2
    near = [b for b in others if b[0] < hi and b[2] > lo       # others 已是真实字迹范围
            and not (b[1] < gy1 and b[3] > gy0)]               # 和本行上下重叠的（压着的装饰字）不算上下邻居
    below_top = min([b[1] for b in near if b[1] > yc] + [H])
    above_bot = max([b[3] for b in near if b[3] < yc] + [0])
    half = min(below_top - yc, yc - above_bot) - 1
    probe = font_at(p['weight'], 100)
    l, t, r, b = probe.getbbox('Ag')
    size0 = min(size0, max(MIN_PX, 100 * 2 * half / (b - t)))
    # 先试略缩小（≥82%）保持原行数，比多折一行更接近原版式
    for scale in SCALES:
        if scale < 0.82:
            break
        f = font_at(p['weight'], max(MIN_PX, size0 * scale))
        lines = wrap(f, p['text'], p['maxw_tight'])
        if len(lines) <= n_orig and all(tw(f, s) <= p['maxw_tight'] for s in lines):
            return f, lines, max((p['pitch'] or p['lh'] * 1.25) * scale, f.size * 1.25)
    for scale in SCALES:
        f = font_at(p['weight'], max(MIN_PX, size0 * scale))
        lines = wrap(f, p['text'], p['maxw'])
        step = max((p['pitch'] or p['lh'] * 1.25) * scale, f.size * 1.25)
        # 色块标签里不折行（折了会超出色块）：放不下靠缩小 + 拉长色块
        fits_v = len(lines) <= n_orig or (not p['plate'] and not p['g'][0].get('seg') and step * len(lines) <= room)
        if all(tw(f, s) <= p['maxw'] for s in lines) and fits_v:
            return f, lines, step
    # 放不下：最小可读字号、行数不超过原来的，宽度允许越界
    f = font_at(p['weight'], max(MIN_PX, size0 * MIN_SCALE))
    # 宁可多折一行也不越出图片（色块标签里不折）
    lines = [p['text']] if (p['plate'] or p['g'][0].get('seg')) and n_orig == 1 else wrap(f, p['text'], p['maxw'])
    return f, lines, f.size * 1.25


def stretch_plate(canvas, plate, extra, l_free):
    """把色块从中间拉长 extra 像素（重复中间那一列，圆角、描边、渐变都保持原样），尽量两边对称。"""
    (px0, py0, px1, py1), _, _ = plate
    a = np.asarray(canvas).copy()
    H, W = a.shape[:2]
    mid = (px0 + px1) // 2
    # 字已擦掉：在色块正中那一列重新量上下边（检测时那列可能落在圆角上，量得偏矮），多留 1px 抗锯齿边
    col = a[:, mid].astype(np.int16)
    cy = (py0 + py1) // 2
    top, bot = cy, cy
    while top > 0 and np.abs(col[top - 1] - col[cy]).max() < 40:
        top -= 1
    while bot < H - 1 and np.abs(col[bot + 1] - col[cy]).max() < 40:
        bot += 1
    Y0, Y1 = max(0, min(py0, top) - 1), min(H, max(py1, bot + 1) + 1)   # 只动色块所在的行，上下的渐变底不跟着平移
    X0, X1 = max(0, px0 - 3), min(W, px1 + 3)
    dl = min(extra // 2, l_free)
    seg = a[Y0:Y1, X0:X1]
    new = np.concatenate([seg[:, :mid - X0], np.repeat(seg[:, mid - X0:mid - X0 + 1], extra, axis=1),
                          seg[:, mid - X0:]], axis=1)
    nx0 = X0 - dl
    new = new[:, max(0, -nx0):W - nx0]
    xs = max(0, nx0)
    old_seg = a[Y0:Y1, xs:xs + new.shape[1]].astype(np.float32)
    # 边缘 3px 渐变融合，色块外有纹理/光晕时不留接缝
    hh, ww = new.shape[:2]
    fy = np.minimum(np.arange(hh), np.arange(hh)[::-1])
    fx = np.minimum(np.arange(ww), np.arange(ww)[::-1])
    alpha = np.clip(np.minimum(fy[:, None], fx[None, :]) / 2.0, 0, 1)[..., None]
    a[Y0:Y1, xs:xs + ww] = (alpha * new + (1 - alpha) * old_seg).round().astype(np.uint8)
    canvas.paste(Image.fromarray(a))
    return px0 - dl, px1 - dl + extra


def draw_lines(canvas, draw, p, f, lines, step, top, W):
    """top 是第一行“行框”的顶；单行未折行时行框就是原文字框，多行时每行占 step 高。"""
    line_h = p.get('lh_eff', p['lh']) if len(lines) == 1 else step
    cx = p['cx']
    if p.get('plate'):
        (px0, py0, px1, py1), _, (l_free, _r) = p['plate']
        need = max(tw(f, s) for s in lines) + 2 * PLATE_PAD
        if need > px1 - px0:                              # 加长色块
            px0, px1 = stretch_plate(canvas, p['plate'], int(need - (px1 - px0)), l_free)
        cx = (px0 + px1) / 2
    for k, s in enumerate(lines):
        w_ = tw(f, s)
        x = {'left': p['box'][0], 'right': p['box'][2] - w_, 'center': cx - w_ / 2}[p['align']]
        x = min(max(x, MARGIN / 2), W - MARGIN / 2 - w_)
        l, t, r, b = f.getbbox(s)
        yc = top + k * step + line_h / 2
        # 落在整数像素上，避免文字被插值糊掉
        draw.text((round(x - l), round(yc - (t + b) / 2)), s, font=f, fill=p['fg'])


# ---------- 主流程 ----------

def assign_strokes(arr, boxes):
    """把所有文字框里的笔画按连通块分给各行：块的中心落在哪个框里就归哪行，落在多个框里归竖直中心最近的。
    中文行和紧挨着的英文行框常重叠，这样两行的笔画不会混在一起。返回每行的笔画掩码列表。"""
    h, w = arr.shape[:2]
    union = np.zeros((h, w), np.uint8)
    core, wide = [], []
    for b in boxes:
        bg, _ = box_background(arr, b)
        # 笔画常超出 OCR 框（首字偏旁、句末标点常被框漏掉）：竖向外扩 4px，横向外扩约 0.6 个字高
        # 只有外扩那一侧基本是同一底色时才外扩（旁边是图标、色块边就不扩，免得把图案当笔画擦掉）
        pad = max(4, int(0.6 * (b[3] - b[1])))
        c = (max(0, b[0] - 4), max(0, b[1] - 4), min(w, b[2] + 4), min(h, b[3] + 4))
        x0, x1 = c[0], c[2]
        for side in (-1, 1):
            sa, sb = (max(0, b[0] - pad), b[0]) if side < 0 else (b[2], min(w, b[2] + pad))
            strip = arr[c[1]:c[3], sa:sb]
            if strip.size and (np.abs(strip.astype(np.int16) - bg).max(axis=2) <= 45).mean() > 0.8:
                if side < 0:
                    x0 = sa
                else:
                    x1 = sb
        y0, y1 = c[1], c[3]
        core.append(c)
        wide.append((x0, y0, x1, y1))
        union[y0:y1, x0:x1] |= (np.abs(arr[y0:y1, x0:x1].astype(np.int16) - bg).max(axis=2) > 45)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(union, connectivity=8)
    masks = [np.zeros((h, w), bool) for _ in boxes]
    centers = [(b[1] + b[3]) / 2 for b in boxes]
    for k in range(1, n):
        sx, sy, sw, sh = stats[k, :4]
        cx, cy = cents[k]
        comp = labels[sy:sy + sh, sx:sx + sw] == k
        # 横向重叠、竖直方向被这个连通块跨过中心的框：说明块把上下两行的笔画连在了一起
        spanned = sorted([i for i, (x0, y0, x1, y1) in enumerate(boxes)
                          if x0 < sx + sw and x1 > sx and sy <= centers[i] < sy + sh], key=lambda i: centers[i])
        if len(spanned) >= 2:
            # 在相邻两行中心之间找笔画最少的一行切开，上半归上一行，下半归下一行
            cuts = []
            for a, b in zip(spanned, spanned[1:]):
                r0, r1 = int(centers[a]) - sy, int(centers[b]) - sy
                counts = comp[r0:r1].sum(axis=1)
                cuts.append(r0 + int(np.argmin(counts)) if len(counts) else r0)
            bounds = [0] + cuts + [sh]
            for i, (u, v) in zip(spanned, zip(bounds, bounds[1:])):
                masks[i][sy + u:sy + v, sx:sx + sw] |= comp[u:v]
            continue
        cands = [i for i, (x0, y0, x1, y1) in enumerate(boxes) if x0 <= cx < x1 and y0 <= cy < y1]
        if cands:
            i = min(cands, key=lambda i: abs(centers[i] - cy))
            x0, y0, x1, y1 = core[i]
            part = np.zeros_like(comp)
            part[max(0, y0 - sy):max(0, y1 - sy), max(0, x0 - sx):max(0, x1 - sx)] = True
            inside = comp & part
            # 块大部分在框内：整块算笔画（含伸出框外的部分）；大部分在框外：可能连着图案，只取框内部分
            masks[i][sy:sy + sh, sx:sx + sw] |= comp if comp.sum() <= 2 * inside.sum() else inside
            continue
        # 框外紧挨着的小块（句末标点等）：完全落在某行的横向外扩范围内且尺寸小于半个字，归该行
        for i, (x0, y0, x1, y1) in enumerate(wide):
            bh = boxes[i][3] - boxes[i][1]
            # 碰到外扩范围边界的块是被截断的图案（如行首色点圆圈），不算
            # 只收框右侧/框内的（句末标点）；框左边外的小块多是边框、虚线、图标
            if x0 < sx and sx + sw < x1 and y0 < sy and sy + sh < y1 and sw <= 0.42 * bh and sh <= 0.42 * bh                     and sx >= boxes[i][0] - 2 and boxes[i][1] - 1 <= sy and sy + sh <= boxes[i][3] + 1:
                masks[i][sy:sy + sh, sx:sx + sw] |= comp
                break
    return masks


def under_picture(arr, box, bg):
    """框正上方（约 3 个字高内）有一块图形，且图形的水平中心和框中心对齐。"""
    x0, y0, x1, y1 = box
    h, bw = y1 - y0, x1 - x0
    H, W = arr.shape[:2]
    a0, a1 = max(0, y0 - 3 * h), max(0, y0 - 3)
    b0, b1 = max(0, x0 - 2 * bw), min(W, x1 + 2 * bw)
    if a1 - a0 < h or b1 <= b0:
        return False
    m, _ = content_mask(arr[a0:a1, b0:b1], bg)
    cols = m.mean(axis=0) > 0.1
    c = (x0 + x1) // 2 - b0
    if not (0 <= c < len(cols)) or not cols[c]:
        return False
    l = c
    while l > 0 and cols[l - 1]:
        l -= 1
    r = c
    while r < len(cols) - 1 and cols[r + 1]:
        r += 1
    span = r - l + 1
    # 图形要有左右边界（不能是一整片背景/色带），宽度也不能远大于标签
    if l == 0 or r == len(cols) - 1:
        return False
    return 1.2 * bw < span and abs((l + r) / 2 - c) < 0.15 * span


def mark_grid(items, boxes):
    """同一行有 ≥3 段字高相近、中心等间距的文字（图标格的标签行）：标记为网格，按居中排。"""
    for it in items:
        x0, y0, x1, y1 = it['box']
        h, yc = y1 - y0, (y0 + y1) / 2
        row = sorted((b[0] + b[2]) / 2 for b in boxes
                     if abs((b[1] + b[3]) / 2 - yc) < 0.4 * h and 0.75 <= (b[3] - b[1]) / h <= 1.33)
        if len(row) >= 3:
            d = np.diff(row)
            it['grid'] = bool(d.min() > 2 * h and d.max() - d.min() < 0.12 * d.mean())


def fill_interp(arr, m, pad=12):
    """把掩码内的像素按同一行左右、同一列上下最近的未擦像素线性插值，两个方向取平均。适合纯色和平滑渐变底。"""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return
    H, W = m.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    sub = arr[y0:y1, x0:x1].astype(np.float32)
    mm = m[y0:y1, x0:x1]
    acc = np.zeros_like(sub)
    cnt = np.zeros(mm.shape, np.float32)
    for axis in (1, 0):
        S = sub if axis == 1 else sub.transpose(1, 0, 2)
        M = mm if axis == 1 else mm.T
        A = acc if axis == 1 else acc.transpose(1, 0, 2)
        C = cnt if axis == 1 else cnt.T
        for r in np.flatnonzero(M.any(axis=1)):
            row_m = M[r]
            known = np.flatnonzero(~row_m)
            if len(known) < 2:
                continue
            hole = np.flatnonzero(row_m)
            # 只用两侧都有已知像素的点（否则外推不可靠）
            ok = (hole > known[0]) & (hole < known[-1])
            hole = hole[ok]
            if not len(hole):
                continue
            for ch in range(3):
                A[r, hole, ch] += np.interp(hole, known, S[r, known, ch])
            C[r, hole] += 1
    done = mm & (cnt > 0)
    sub[done] = acc[done] / cnt[done][:, None]
    arr[y0:y1, x0:x1][done] = sub[done].round().clip(0, 255).astype(np.uint8)
    rest = mm & ~done
    if rest.any():
        arr[y0:y1, x0:x1] = cv2.inpaint(arr[y0:y1, x0:x1], rest.astype(np.uint8), 3, cv2.INPAINT_TELEA)


def icon_end(arr, box):
    """框最左边一段若是实心圆圈图标（颜色与后面文字明显不同、填充饱满），返回图标后面文字开始的 x；否则 None。"""
    x0, y0, x1, y1 = box
    h = y1 - y0
    bg, _ = box_background(arr, box)
    ink = np.abs(arr[y0:y1, x0:x1].astype(np.int16) - bg).max(axis=2) > 45
    cols = ink.any(axis=0)
    k = 0
    while k < len(cols) and not cols[k]:
        k += 1
    s = k
    while k < len(cols) and cols[k]:
        k += 1
    e = k
    if not (0.6 * h <= e - s <= 1.3 * h) or ink[:, s:e].mean() < 0.3:
        return None
    rest = ink[:, e:]
    if rest.sum() < 20:
        return None
    region = arr[y0:y1, x0:x1].astype(np.int16)
    icon_c = np.median(region[:, s:e][ink[:, s:e]], axis=0)
    text_c = np.median(region[:, e:][rest], axis=0)
    if np.abs(icon_c - text_c).max() < 60:
        return None
    return x0 + e + 1


def fit_vertical(canvas, it):
    """竖排的中文：马来文顺时针转 90°（从上往下读）。字号按竖栏宽度，长度放不下就缩小；返回 (字号, 可用的上下端)。"""
    x0, y0, x1, y1 = it['box']
    # 竖的胶囊标签：字已擦掉，沿框中线上下找胶囊两端（颜色和中间一样的范围），长度按胶囊算、两端留边
    a = np.asarray(canvas)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    col = a[:, cx].astype(np.int16)
    top, bot = cy, cy
    while top > 0 and np.abs(col[top - 1] - col[cy]).max() < 30:
        top -= 1
    while bot < a.shape[0] - 1 and np.abs(col[bot + 1] - col[cy]).max() < 30:
        bot += 1
    if np.abs(col[cy] - np.array(it['bg'])).max() < 30 and bot - top < 1.5 * (y1 - y0) and             top <= y0 and bot >= y1 - 1:                 # 胶囊要把整段字包住，照片底上量到的一小段不算
        pad = (x1 - x0) // 3
        y0, y1 = top + pad, bot - pad
    size0 = base_size(it['weight'], 0.75 * (x1 - x0))
    for scale in SCALES + (0.55, 0.5, 0.45, 0.4, 0.35):
        f = font_at(it['weight'], max(MIN_PX, size0 * scale))
        if tw(f, it['text']) <= y1 - y0:
            break
    return f.size, y0, y1


def draw_vertical(canvas, it, size, y0, y1):
    x0, x1 = it['box'][0], it['box'][2]
    f = font_at(it['weight'], size)
    l, t, r, b = f.getbbox(it['text'])
    img = Image.new('RGBA', (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - l, 2 - t), it['text'], font=f, fill=tuple(it['fg']) + (255,))
    rot = img.rotate(-90, expand=True)
    y = y0 if rot.height <= y1 - y0 else max(0, (y0 + y1) // 2 - rot.height // 2)
    canvas.paste(rot, (round((x0 + x1) / 2 - rot.width / 2), y), rot)
    return f"[{it['i']}] 竖排   {it['weight']:7s} {f.size:3d}px 1行 {it['text']}"


def draw_verticals(canvas, items):
    """竖排的几条并排（上下有重叠、彼此挨着）：统一成最小字号、多数的粗细。"""
    vs = sorted((it for it in items if it.get('vertical')), key=lambda it: it['box'][0])
    fits = [fit_vertical(canvas, it) for it in vs]
    groups, out = [], []
    for k, it in enumerate(vs):
        g = groups[-1] if groups else None
        if g:
            p = vs[g[-1]]
            w = max(it['box'][2] - it['box'][0], p['box'][2] - p['box'][0])
            if it['box'][0] - p['box'][2] < 3 * w and min(it['box'][3], p['box'][3]) > max(it['box'][1], p['box'][1]):
                g.append(k)
                continue
        groups.append([k])
    for g in groups:
        size = min(fits[k][0] for k in g)
        ws = [vs[k]['weight'] for k in g]
        weight = max(set(ws), key=ws.count)
        for k in g:
            vs[k]['weight'] = weight
            out.append(draw_vertical(canvas, vs[k], size, fits[k][1], fits[k][2]))
    return out


def process(src, ocr_lines, tr):
    arr = np.asarray(Image.open(src).convert('RGB')).copy()
    H, W = arr.shape[:2]
    boxes = [box_of(l['poly']) for l in ocr_lines]
    tr = dict(tr)
    for i, l in enumerate(ocr_lines):
        # 色点、圆圈图标常被 OCR 认成行首的“）”：框左边去掉半个字宽（括号本身很窄），图标不擦，译文也去掉它
        if re.match(r'\s*[）)]', l['text']) and not re.search(r'[（(]', l['text']):
            boxes[i][0] += int(0.45 * (boxes[i][3] - boxes[i][1]))
            if str(i) in tr:
                tr[str(i)] = re.sub(r'^\s*[）)]\s*', '', tr[str(i)])
        # 行首的实心圆圈序号（①这类彩色圆点里写数字）被 OCR 连进了文字：圆圈不擦、不写进译文
        elif re.match(r'\s*\d[一-鿿]', l['text']):
            end = icon_end(arr, boxes[i])
            if end:
                boxes[i][0] = end
                if str(i) in tr:
                    tr[str(i)] = re.sub(r'^\s*\d\s*', '', tr[str(i)])
    strokes = assign_strokes(arr, boxes)
    # 每行真实的字迹范围（横向用框，竖向用笔画），排版避让时用它代替 OCR 框
    ink_boxes = []
    for b, m in zip(boxes, strokes):
        rows = np.flatnonzero(m[b[1]:b[3], b[0]:b[2]].any(axis=1))
        ink_boxes.append([b[0], b[1] + rows[0], b[2], b[1] + rows[-1] + 1] if len(rows) else b)
    # 很淡的装饰字（底纹里的花体英文等）不算挡住排版的“其它文字”
    faint = set()
    for k, (b, m) in enumerate(zip(boxes, strokes)):
        if CJK.search(ocr_lines[k]['text']) or not m.any():
            continue
        bgk, _ = box_background(arr, b)
        if float(np.median(np.abs(arr[m].astype(np.int16) - bgk).max(axis=1))) < 60:
            faint.add(k)
    items, keep = [], []
    for i, l in enumerate(ocr_lines):
        t = tr.get(str(i))
        if t and CJK.search(l['text']):
            it = dict(i=i, text=t, zh=l['text'], box=boxes[i], seg=l.get('seg', False), **analyse(arr, boxes[i]))
            if it['seg']:
                it['plate'] = None                        # 双色胶囊的一半：只在自己这一段里擦、写，不拉长色块
            if strokes[i].any():
                it['mask'] = strokes[i]
                it['weight'] = weight_of(arr, boxes[i], it['bg'], strokes[i])
                # 横向起止用真实笔画（OCR 框常比字宽几像素），靠左排时新文字和图标的间距与原来一致
                xs = np.flatnonzero(strokes[i][boxes[i][1]:boxes[i][3]].any(axis=0))
                if len(xs) and xs[-1] - xs[0] > 0.5 * (boxes[i][2] - boxes[i][0]):
                    it['box'] = [max(boxes[i][0], int(xs[0])), boxes[i][1], min(boxes[i][2], int(xs[-1]) + 1), boxes[i][3]]
            # 弧形/斜排的字（圆形标注上的“台面增高”）墨迹高度远大于字宽：字号按字宽算，避免写得巨大
            n_cjk = len(CJK.findall(l['text']))
            char_w = (boxes[i][2] - boxes[i][0]) / max(n_cjk, 1)
            if n_cjk >= 2 and boxes[i][3] - boxes[i][1] > 2 * (boxes[i][2] - boxes[i][0]):
                it['vertical'] = True                     # 竖排的字：单独处理（马来文转 90° 写）
                if 'mask' in it:
                    # 竖条框很窄，框边一圈常落在笔画上：底色取框内非笔画像素，字色取与底色差得最多的笔画像素
                    x0_, y0_, x1_, y1_ = boxes[i]
                    sub = arr[y0_:y1_, x0_:x1_].astype(np.int16)
                    mk = cv2.dilate(it['mask'][y0_:y1_, x0_:x1_].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    band = np.zeros_like(mk)                # 只看中间一半宽：框两边可能已出了胶囊
                    q = (x1_ - x0_) // 4
                    band[:, q:x1_ - x0_ - q] = True
                    dom, share, _ = dominant(sub[band])
                    if share > 0.15 and mk.mean() > 0.6:
                        # 整个胶囊都被当成了笔画（框内底色判错）：以胶囊主色为底，胶囊内与之不同的才是笔画
                        near = (np.abs(sub - dom).max(axis=2) < 40).astype(np.uint8)
                        k = max(3, (x1_ - x0_) // 2) | 1
                        pill = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
                        new_m = np.zeros_like(it['mask'])
                        new_m[y0_:y1_, x0_:x1_] = pill & (np.abs(sub - dom).max(axis=2) > 45)
                        it['mask'] = new_m
                        mk = cv2.dilate(new_m[y0_:y1_, x0_:x1_].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    if (~mk & band).sum() > 20 and mk.any():
                        bgv = np.median(sub[~mk & band], axis=0)
                        d = np.abs(sub - bgv).max(axis=2)
                        strong = it['mask'][y0_:y1_, x0_:x1_] & (d > 0.5 * d.max())
                        if strong.any():
                            it['bg'] = tuple(int(c) for c in bgv)
                            it['fg'] = tuple(int(c) for c in np.median(sub[strong], axis=0))
            elif n_cjk >= 2 and it['ink_h'] > 1.35 * char_w:
                it['ink_h'] = int(0.8 * char_w)
            items.append(it)
        else:
            keep.append(i)
    items.sort(key=lambda it: (it['box'][1], it['box'][0]))
    mark_grid(items, boxes)
    # 尺码徽章（S号/M号/L号）：同一张图里粗细统一（有的判成粗体、有的判成常规）
    badges = [it for it in items if re.fullmatch(r'\s*[A-Za-z]{1,3}号\s*', it['zh'])]
    if len(badges) > 1:
        ws = [it['weight'] for it in badges]
        w = 'bold' if 'bold' in ws else max(set(ws), key=ws.count)
        for it in badges:
            it['weight'] = w

    # 擦字：只保护不翻译行（英文等）自己的笔画
    protect = np.zeros((H, W), bool)
    for i in keep:
        protect |= strokes[i]
    protect = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    lama_mask = np.zeros((H, W), bool)
    smooth_mask = np.zeros((H, W), bool)
    for it in items:
        x0, y0, x1, y1 = it['box']
        if 'mask' in it:                                  # 笔画可能伸出 OCR 框（首字偏旁、句末标点），擦除范围随笔画外扩
            ys, xs = np.nonzero(it['mask'])
            x0, y0, x1, y1 = min(x0, xs.min()), min(y0, ys.min()), max(x1, xs.max() + 1), max(y1, ys.max() + 1)
        # 把笔画边缘的抗锯齿半透明像素也算进来（与底色差 >20 且挨着笔画），否则修补后会留下斑点
        soft = np.zeros((H, W), bool)
        soft[y0:y1, x0:x1] = np.abs(arr[y0:y1, x0:x1].astype(np.int16) - np.array(it['bg'])).max(axis=2) > 12
        # 大字常带投影/外发光：外扩范围随字高变大（30px 字约 5px，60px 字约 9px）
        k = max(5, int(0.15 * it['ink_h']) | 1)
        core = cv2.dilate(it['mask'].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        m = cv2.dilate((core & soft | it['mask']).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        # 描边字（白字深色描边）：笔画中间的白色填充离描边较远，用闭运算把被描边围住的部分也算进来
        # 只补被掩码完全围住的“洞”，不碰旁边的边框、虚线
        sub_m = m[y0:y1, x0:x1].astype(np.uint8)
        if sub_m.any():
            inv = np.pad(1 - sub_m, 1, constant_values=1)
            ff = inv.copy()
            cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 2)
            holes = (ff == 1)[1:-1, 1:-1]
            m[y0:y1, x0:x1] |= holes & soft[y0:y1, x0:x1]
        inside = np.zeros((H, W), bool)                   # 只擦文字框附近（笔画常略超出 OCR 框），不碰更远的线条、色块边
        inside[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = True
        # 保护的是不翻译的字（英文、装饰字）；和本行字色相同的像素是压在上面的中文笔画，照样擦
        own = np.abs(arr.astype(np.int16) - np.array(it['fg'])).max(axis=2) < 50 if protect[m].any() else np.zeros_like(protect)
        m &= ~(protect & ~own) & inside
        if protect[y0:y1, x0:x1].any():
            # 和装饰字连在一起、被分给装饰字那一行的中文笔画：按字色在本行框内找回来一起擦
            m |= cv2.dilate((own & protect & inside).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & inside
        # 近似纯色底（色块、标签，允许 JPG 噪点）：直接用底色填，最干净
        ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~m & inside
        ring_px = arr[ring].astype(np.int16)
        # 插值用的“已知像素”要在字的外发光/描边之外，否则补出来的是一块发亮的方块：插值类擦法用外扩后的掩码
        k2 = max(5, int(0.18 * it['ink_h']) | 1)
        inside2 = np.zeros((H, W), bool)
        inside2[max(0, y0 - k2):y1 + k2, max(0, x0 - k2):x1 + k2] = True
        m_wide = cv2.dilate(m.astype(np.uint8), np.ones((k2, k2), np.uint8)).astype(bool) & inside2 & ~protect
        # 用 90% 分位判断：渐变底（一侧白一侧灰）平均差可能很小，整块填一个底色会留下明显色块
        # 浅色渐变底（墙面、光照）差值也可能 <12：整块填一个色会留下一块块补丁，所以还要求周围几乎没有起伏
        near_flat = len(ring_px) and float(np.percentile(np.abs(ring_px - np.array(it['bg'])).max(axis=1), 90)) < 8             and float(ring_px.std(axis=0).max()) < 3
        if it.get('seg'):
            # 双色胶囊的一半：每行只用本段内（左右两端）未擦像素插值，不碰另一半和外面的背景
            bx0, by0, bx1, by1 = it['box']
            segc = np.median(arr[by0:by1, bx0:bx1].reshape(-1, 3), axis=0)
            # 段内底色均匀：与段底色差得多的就是字（含抗锯齿边），整段一起擦
            ms = np.zeros_like(m)
            ms[by0:by1, bx0 + 1:bx1 - 1] = np.abs(arr[by0:by1, bx0 + 1:bx1 - 1].astype(np.int16) - segc).max(axis=2) > 30
            ms = cv2.dilate(ms.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ms[:, :bx0 + 1] = False
            ms[:, bx1 - 1:] = False
            for y in np.flatnonzero(ms[by0:by1].any(axis=1)) + by0:
                row = arr[y, bx0:bx1].astype(np.int16)
                # 只用本段底色的像素当参考（段边的弧形另一种颜色不算）
                keep_x = np.flatnonzero(~ms[y, bx0:bx1] & (np.abs(row - segc).max(axis=1) < 40))
                xs = np.flatnonzero(ms[y, bx0:bx1])
                if len(keep_x) >= 2 and len(xs):
                    for ch in range(3):
                        arr[y, bx0 + xs, ch] = np.interp(xs, keep_x, arr[y, bx0 + keep_x, ch].astype(float)).round()
            continue
        plate = it['plate']
        if plate:
            # 色块内部（字以外）要基本均匀；照片底上误认的“色块”按行插值会抹成一块灰
            bx0, by0, bx1, by1 = plate[0]
            inner = arr[by0:by1, bx0:bx1][~m[by0:by1, bx0:bx1]].astype(np.int16)
            if len(inner) == 0 or float(inner.std(axis=0).max()) > 12:
                plate = None
        if plate and plate[0][0] <= x0 and x1 <= plate[0][2] and not it.get('vertical'):
            # 文字在色块（常是横向渐变的标签）里：每一行在色块内按左右未擦像素线性插值，渐变连续、不留色斑
            px0, _, px1, _ = plate[0]
            mp = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~protect
            for y in np.flatnonzero(mp[:, px0:px1].any(axis=1)):
                row_m = mp[y, px0:px1]
                keep_x = np.flatnonzero(~row_m)
                if len(keep_x) >= 2:
                    xs = np.flatnonzero(row_m)
                    for ch in range(3):
                        arr[y, px0 + xs, ch] = np.interp(xs, keep_x, arr[y, px0 + keep_x, ch].astype(float)).round()
            continue
        # 框边一圈是纯色、但擦除区周围有起伏（压在照片/泡沫上的部分）：不能整块填色
        ring_flat = len(ring_px) and float(ring_px.std(axis=0).max()) < 6
        if (it['flat'] and ring_flat) or near_flat:
            # 纯色/近纯色底也按周围插值填（行、列两个方向平均），淡淡的渐变不会被一块单色补丁打断
            fill_interp(arr, m_wide)
            continue
        # 底色平滑（渐变、半透明色块）：用平滑插值，不让 LaMa 凭空生成纹理斑点；有照片纹理才用 LaMa
        ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~m
        texture = float(arr[ring].astype(np.int16).std(axis=0).mean()) if ring.any() else 99
        if texture < 12:                                  # 照片/纹理底（大理石、泡沫）用 LaMa，平滑插值会糊成一块灰
            fill_interp(arr, m_wide)                      # 平滑底：行列插值（TELEA 对粗大字会留下亮斑）
        else:
            lama_mask |= m
    if smooth_mask.any():
        arr = cv2.inpaint(arr, smooth_mask.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    if lama_mask.any():
        # LaMa（显卡）只处理需要修补的区域加 64px 上下文，整张图推理太占显存和内存
        ys, xs = np.nonzero(lama_mask)
        y0, y1 = max(0, ys.min() - 64), min(H, ys.max() + 65)
        x0, x1 = max(0, xs.min() - 64), min(W, xs.max() + 65)
        sub, sm = arr[y0:y1, x0:x1], lama_mask[y0:y1, x0:x1]
        with torch.inference_mode():
            out = np.asarray(lama()(Image.fromarray(sub), Image.fromarray(sm.astype(np.uint8) * 255))
                             .convert('RGB'))[:y1 - y0, :x1 - x0]
        sub[sm] = out[sm]

    canvas = Image.fromarray(arr)
    if DEBUG:
        canvas.save(Path(DEBUG_DIR) / ('erased_' + Path(src).name + '.png'))
    draw = ImageDraw.Draw(canvas)
    report = []
    jobs = []
    for stack in stack_groups(group_lines([it for it in items if not it.get('vertical')])):
        mine = {it['i'] for g in stack for it in g}
        others = [b for k, b in enumerate(ink_boxes) if k not in mine and k not in faint]
        plans = [plan_group(arr, g, others, W, len(stack) > 1) for g in stack]
        if len(plans) > 1:
            placed = layout_stack(arr, plans, others, H, W)
        else:
            f, lines, step = layout_single(arr, plans[0], others, H)
            placed = [(f, lines, step, None)]
        jobs.append((plans, placed, len(plans) > 1))
        if DEBUG:
            for p in plans:
                print('   plan', [it['i'] for it in p['g']], p['box'], p['align'], 'L', p['left'], 'R', p['right'],
                      'maxw', p['maxw'], 'plate', p['plate'])

    # 同一行或同一列里、同样式（粗细、颜色、原字高相近）的并列项统一成其中最小的字号
    singles = [(plans[0], placed[0]) for plans, placed, is_stack in jobs if not is_stack]

    def parallel(p, q):
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = p['box'], q['box']
        same_row = abs((ay0 + ay1) - (by0 + by1)) / 2 < 0.5 * p['lh']
        same_col = abs(ax0 - bx0) <= 8 or abs(ax1 - bx1) <= 8 or abs((ax0 + ax1) - (bx0 + bx1)) <= 16
        return same_row or same_col

    # 并列关系可传递（同一行的 A、B，B 下面同一列的 C……），按连通分组取组内最小字号
    root = list(range(len(singles)))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for a, (p, _) in enumerate(singles):
        for b, (q, _) in enumerate(singles[:a]):
            if abs(q['ink_h'] - p['ink_h']) <= 0.15 * p['ink_h'] and \
                    max(abs(x - y) for x, y in zip(q['fg'], p['fg'])) < 40 and parallel(p, q) and                     bool(q['plate']) == bool(p['plate']) and                     max(abs(x - y) for x, y in zip(q['bg'], p['bg'])) < 40:   # 色块标签和普通文字不互相拉小
                root[find(a)] = find(b)
    sizes, votes = {}, {}
    for a, (p, (f, *_r)) in enumerate(singles):
        sizes.setdefault(find(a), []).append(f.size)
        votes.setdefault(find(a), []).append(p['weight'])
    # 组里个别特别长、被迫缩得很小的（如规格表里一条长说明）不拖累整组：只按不小于中位数 75% 的那些取最小
    group_min = {}
    for r, ss in sizes.items():
        med = sorted(ss)[len(ss) // 2]
        group_min[r] = min(x for x in ss if x >= 0.75 * med) if len(ss) >= 3 else min(ss)
    for a, (p, (f, *_r)) in enumerate(singles):
        p['size'] = min(f.size, group_min[find(a)])
        v = votes[find(a)]                                # 小字的粗细容易判错：并列项按多数统一粗细
        p['weight'] = max(set(v), key=v.count)
    for plans, placed, is_stack in jobs:
        if not is_stack:
            p = plans[0]
            f, lines, step, _ = placed[0]
            if p['size'] < f.size or p['weight'] != p['g'][0]['weight']:
                f = font_at(p['weight'], min(p['size'], f.size))
                lines = wrap(f, p['text'], p['maxw']) if len(lines) > 1 else [p['text']]
                step = f.size * 1.25 if len(lines) > 1 else step
            block = p['lh'] if len(lines) == 1 else step * len(lines)
            top = (p['box'][1] + p['box'][3]) / 2 - block / 2 if len(lines) <= len(p['g']) else p['box'][1]
            placed = [(f, lines, step, top)]
        for p, (f, lines, step, top) in zip(plans, placed):
            draw_lines(canvas, draw, p, f, lines, step, top, W)
            report.append(f"{[it['i'] for it in p['g']]} {p['align']:6s} {p['weight']:7s} {f.size:3d}px "
                          f"{len(lines)}行{' 叠' if len(plans) > 1 else ''} {p['text']}")
    report += draw_verticals(canvas, items)
    return canvas, report


if __name__ == '__main__':
    DEBUG = '--debug' in sys.argv
    sys.argv = [a for a in sys.argv if a != '--debug']
    src_dir, work, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    ocr = json.loads((work / 'ocr.json').read_text(encoding='utf-8'))
    trans = json.loads((work / 'translations.json').read_text(encoding='utf-8'))
    for name in sorted(ocr):
        canvas, rep = process(src_dir / name, ocr[name], trans.get(name, {}))
        canvas.save(out_dir / name, quality=95, subsampling=0)   # 4:4:4，彩色小字边缘不糊
        print('==', name)
        for r in rep:
            print('  ', r)
