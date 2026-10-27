"""商详图切点算法：只在版块边界切，每段尽量是一个完整版块（不把照片、标题和内容拆开）。

切点代价：
- 两块照片/内容直接相接：只有贯穿整行的水平分界才便宜。
- 纯色底区域（标题、说明文字、参数表所在的底色区）里：只能切在空白间隙里，
  区域内最宽的间隙最便宜，越窄越贵（窄的多半只是行距、图注间距），
  这样标题会跟着下面的内容、图注会跟着上面的照片。
再用动态规划在“切点代价 + 段长偏离 3:4 的代价”之间取最优。
"""
import numpy as np

TARGET = 4 / 3      # 理想段高/宽（3:4 画布）
MIN_RATIO = 0.45    # 段高下限（只有一段时不受限）
MAX_RATIO = 2.4     # 段高硬上限：单个版块再长才允许在内容里切
STEP = 2
LINE = 8            # 不超过这么多行的异色细条（分隔线、下划线）不打断纯色区域
CONTENT = 5.0       # 切在内容中间的代价
TITLE_MIN, TITLE_MAX = 0.032, 0.12   # 标题字高范围（相对图宽）：比图注、正文大，比图块小


def runs(mask):
    """mask 中连续为 True 的区间列表 [(start, end), ...]。"""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def row_features(gray):
    h, w = gray.shape
    q = (gray // 16).astype(np.int64)
    counts = np.zeros((h, 16), dtype=np.int64)
    for v in range(16):
        counts[:, v] = (q == v).sum(axis=1)
    cover = counts.max(axis=1) / w        # 主色占比
    dom = counts.argmax(axis=1)           # 主色
    flat = gray.std(axis=1) + np.abs(np.diff(gray, axis=1)).mean(axis=1) * 2
    blank = flat < 1.0                    # 设计稿里的空白行几乎完全平；照片里的墙面会有细微起伏（≥1.4），不算空白
    return cover, dom, blank


def plain_regions(cover, dom):
    """纯色底区域：连续、主色相同、主色占比 > 0.6 的行；细分隔线不打断。返回 [(a, b), ...]。"""
    h = len(cover)
    plain = cover > 0.6
    regions = []
    a, last_dom, last_y = None, -1, -100
    for y in range(h):
        if not plain[y]:
            continue
        if a is not None and (dom[y] == last_dom or _short_run(dom, plain, y)) and y - last_y <= LINE + 1:
            if dom[y] == last_dom:
                last_y = y
            else:
                last_y = y
            continue
        if a is not None:
            regions.append((a, last_y + 1))
        a, last_dom, last_y = y, dom[y], y
    if a is not None:
        regions.append((a, last_y + 1))
    return regions


def _short_run(dom, plain, y):
    """y 开始的同主色纯色行不超过 LINE 行（是细分隔线）。"""
    v, n = dom[y], 0
    while y + n < len(dom) and plain[y + n] and dom[y + n] == v and n <= LINE:
        n += 1
    return n <= LINE


def boundary_cost(arr):
    """cost[y]：在第 y 行上方切开的代价。"""
    gray = arr.mean(axis=2)
    h, w = gray.shape
    cover, dom, blank = row_features(gray)
    cost = np.full(h, CONTENT)

    # 1) 内容与内容直接相接：贯穿整行的水平分界
    d = np.abs(np.diff(gray, axis=0))
    frac = (d > 12).mean(axis=1)
    edge = 1 - np.clip((frac - 0.55) / 0.35, 0, 1)      # 占比 ≥90% → 0
    cost[1:] = np.where(edge < 0.35, edge, CONTENT)

    # 2) 纯色底区域：只在空白间隙里切，最宽的间隙最便宜
    for a, b in plain_regions(cover, dom):
        gaps = runs(blank[a:b])
        if not gaps:                                     # 没有空白行（照片里偶然的纯色条）：不提供切点
            continue
        if blank[a:b].all():                             # 整段空白：哪里切都行
            cost[a:b + 1] = 0.0
            continue
        widest = max(v - u for u, v in gaps)
        inks = runs(~blank[a:b])                         # 区域内的文字/图块（行区间）
        for u, v in gaps:
            length = v - u
            c = 0.0 if length >= widest * 0.9 else 0.25 + 0.6 * (1 - length / widest)
            if length < w * 0.02:                        # 极窄的缝：基本是行距
                c = max(c, 1.0)
            # 间隙上方不远处有大字（标题）：标题属于下面的内容，不能切在它下面
            if any(iv <= u and u - iv < w * 0.15 and TITLE_MIN * w <= iv - iu <= TITLE_MAX * w
                   for iu, iv in inks):
                c += 0.9
            # 间隙紧挨着下方的大字（标题）：这是版块开头，优先在这里切
            elif any(iu >= v and iu - v < w * 0.02 and TITLE_MIN * w <= iv - iu <= TITLE_MAX * w
                     for iu, iv in inks):
                c = min(c, 0.1)
            cost[a + u:a + v + 1] = c
    return cost


def find_cuts(arr):
    h, w = arr.shape[:2]
    tgt, min_h, max_h = w * TARGET, int(w * MIN_RATIO), int(w * MAX_RATIO)
    if h <= tgt * 1.15:
        return [0, h]
    cost = boundary_cost(arr)
    pos = list(range(0, h, STEP)) + [h]
    pos_a = np.array(pos)
    n = len(pos)
    best = np.full(n, np.inf)
    prev = np.full(n, -1)
    best[0] = 0
    for j in range(1, n):
        cut = 0.0 if j == n - 1 else float(cost[pos[j]])
        lo = np.searchsorted(pos_a, pos[j] - max_h)
        hi = np.searchsorted(pos_a, pos[j] - min_h, side='right')
        if hi <= lo:
            continue
        r = (pos[j] - pos_a[lo:hi]) / tgt
        # 段长偏离 3:4 的代价：偏短轻罚，略长（≤1.2 倍）很轻，再长逐渐加重；每段固定 0.3 避免切太碎
        shape = np.where(r < 1, (1 - r) * 0.8,
                         np.where(r < 1.2, (r - 1) * 0.5, 0.1 + (r - 1.2) * 1.5)) + 0.3
        cand = best[lo:hi] + shape
        i = int(np.argmin(cand))
        if np.isfinite(cand[i]):
            best[j] = cand[i] + cut
            prev[j] = lo + i
    cuts, j = [], n - 1
    while j > 0:
        cuts.append(pos[j])
        j = prev[j]
    return [0] + cuts[::-1]
