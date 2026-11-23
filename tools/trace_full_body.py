"""High-precision tracing of the full-body figure in reference.jpg.

Pipeline (numpy + Pillow only):
  1. crop the figure, upscale x4 (Lanczos) and lightly denoise
  2. k-means colour quantisation; near-white paper = background
  3. majority filter on the label map to drop JPEG speckles
  4. stacked layers: layer k covers every pixel whose label rank >= k, so
     the painted result reproduces the label map pixel-for-pixel, no gaps
  5. trace pixel-edge contours, smooth, simplify (RDP) and fit Catmull-Rom
     cubic Beziers -> SVG paths in the original image coordinates
"""
import sys, json
import numpy as np
from PIL import Image, ImageFilter

SRC = sys.argv[1] if len(sys.argv) > 1 else "reference.jpg"
OUT = sys.argv[2] if len(sys.argv) > 2 else "full_body_paths.json"
X0, Y0, X1, Y1 = 700, 40, 922, 516   # crop box in original px
S = 4                                # upscale factor
K = 40                               # colours
rng = np.random.default_rng(1)

im = Image.open(SRC).convert("RGB").crop((X0, Y0, X1, Y1))
im = im.resize(((X1 - X0) * S, (Y1 - Y0) * S), Image.LANCZOS)
im = im.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.0))
a = np.asarray(im).astype(np.float32)
H, W, _ = a.shape

mn, mx = a.min(2), a.max(2)
bg_like = (mn >= 214) & (mx - mn <= 42)
bg_like[int((488 - Y0) * S):] = ((mn >= 236) & (mx - mn <= 14))[int((488 - Y0) * S):]
# background = paper-coloured pixels connected to the crop border
# (white inside the figure, e.g. the eye whites, stays foreground)
from collections import deque
bg = np.zeros((H, W), bool)
q = deque()
for y in range(H):
    for x in (0, W - 1):
        if bg_like[y, x] and not bg[y, x]: bg[y, x] = True; q.append((y, x))
for x in range(W):
    for y in (0, H - 1):
        if bg_like[y, x] and not bg[y, x]: bg[y, x] = True; q.append((y, x))
while q:
    y, x = q.popleft()
    for ny, nx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
        if 0 <= ny < H and 0 <= nx < W and bg_like[ny, nx] and not bg[ny, nx]:
            bg[ny, nx] = True; q.append((ny, nx))

# k-means on foreground pixels
fg_px = a[~bg]
sample = fg_px[rng.choice(len(fg_px), 60000, replace=False)]
# k-means++ init so small but important colours (eyes, mouth) get centres
C = [sample[rng.integers(len(sample))]]
for _ in range(K - 1):
    d = np.min([((sample - c) ** 2).sum(1) for c in C], axis=0)
    C.append(sample[rng.choice(len(sample), p=d / d.sum())])
C = np.array(C)
for _ in range(25):
    d = ((sample[:, None, :] - C[None]) ** 2).sum(2)
    lab = d.argmin(1)
    for k in range(K):
        m = lab == k
        if m.any():
            C[k] = sample[m].mean(0)

# merge near-identical centres: they only create speckled patches
merged = True
while merged:
    merged = False
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            if np.sqrt(((C[i] - C[j]) ** 2).sum()) < 15:
                C[i] = (C[i] + C[j]) / 2; C = np.delete(C, j, 0); merged = True; break
        if merged: break
K = len(C)
print("colours after merge:", K, file=sys.stderr)

flat = a.reshape(-1, 3)
labels = np.empty(len(flat), np.int16)
for i in range(0, len(flat), 200000):
    d = ((flat[i:i + 200000, None, :] - C[None]) ** 2).sum(2)
    labels[i:i + 200000] = d.argmin(1)
labels = labels.reshape(H, W)
dist = np.sqrt(((a - C[labels]) ** 2).sum(2))
labels[bg] = K  # background label
dist[bg] = 0

def box(m, r):
    c = np.pad(m.astype(np.int32), ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    return c[2*r+1:, 2*r+1:] - c[:-2*r-1, 2*r+1:] - c[2*r+1:, :-2*r-1] + c[:-2*r-1, :-2*r-1]

# anti-aliased / blended pixels (far from every centre, or light pixels on
# the silhouette edge) take the majority label of confident neighbours
def resolve(labels, unsure, r):
    best = np.full((H, W), -1, np.int32); arg = labels.copy()
    for k in range(K + 1):
        cnt = box((labels == k) & ~unsure, r)
        upd = cnt > best
        best[upd] = cnt[upd]; arg[upd] = k
    out = labels.copy(); fix = unsure & (best > 0)
    out[fix] = arg[fix]
    return out, unsure & ~fix

near_bg = box(bg, 10) > 0
light = C[np.minimum(labels, K - 1)].min(2) > 185
light[int((490 - Y0) * S):] = False   # keep the floor shadow intact
lum0 = a @ np.array([0.299, 0.587, 0.114], np.float32)
g0y, g0x = np.gradient(lum0)
# transition pixels between two colours (anti-aliasing) are split between
# the colours on either side instead of forming rings of in-between colour
transition = np.hypot(g0x, g0y) > 9
unsure = ((dist > 20) | (near_bg & light)) & ~bg
for r in (2, 4, 8, 16):
    if not unsure.any(): break
    labels, unsure = resolve(labels, unsure, r)

for _ in range(2):  # majority filter
    best = np.full((H, W), -1, np.int32); arg = labels.copy()
    for k in range(K + 1):
        cnt = box(labels == k, 3)
        upd = cnt > best
        best[upd] = cnt[upd]; arg[upd] = k
    labels = arg

# "mixture" colours: thin bands whose colour is a blend of two neighbouring
# colours are anti-aliasing, not paint. Split each band between the two
# colours it blends (by blend ratio) so edges meet cleanly with no ring.
for _ in range(3):
    edge_k = np.zeros((H, W), bool)
    edge_k[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge_k[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge_k[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge_k[:-1, :] |= labels[1:, :] != labels[:-1, :]
    changed = False
    for k in range(K):
        m = labels == k
        n = m.sum()
        if n == 0: continue
        thin = (edge_k & m).sum() / n
        if thin < 0.55: continue
        # which colours touch this band?
        ring = (box(m, 2) > 0) & ~m
        nb = np.bincount(labels[ring], minlength=K + 1)
        nb[k] = 0
        cand = [j for j in np.argsort(-nb)[:4] if nb[j] > 0]
        best = None
        for i1 in range(len(cand)):
            for i2 in range(i1 + 1, len(cand)):
                A = C[cand[i1]] if cand[i1] < K else np.array([253, 252, 249.])
                B = C[cand[i2]] if cand[i2] < K else np.array([253, 252, 249.])
                ab = B - A
                t = float(np.clip(((C[k] - A) @ ab) / max(ab @ ab, 1e-6), 0, 1))
                err = np.sqrt(((A + t * ab - C[k]) ** 2).sum())
                if 0.1 < t < 0.9 and (best is None or err < best[0]):
                    best = (err, cand[i1], cand[i2], A, ab)
        if best is None or best[0] > 14: continue
        _, j1, j2, A, ab = best
        ys, xs = np.nonzero(m)
        t = ((a[ys, xs] - A) @ ab) / (ab @ ab)
        labels[ys, xs] = np.where(t < 0.5, j1, j2)
        changed = True
    if not changed: break

areas = [(labels == k).sum() for k in range(K)]
order = [k for k in sorted(range(K), key=lambda k: -areas[k]) if areas[k] > 0]
rank = np.full(K + 1, -1, np.int32)
for i, k in enumerate(order):
    rank[k] = i
R = rank[labels]  # -1 = background

def trace(mask):
    """Return closed loops of lattice points; fg on the right-hand side."""
    m = np.pad(mask, 1)
    out = {}
    def add(sx, sy, ex, ey):
        out.setdefault((sx, sy), []).append((ex, ey))
    # horizontal edges: pixel above (y-1) vs below (y), lattice y in padded coords
    dh = m[1:, :].astype(np.int8) - m[:-1, :].astype(np.int8)   # +1: fg below
    ys, xs = np.nonzero(dh == 1)
    for y, x in zip(ys + 1, xs): add(x, y, x + 1, y)
    ys, xs = np.nonzero(dh == -1)
    for y, x in zip(ys + 1, xs): add(x + 1, y, x, y)
    dv = m[:, 1:].astype(np.int8) - m[:, :-1].astype(np.int8)   # +1: fg right
    ys, xs = np.nonzero(dv == 1)
    for y, x in zip(ys, xs + 1): add(x, y + 1, x, y)
    ys, xs = np.nonzero(dv == -1)
    for y, x in zip(ys, xs + 1): add(x, y, x, y + 1)
    loops = []
    while out:
        start = next(iter(out))
        pts = [start]; cur = start; prev_dir = None
        while True:
            nxts = out[cur]
            if len(nxts) == 1 or prev_dir is None:
                nxt = nxts.pop(0)
            else:  # ambiguous corner: prefer right turn
                def turn(p):
                    d = (p[0] - cur[0], p[1] - cur[1])
                    return 0 if d == (-prev_dir[1], prev_dir[0]) else 1
                nxts.sort(key=turn); nxt = nxts.pop(0)
            if not nxts: del out[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start: break
            pts.append(cur)
        loops.append(np.array(pts, np.float64) - 1)  # undo padding
    return loops

def rdp(p, eps):
    if len(p) < 3: return p
    a, b = p[0], p[-1]
    ab = b - a; n = np.hypot(*ab)
    if n == 0: d = np.hypot(*(p - a).T)
    else: d = np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n
    i = d.argmax()
    if d[i] > eps:
        return np.vstack([rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)])
    return np.vstack([a, b])

def smooth_loop(p, it=3):
    for _ in range(it):
        p = (np.roll(p, 2, 0) + np.roll(p, 1, 0) + p + np.roll(p, -1, 0) + np.roll(p, -2, 0)) / 5
    return p

def to_path(p):
    # split closed loop at farthest pair to run RDP, then Catmull-Rom -> Bezier
    j = np.argmax(np.hypot(*(p - p[0]).T))
    s = np.vstack([rdp(np.vstack([p[:j + 1]]), 0.55)[:-1], rdp(np.vstack([p[j:], p[:1]]), 0.55)[:-1]])
    n = len(s)
    if n < 3: return ""
    f = lambda v: f"{X0 + v[0] / S:.2f},{Y0 + v[1] / S:.2f}".replace(".00", "")
    d = [f"M{f(s[0])}"]
    for i in range(n):
        p0, p1, p2, p3 = s[i - 1], s[i], s[(i + 1) % n], s[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6; c2 = p2 - (p3 - p1) / 6
        d.append(f"C{f(c1)} {f(c2)} {f(p2)}")
    return "".join(d) + "Z"

# edge softness per colour: mean luminance gradient on the colour's border.
# Soft-edged colours (cheek blush, shading) get a small blur in the SVG.
lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
gy, gx = np.gradient(lum)
grad = np.hypot(gx, gy)
edge = np.zeros((H, W), bool)
edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
edge[1:, :] |= labels[1:, :] != labels[:-1, :]
soft = {}
for k in range(K):
    m = edge & (labels == k)
    soft[k] = float(grad[m].mean()) if m.any() else 99.0

from PIL import ImageDraw

hexc = lambda v: "#%02x%02x%02x" % tuple(int(np.clip(round(c), 0, 255)) for c in v)
gpt = lambda v: [round(X0 + v[0] / S, 2), round(Y0 + v[1] / S, 2)]
# pixels that really carry the colour (not anti-aliased blends) drive the fills
cdist = np.sqrt(((a - C[np.minimum(labels, K - 1)]) ** 2).sum(2))
trusted = (cdist < 16) & (labels < K)

def inside(poly, x, y):
    xs, ys = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    c = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-12) + xs)
    return c.sum() % 2 == 1

def fit_fill(pm, k):
    """Colour for one region: a linear gradient fitted to the original pixels
    of colour k inside the region, or their mean colour."""
    sel = pm & (labels == k)
    ys, xs = np.nonzero(sel & trusted)
    if len(xs) < 30:
        ys, xs = np.nonzero(sel)
    if len(xs) == 0:
        return hexc(C[k]), None
    px = a[ys, xs].astype(np.float64)
    if len(xs) < 300:
        return hexc(px.mean(0)), None
    lumk = px @ np.array([0.299, 0.587, 0.114])
    M = np.column_stack([np.ones(len(xs)), xs, ys]).astype(np.float64)
    dvec = np.linalg.lstsq(M, lumk, rcond=None)[0][1:]
    n = np.hypot(*dvec)
    if n < 1e-6:
        return hexc(px.mean(0)), None
    dvec = dvec / n
    t = xs * dvec[0] + ys * dvec[1]
    t0, t1 = np.percentile(t, 2), np.percentile(t, 98)
    if t1 - t0 < 4:
        return hexc(px.mean(0)), None
    T = np.column_stack([np.ones(len(t)), (t - t0) / (t1 - t0)])
    cc = np.linalg.lstsq(T, px, rcond=None)[0]
    ca, cb = cc[0], cc[0] + cc[1]
    if np.abs(cb - ca).max() < 5:
        return hexc(px.mean(0)), None
    cx, cy = xs.mean(), ys.mean()
    proj = cx * dvec[0] + cy * dvec[1]
    p0 = np.array([cx, cy]) + dvec * (t0 - proj)
    p1 = np.array([cx, cy]) + dvec * (t1 - proj)
    return hexc(C[k]), {"from": gpt(p0), "to": gpt(p1), "c0": hexc(ca), "c1": hexc(cb)}

layers = []
for i, k in enumerate(order):
    loops = []
    for lp in trace(R >= i):
        if len(lp) < 12: continue
        area = 0.5 * np.sum(lp[:, 0] * np.roll(lp[:, 1], -1) - np.roll(lp[:, 0], -1) * lp[:, 1])
        if abs(area) < 20: continue
        loops.append((area, lp))
    outers = [lp for ar, lp in loops if ar > 0]
    holes = [lp for ar, lp in loops if ar < 0]
    owner = [[] for _ in outers]
    order_o = sorted(range(len(outers)), key=lambda j: abs(loops[0][0]) and len(outers[j]))
    for h in holes:
        hx, hy = h[0] + 0.5
        best = None
        for j, o in enumerate(outers):
            x0, y0 = o.min(0); x1, y1 = o.max(0)
            if x0 <= hx <= x1 and y0 <= hy <= y1 and inside(o, hx, hy):
                sz = (x1 - x0) * (y1 - y0)
                if best is None or sz < best[0]: best = (sz, j)
        if best: owner[best[1]].append(h)
    for j, o in enumerate(outers):
        x0, y0 = np.floor(o.min(0)).astype(int); x1, y1 = np.ceil(o.max(0)).astype(int)
        img = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1), 0)
        dr = ImageDraw.Draw(img)
        dr.polygon([tuple(v) for v in (o - [x0, y0])], fill=1)
        for h in owner[j]:
            dr.polygon([tuple(v) for v in (h - [x0, y0])], fill=0)
        pm = np.zeros((H, W), bool)
        sub = np.asarray(img, bool)
        yy0, xx0 = max(y0, 0), max(x0, 0)
        yy1, xx1 = min(y1 + 1, H), min(x1 + 1, W)
        pm[yy0:yy1, xx0:xx1] = sub[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0]
        fill, grad = fit_fill(pm, k)
        d = to_path(smooth_loop(o)) + "".join(to_path(smooth_loop(h)) for h in owner[j])
        layers.append({"fill": fill, "grad": grad, "d": d, "base": i == 0, "edge": 99, "area": int(sub.sum())})
    print(i, hexc(C[k]), areas[k], len(outers), len(holes), file=sys.stderr)

json.dump(layers, open(OUT, "w"))
