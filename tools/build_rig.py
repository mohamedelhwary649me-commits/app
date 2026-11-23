"""Builds character-rig.html: the full-body figure from reference.jpg as clean,
separate, animatable parts.

Every part's outline comes from the reference image (colour classes cut by
hand-placed part regions), then the mask is smoothed and the contour fitted
with a few smooth Bezier curves: one sharp, clean edge per part, no halos.
Each part gets one flat colour from a small palette.
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC = "reference.jpg"
OUT = "character-rig.html"
X0, Y0, X1, Y1 = 700, 40, 922, 516
S = 4
W, H = (X1 - X0) * S, (Y1 - Y0) * S

im = Image.open(SRC).convert("RGB").crop((X0, Y0, X1, Y1)).resize((W, H), Image.LANCZOS)
img = np.asarray(im.filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)

# ---------------------------------------------------------------- classes
PROTO = {
    "hair": (0x22, 0x26, 0x34), "hairhi": (0x3a, 0x41, 0x52),
    "cap": (0x84, 0xb1, 0xba), "capdark": (0x6b, 0x9a, 0xa7),
    "skin": (0xee, 0x9d, 0x6f), "skinsh": (0xdc, 0x80, 0x5c),
    "white": (0xf1, 0xf1, 0xf1), "mouth": (0xb4, 0x5a, 0x3a),
    "robe": (0xc9, 0x65, 0x3a), "robesh": (0xab, 0x49, 0x30),
    "robedark": (0x90, 0x3c, 0x28), "gold": (0xe2, 0xa0, 0x66),
    "sandal": (0x86, 0x3c, 0x2e), "floor": (0xef, 0xdd, 0xd3),
}
names = list(PROTO)
P = np.array([PROTO[n] for n in names], np.float32)
d = ((img[:, :, None, :] - P[None, None]) ** 2).sum(3)
cls = d.argmin(2)
mn, mx = img.min(2), img.max(2)
bg_like = (mn >= 228) & (mx - mn <= 22)

# background = paper connected to the border (eye whites stay inside)
def components(mask):
    """Connected components (4-neighbour) via run-length union-find."""
    parent = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    runs_prev, lab = [], np.zeros(mask.shape, np.int32)
    allruns = []
    for y in range(mask.shape[0]):
        row = mask[y]
        dif = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        starts, ends = np.nonzero(dif == 1)[0], np.nonzero(dif == -1)[0]
        runs = []
        j = 0
        for s, e in zip(starts, ends):
            rid = len(parent); parent.append(rid)
            while j < len(runs_prev) and runs_prev[j][1] <= s: j += 1
            k = j
            while k < len(runs_prev) and runs_prev[k][0] < e:
                a, b = find(rid), find(runs_prev[k][2])
                if a != b: parent[max(a, b)] = min(a, b)
                k += 1
            runs.append((s, e, rid)); allruns.append((y, s, e, rid))
        runs_prev = runs
    for y, s, e, rid in allruns:
        lab[y, s:e] = find(rid) + 1
    return lab

bl = components(bg_like)
border = set(np.unique(np.concatenate([bl[0], bl[-1], bl[:, 0], bl[:, -1]]))) - {0}
bg = np.isin(bl, list(border))
fg = ~bg

def C(*n):
    return np.isin(cls, [names.index(x) for x in n])

# ---------------------------------------------------------------- geometry helpers
def poly(pts):
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).polygon([((x - X0) * S, (y - Y0) * S) for x, y in pts], fill=1)
    return np.asarray(m, bool)

def ellipse(cx, cy, rx, ry):
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).ellipse([((cx - rx - X0) * S, (cy - ry - Y0) * S), ((cx + rx - X0) * S, (cy + ry - Y0) * S)], fill=1)
    return np.asarray(m, bool)

def superellipse(cx, cy, rx, ry, n):
    yy, xx = np.mgrid[0:H, 0:W]
    x = (X0 + (xx + 0.5) / S - cx) / rx
    y = (Y0 + (yy + 0.5) / S - cy) / ry
    return np.abs(x) ** n + np.abs(y) ** n <= 1

def shift_up(m, k):
    out = np.zeros_like(m); out[:-k] = m[k:]; return out

def box(x0, y0, x1, y1):
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

def fill_holes(m):
    lab = components(~m)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return m | ~np.isin(lab, list(edge))

def keep(m, min_area=0, largest=False):
    lab = components(m)
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    if len(ids) == 0: return m & False
    if largest:
        return lab == ids[cnt.argmax()]
    return np.isin(lab, ids[cnt >= min_area])

def smooth_mask(m, r):
    if r <= 0: return m
    im_ = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im_) >= 128

def close(m, r):
    """Morphological closing (bridges thin anti-aliasing gaps)."""
    d = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))) >= 30
    return np.asarray(Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))) >= 225

def trace(mask):
    m = np.pad(mask, 1)
    out = {}
    def add(sx, sy, ex, ey): out.setdefault((sx, sy), []).append((ex, ey))
    dh = m[1:, :].astype(np.int8) - m[:-1, :].astype(np.int8)
    for y, x in zip(*np.nonzero(dh == 1)): add(x, y + 1, x + 1, y + 1)
    for y, x in zip(*np.nonzero(dh == -1)): add(x + 1, y + 1, x, y + 1)
    dv = m[:, 1:].astype(np.int8) - m[:, :-1].astype(np.int8)
    for y, x in zip(*np.nonzero(dv == 1)): add(x + 1, y + 1, x + 1, y)
    for y, x in zip(*np.nonzero(dv == -1)): add(x + 1, y, x + 1, y + 1)
    loops = []
    while out:
        start = next(iter(out)); pts = [start]; cur = start; pd = None
        while True:
            nx = out[cur]
            if len(nx) > 1 and pd is not None:
                nx.sort(key=lambda p: 0 if (p[0] - cur[0], p[1] - cur[1]) == (-pd[1], pd[0]) else 1)
            n = nx.pop(0)
            if not nx: del out[cur]
            pd = (n[0] - cur[0], n[1] - cur[1]); cur = n
            if cur == start: break
            pts.append(cur)
        loops.append(np.array(pts, float) - 1)
    return loops

def gsmooth(p, sigma):
    n = len(p)
    r = int(3 * sigma)
    if sigma <= 0 or n < 2 * r + 3: return p
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pp = np.vstack([p[-r:], p, p[:r]])
    return np.column_stack([np.convolve(pp[:, 0], k, "valid"), np.convolve(pp[:, 1], k, "valid")])

def rdp(p, eps):
    if len(p) < 3: return p
    a, b = p[0], p[-1]; ab = b - a; n = np.hypot(*ab)
    dd = np.hypot(*(p - a).T) if n == 0 else np.abs(ab[0] * (p[:, 1] - a[1]) - ab[1] * (p[:, 0] - a[0])) / n
    i = dd.argmax()
    if dd[i] > eps:
        return np.vstack([rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)])
    return np.vstack([a, b])

def f(v):
    return f"{X0 + v[0] / S:.2f},{Y0 + v[1] / S:.2f}"

def loop_path(p, sigma, eps):
    p = gsmooth(p, sigma)
    j = np.argmax(np.hypot(*(p - p[0]).T))
    s = np.vstack([rdp(p[:j + 1], eps)[:-1], rdp(np.vstack([p[j:], p[:1]]), eps)[:-1]])
    n = len(s)
    if n < 3: return ""
    out = [f"M{f(s[0])}"]
    for i in range(n):
        p0, p1, p2, p3 = s[i - 1], s[i], s[(i + 1) % n], s[(i + 2) % n]
        out.append(f"C{f(p1 + (p2 - p0) / 6)} {f(p2 - (p3 - p1) / 6)} {f(p2)}")
    return "".join(out) + "Z"

def path(mask, blur=4, sigma=4, eps=1.6, min_loop=60):
    """Mask -> one clean SVG path (outer contours + holes)."""
    m = smooth_mask(mask, blur)
    parts = []
    for lp in trace(m):
        area = 0.5 * np.sum(lp[:, 0] * np.roll(lp[:, 1], -1) - np.roll(lp[:, 0], -1) * lp[:, 1])
        if abs(area) < min_loop: continue
        parts.append(loop_path(lp, sigma, eps))
    return "".join(parts)

# ---------------------------------------------------------------- palette
def colour(mask, *n):
    sel = mask & C(*n) if n else mask
    if sel.sum() < 10: sel = mask
    v = np.median(img[sel], axis=0)
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in v)

svg = []  # (indent-free) svg lines
def el(id_, d, fill, extra=""):
    if d:
        svg.append(f'<path id="{id_}" fill="{fill}" d="{d}"{extra}/>')

# ---------------------------------------------------------------- regions (original px)
head_zone = box(745, 44, 868, 176)
cap_zone = box(752, 46, 862, 92)
ear_L = poly([(750, 118), (771, 118), (772, 156), (750, 156)])
ear_R = poly([(843, 118), (864, 118), (864, 156), (842, 156)])
face_zone = poly([(764, 96), (850, 96), (850, 175), (764, 175)])
eye_L, eye_R = box(770, 119, 797, 140), box(812, 119, 841, 140)
brow_L, brow_R = box(768, 104, 796, 118), box(812, 104, 841, 118)
nose_z, mouth_z = box(795, 136, 812, 148), box(786, 145, 824, 157)
neck_z = box(784, 158, 828, 184)
torso_poly = poly([(758, 172), (856, 172), (864, 250), (877, 480), (736, 480), (750, 250)])
sleeve_L = poly([(764, 186), (742, 205), (712, 312), (712, 326), (752, 326), (760, 300), (763, 222)])
sleeve_R = poly([(850, 186), (872, 205), (902, 312), (902, 326), (862, 326), (854, 300), (851, 222)])
cuff_L = poly([(712, 298), (754, 306), (752, 327), (712, 318)])
cuff_R = poly([(902, 298), (860, 306), (862, 327), (902, 318)])
hand_L, hand_R = box(708, 318, 748, 366), box(866, 318, 906, 366)
foot_L, foot_R = box(724, 463, 792, 506), box(824, 463, 892, 506)
collar_z = poly([(784, 168), (830, 168), (832, 196), (782, 196)])
placket_z = poly([(795, 186), (819, 186), (819, 256), (807, 268), (795, 256)])

# ---------------------------------------------------------------- floor shadow (ellipse fit)
fl = C("floor") & fg & box(705, 488, 918, 514) & ~foot_L & ~foot_R | (C("floor") & fg & box(705, 496, 918, 514))
ys, xs = np.nonzero(fl)
cx, cy = xs.mean(), ys.mean()
rx = (np.percentile(xs, 99.5) - np.percentile(xs, 0.5)) / 2
ry = (np.percentile(ys, 99.5) - np.percentile(ys, 0.5)) / 2
svg.append(f'<g id="floor-shadow"><ellipse cx="{X0 + cx / S:.2f}" cy="{Y0 + cy / S:.2f}" rx="{rx / S:.2f}" ry="{ry / S:.2f}" fill="{colour(fl)}"/></g>')

# ---------------------------------------------------------------- feet
for side, zone in (("left", foot_L), ("right", foot_R)):
    sk = keep(fg & zone & C("skin", "skinsh", "white", "floor") & ~C("floor") | (fg & zone & C("skin", "skinsh")), largest=True)
    sk = fill_holes(sk)
    sd = keep(fg & zone & C("sandal", "robedark", "mouth", "robesh"), min_area=200)
    svg.append(f'<g id="foot-{side}">')
    el(f"foot-{side}-skin", path(sk, 4, 5, 1.6), colour(sk, "skin"))
    el(f"sandal-{side}", path(sd, 3, 4, 1.4), "#8a3d2c")
    svg.append("</g>")

# ---------------------------------------------------------------- body
body = fill_holes(keep(fg & torso_poly & ~C("skin", "skinsh"), largest=True))
# under the sleeves the torso gets a clean straight side, so moving an arm
# never reveals a ragged edge
row = body[int((338 - Y0) * S)]
bxl, bxr = X0 + np.nonzero(row)[0].min() / S, X0 + np.nonzero(row)[0].max() / S
upper = poly([(790, 173), (822, 173), (842, 182), (853, 192), (858, 210), (bxr, 338), (bxl, 338), (754, 210), (759, 192), (770, 182)])
body = (body & ~box(700, 40, 922, 337)) | (body & upper) | (upper & ~box(700, 40, 922, 192))
body_full = body  # under the sleeves
neck = fill_holes(keep(fg & neck_z & C("skin", "skinsh", "white"), largest=True)) | box(791, 150, 821, 184)
lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
lum_s = np.asarray(Image.fromarray(np.clip(lum, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24)), np.float32)
svg.append('<g id="body">')
el("neck", path(neck, 4, 5, 1.6), colour(neck, "skin"))
el("neck-shade", "M791,166 C798,173 814,173 821,166 L821,176 C814,181 798,181 791,176Z", "#dd8862")
robe_m = body_full & ~box(700, 40, 922, 174) | (body & box(700, 172, 922, 190)) | box(786, 174, 826, 192)
el("robe", path(robe_m, 8, 10, 1.6), "#c8643a")
# one smooth shadow area: where the original robe is clearly darker (bottom / right)
# shadow side: a strip along the robe's right edge, as in the original
# shadow side: a band along the robe's right edge that widens toward the hem.
# The edge is measured per row and fitted with a smooth curve.
rows = np.arange(int((300 - Y0) * S), int((470 - Y0) * S), 4)
xr = np.array([np.nonzero(body[y])[0].max() for y in rows])
k = np.polyfit(rows, xr, 2)
yy = np.linspace(rows[0], rows[-1], 24)
outer = [(np.polyval(k, y) + 8, y) for y in yy]
inner = [(np.polyval(k, y) - S * (3 + 33 * ((y - yy[0]) / (yy[-1] - yy[0])) ** 1.8), y) for y in yy[::-1]]
shade_poly = Image.new("1", (W, H), 0)
ImageDraw.Draw(shade_poly).polygon(outer + inner, fill=1)
shade = body & np.asarray(shade_poly, bool)
el("robe-shade", path(shade, 6, 10, 1.6), "#b44f31")
# fold lines, measured on the reference
svg.append('<path id="robe-folds" d="M788.5,354 C797,382 809,411 821.5,439 M812,402 C815,408.5 818.5,415 821.5,421" fill="none" stroke="#97402a" stroke-width="1.3" stroke-linecap="round"/>')
svg.append('''<g id="collar">
<path id="collar-left" fill="#c4683f" d="M786.5,170.5 C792,174.5 799,177 805,178 L805,190.5 C797.5,188.5 790.5,185 785.5,180.5 Z"/>
<path id="collar-right" fill="#bb5f39" d="M823.5,170.5 C818,174.5 811,177 805,178 L805,190.5 C812.5,188.5 819.5,185 824.5,180.5 Z"/>
<path fill="none" stroke="#e6a86e" stroke-width="1.1" stroke-dasharray="2.2 1.4" d="M787.5,174.5 C793,178 798.5,180 803,181 M822.5,174.5 C817,178 811.5,180 807,181"/>
</g>
<g id="placket">
<path fill="#c96f40" d="M797.5,186 L814.5,186 L814.5,254 L806,263 L797.5,254 Z"/>
<path fill="none" stroke="#e6a86e" stroke-width="1.1" stroke-dasharray="2.2 1.4" d="M799.5,188 L799.5,253 L806,260 L812.5,253 L812.5,188"/>
<path fill="#8e3a26" d="M804.3,186 L807.7,186 L807.7,249 L806,251.5 L804.3,249 Z"/>
</g>''')
svg.append("</g>")

# ---------------------------------------------------------------- arms
for side, sl, cf, hz, pivot in (("left", sleeve_L, cuff_L, hand_L, (760, 196)), ("right", sleeve_R, cuff_R, hand_R, (854, 196))):
    s = fill_holes(keep(fg & sl & ~C("skin", "skinsh"), largest=True))
    hand = fill_holes(keep(fg & hz & C("skin", "skinsh", "floor", "white") & ~s, largest=True))
    wrist = (min(np.nonzero(hand)[1]) / S + X0, 320)
    svg.append(f'<g id="arm-{side}" data-pivot="{pivot[0]},{pivot[1]}">')
    svg.append(f'<g id="hand-{side}" data-pivot="{(X0 + np.nonzero(hand)[1].mean() / S):.1f},322">')
    el(f"hand-{side}-skin", path(hand | (hz & box(0, 316, 2000, 324) & s), 4, 5, 1.4), colour(hand, "skin"))
    svg.append("</g>")
    el(f"sleeve-{side}", path(s, 5, 6, 1.6), "#c8643a" if side == "left" else "#bd5a35")
    cuff = fill_holes(keep(s & cf, largest=True))
    el(f"cuff-{side}", path(cuff, 2, 3, 1.2), "#b9592f")
    if side == "left":
        trim = "M716.5,305 C727,309 737,311.5 750,312.5 M715.5,313.5 C726,317 736,319.5 749,320.5"
    else:
        trim = "M897.5,305 C887,309 877,311.5 864,312.5 M898.5,313.5 C888,317 878,319.5 865,320.5"
    svg.append(f'<path id="cuff-{side}-trim" d="{trim}" fill="none" stroke="#e6a86e" stroke-width="1.1" stroke-dasharray="2.2 1.4"/>')
    svg.append("</g>")

# ---------------------------------------------------------------- head
hair_all = fg & head_zone & C("hair", "hairhi")
hair = fill_holes(keep(hair_all, largest=True))
cap = fill_holes(keep(fg & cap_zone & ~C("hair", "hairhi") & ~box(745, 84, 760, 95), largest=True))
FACE = superellipse(807.5, 126, 40, 43, 2.4)
face = fill_holes(keep(fg & ~hair & ~cap & (FACE | (box(762, 92, 852, 124) & ~C("hair", "hairhi"))), largest=True))
face_full = face | (FACE & hair) | (superellipse(807.5, 112, 40, 44, 2.2) & hair)
ear_L = ellipse(761.5, 136, 10.5, 14.5); ear_R = ellipse(851, 136, 10, 14.5)
earL = fill_holes(keep(fg & ear_L & ~hair & ~FACE & C("skin", "skinsh"), largest=True))
earR = fill_holes(keep(fg & ear_R & ~hair & ~FACE & C("skin", "skinsh"), largest=True))
earL = earL | (smooth_mask(earL, 10) & FACE) | (ear_L & FACE & box(765, 124, 778, 148))
earR = earR | (smooth_mask(earR, 10) & FACE) | (ear_R & FACE & box(837, 124, 850, 148))
svg.append('<g id="head" data-pivot="806,168">')
# skin-coloured underlay so no paper ever shows between head parts
head_base = fill_holes(close(face_full | hair | earL | earR, 2.5)) & ~cap
el("head-base", path(head_base & ~box(700, 40, 922, 92) | (head_base & hair), 6, 8, 2.0), "#ec9a6c")
for side, e in (("left", earL), ("right", earR)):
    svg.append(f'<g id="ear-{side}">')
    el(f"ear-{side}-skin", path(e | (hair & (ear_L if side == "left" else ear_R)), 3, 4, 1.2), colour(e, "skin"))
    el(f"ear-{side}-inner", path(keep(e & C("skinsh", "mouth"), min_area=40), 3, 4, 1.2), "#d9785a")
    svg.append("</g>")
el("face", path(face_full, 10, 16, 1.6), "#ee9d6f")
el("face-shade", path(keep(face & C("skinsh") & box(764, 96, 850, 118), min_area=150), 4, 5, 1.6), "#de8760")
svg.append('<g id="cheeks">')
for cx_, cy_ in ((779.5, 143.5), (832, 142.5)):
    svg.append(f'<ellipse cx="{cx_}" cy="{cy_}" rx="8.5" ry="5.5" fill="#e88565" opacity=".55"/>')
svg.append("</g>")
el("nose", path(keep(face & nose_z & C("skinsh", "mouth"), min_area=10), 2, 3, 1.0, 10), "#d97c5a")
el("nose-tip", path(keep(face & nose_z & C("white"), min_area=4), 1, 1, 0.6, 4), "#fbe0cf")
el("mouth", path(keep(face & mouth_z & C("mouth", "robedark", "robesh"), largest=True), 2, 2.5, 0.8, 10), "#b25a3c")
# eyes: measured on the reference at 6x (white = ellipse, iris = circle,
# shine = small circle, lid = one smooth arc) - clean shapes that blink well
EYES = {"left": dict(c=(783.6, 129.2), r=(10.6, 8.3), iris=(785.2, 131.0, 6.6), shine=(781.0, 128.4, 1.9)),
        "right": dict(c=(826.4, 129.0), r=(11.4, 8.3), iris=(821.4, 130.6, 6.6), shine=(820.0, 128.3, 1.9))}
for side, bz in (("left", brow_L), ("right", brow_R)):
    e = EYES[side]; (cx, cy), (rx, ry) = e["c"], e["r"]
    ix, iy, ir = e["iris"]; sx, sy, sr = e["shine"]
    svg.append(f'<g id="eye-{side}" data-pivot="{cx},{cy}">')
    svg.append(f'<clipPath id="eye-{side}-clip"><ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}"/></clipPath>')
    svg.append(f'<ellipse id="eye-{side}-white" cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#f7f5f2"/>')
    svg.append(f'<g clip-path="url(#eye-{side}-clip)"><circle id="eye-{side}-iris" cx="{ix}" cy="{iy}" r="{ir}" fill="#1e2536"/>'
               f'<circle cx="{ix}" cy="{iy + 1.2}" r="{ir * 0.55}" fill="#2b3550"/></g>')
    svg.append(f'<circle id="eye-{side}-shine" cx="{sx}" cy="{sy}" r="{sr}" fill="#ffffff"/>')
    svg.append(f'<path id="eye-{side}-lid" d="M{cx - rx - 0.6},{cy + 1.2} C{cx - rx + 1},{cy - ry - 1.6} {cx + rx - 1},{cy - ry - 1.6} {cx + rx + 0.6},{cy + 0.4}" '
               f'fill="none" stroke="#1b1a22" stroke-width="2.3" stroke-linecap="round"/>')
    svg.append("</g>")
    el(f"brow-{side}", path(keep(fg & bz & C("hair", "hairhi"), largest=True), 3, 4, 1.0), "#232633")
svg.append('<g id="hair">')
el("hair-base", path(hair, 4, 5, 1.6), "#222634")
el("hair-strands", path(keep(hair & C("hairhi") & ~smooth_mask(~hair, 3), min_area=120), 3, 4, 1.4), "#394055")
svg.append("</g>")
svg.append('<g id="cap">')
cap = smooth_mask(cap, 6)
el("cap-base", path(cap, 4, 6, 1.6), "#84b1ba")
# the embroidered band follows the cap's lower edge (10 px tall)
band = cap & ~shift_up(cap, 10 * S)
el("cap-band", path(band, 3, 5, 1.4), "#709fac")
# zigzag embroidery along the band's centre line
cols = np.nonzero(band.any(0))[0]
bottoms = np.array([np.nonzero(cap[:, x])[0].max() for x in cols])
coef = np.polyfit(cols, bottoms - 5 * S, 3)
xl, xr = cols.min(), cols.max()
zz, up = [], True
for x in np.arange(xl + 14, xr - 14, 12.0):
    y = np.polyval(coef, x) + (-6 if up else 6)
    zz.append(f"{X0 + x / S:.2f},{Y0 + y / S:.2f}"); up = not up
svg.append(f'<path id="cap-zigzag" d="M{" L".join(zz)}" fill="none" stroke="#a7cad2" stroke-width="1" stroke-linejoin="round"/>')
svg.append("</g>")
svg.append("</g>")

# arms are drawn before the body's front so sleeves sit over the robe edge;
# reorder: floor, feet, arms(back), body, head  -> sleeves over torso sides
def block(start):
    i = next(j for j, l in enumerate(svg) if l.startswith(start))
    depth, j = 0, i
    while True:
        l = svg[j]
        depth += l.count("<g") - l.count("</g>")
        j += 1
        if depth == 0: break
    return svg[i:j]
order = ['<g id="floor-shadow"', '<g id="foot-left"', '<g id="foot-right"', '<g id="body"',
         '<g id="arm-left"', '<g id="arm-right"', '<g id="head"']
final = [l for o in order for l in block(o)]
art = "\n".join(final)

html = f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>شخصية قابلة للتحريك</title>
<style>
  :root {{ --bg:#efe9df; --panel:#fff; --ink:#3a2a1e; --muted:#7a6a5c; --accent:#c4613a; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }} }}
  :root[data-theme="dark"] {{ --bg:#1d1a17; --panel:#2a2622; --ink:#f1e6d8; --muted:#b8a998; --accent:#e07a3e; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI",Tahoma,sans-serif; }}
  main {{ max-width:720px; margin:0 auto; padding:24px 16px 40px; }}
  h1 {{ font-size:1.4rem; margin:0 0 6px; }}
  p {{ margin:0 0 16px; color:var(--muted); }}
  .controls {{ display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; background:var(--panel); padding:12px 16px; border-radius:10px; margin-bottom:16px; }}
  label {{ display:flex; gap:8px; align-items:center; }}
  button {{ font:inherit; padding:8px 16px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }}
  .sheet {{ background:#fff; border-radius:14px; box-shadow:0 6px 24px rgba(0,0,0,.12); overflow:hidden; }}
  .sheet svg {{ display:block; width:100%; height:auto; }}
  svg [data-pivot] {{ transform-box: view-box; }}
  .animate #head {{ animation: nod 2.4s ease-in-out infinite; }}
  .animate #arm-left {{ animation: swingL 2.4s ease-in-out infinite; }}
  .animate #arm-right {{ animation: swingR 2.4s ease-in-out infinite; }}
  .animate #eye-left, .animate #eye-right {{ animation: blink 3.2s infinite; transform-box: fill-box; transform-origin: center; }}
  @keyframes nod {{ 50% {{ transform: rotate(-5deg); }} }}
  @keyframes swingL {{ 50% {{ transform: rotate(8deg); }} }}
  @keyframes swingR {{ 50% {{ transform: rotate(-8deg); }} }}
  @keyframes blink {{ 0%, 92%, 100% {{ transform: scaleY(1); }} 96% {{ transform: scaleY(.1); }} }}
</style>
</head>
<body>
<main>
  <h1>الشخصية — أجزاء منفصلة قابلة للتحريك</h1>
  <p>كل عضو مجموعة مستقلة لها نقطة ارتكاز: الرأس، العينان، الحاجبان، الشعر، الطاقية، الأذنان، الذراعان، اليدان، القدمان، الجسم.</p>
  <div class="controls">
    <label><input type="checkbox" id="showRef"> إظهار الصورة الأصلية</label>
    <label>شفافية الرسم <input type="range" id="vecOp" min="0" max="100" value="100"></label>
    <label><input type="checkbox" id="anim"> تحريك تجريبي</label>
    <button id="dl">تنزيل ملف SVG</button>
  </div>
  <div class="sheet">
<svg id="art" xmlns="http://www.w3.org/2000/svg" viewBox="700 40 222 476" role="img" aria-label="ولد بطاقية زرقاء وثوب برتقالي واقف">
<image id="ref" href="reference.jpg" x="0" y="0" width="1024" height="559" style="display:none"/>
<rect id="paper" x="700" y="40" width="222" height="476" fill="#fefefd"/>
<g id="character">
{art}
</g>
</svg>
  </div>
</main>
<script>
  // rotation pivots come from each group's data-pivot attribute
  document.querySelectorAll('[data-pivot]').forEach(function (g) {{
    var p = g.getAttribute('data-pivot').split(',');
    g.style.transformOrigin = p[0] + 'px ' + p[1] + 'px';
  }});
  var ref = document.getElementById('ref'), vec = document.getElementById('character'), paper = document.getElementById('paper');
  function sync() {{
    var on = document.getElementById('showRef').checked;
    ref.style.display = on ? '' : 'none';
    paper.style.display = on ? 'none' : '';
    vec.style.opacity = document.getElementById('vecOp').value / 100;
    document.querySelector('.sheet').classList.toggle('animate', document.getElementById('anim').checked);
  }}
  ['showRef', 'vecOp', 'anim'].forEach(function (id) {{
    document.getElementById(id).addEventListener('input', sync);
    document.getElementById(id).addEventListener('change', sync);
  }});
  document.getElementById('dl').addEventListener('click', function () {{
    var svg = document.getElementById('art').cloneNode(true);
    svg.querySelector('#ref').remove();
    svg.querySelector('#paper').remove();
    svg.querySelector('#character').style.opacity = '';
    svg.querySelectorAll('[style]').forEach(function (n) {{ if (n.id !== 'character') n.removeAttribute('style'); }});
    var blob = new Blob([svg.outerHTML], {{ type: 'image/svg+xml' }});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'character-rig.svg'; a.click();
    URL.revokeObjectURL(a.href);
  }});
</script>
</body>
</html>
"""
open(OUT, "w", encoding="utf-8").write(html)
print("written", OUT, len(html) // 1024, "KB")
