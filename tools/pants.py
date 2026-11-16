"""Draw the shading pictures for joggers and shorts (front and back) that the page colours and prints on.

There is no product photo of the trousers yet, so these are illustrated: a grey picture where 200 means
"the fabric's own colour", darker is shadow, lighter is highlight, with the garment's outline in the alpha
channel. Same format as the photo-based img/tee-front.png etc. Replace with real photos when they exist.

Run from the repo root: python3 tools/pants.py && python3 tools/webp.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, K = 1024, 1280, 2          # output size; K = output px per frame unit (the page's frame is 512x640)
SS = 2                           # supersampling for smooth outlines
rng = np.random.default_rng(7)


def box(a, r, axis):
    r = max(1, int(round(r)))
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(a, r):
    """Gaussian-like blur: three box passes each way."""
    a = a.astype(np.float32)
    for _ in range(3):
        a = box(box(a, r * .58, 0), r * .58, 1)
    return a


def noise(r, amp):
    n = rng.random((H, W)).astype(np.float32)
    n = blur(n, r)
    n = (n - n.mean()) / (n.std() + 1e-6)
    return n * amp


def outline(kind):
    """Right half (as seen) of the garment, frame units; mirrored for the left."""
    if kind == 'pants':
        outer = [(374, 34), (376, 68), (390, 120), (394, 170), (390, 260), (380, 380), (366, 480), (354, 574), (348, 580), (346, 622)]
        inner = [(294, 622), (292, 580), (287, 574), (279, 480), (268, 380), (256, 268)]
    else:
        outer = [(374, 34), (376, 68), (390, 120), (395, 170), (396, 250), (398, 338)]
        inner = [(268, 344), (256, 262)]
    return outer + inner


def mask_of(kind):
    im = Image.new('L', (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    right = outline(kind)
    left = [(512 - x, y) for x, y in reversed(right)]
    pts = [(x * K * SS, y * K * SS) for x, y in right + left]
    d.polygon(pts, fill=255)
    im = im.filter(ImageFilter.GaussianBlur(3 * SS)).point(lambda v: 255 if v > 127 else 0)   # round the corners a little
    im = im.resize((W, H), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255


def seg_dist(X, Y, p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / (dx * dx + dy * dy + 1e-6), 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    side = np.sign((X - px) * dy - (Y - py) * dx)
    return np.hypot(X - px, Y - py), side, t


def build(kind, back):
    m = mask_of(kind)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    X /= K
    Y /= K                         # frame units from here on
    S = np.ones((H, W), np.float32)

    # each leg is a soft tube: darker towards its edges, a broad highlight a little left of its middle
    inside = m > .5
    T = np.ones((H, W), np.float32)
    for y in range(H):
        row = inside[y]
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        cuts = np.flatnonzero(np.diff(xs) > 1)
        runs = np.split(xs, cuts + 1)
        for r in runs:
            a, b = r[0], r[-1]
            if b - a < 6:
                continue
            t = (np.arange(a, b + 1) - (a + b) / 2) / ((b - a) / 2)
            wide = 1.0 if len(runs) > 1 else .55     # the hips are flatter than a leg
            T[y, a:b + 1] = 1 - .17 * wide * np.abs(t) ** 2.4 + .035 * np.exp(-((t + .28) / .4) ** 2)

    for _ in range(2):                       # no hard step where the hips split into two legs
        T = box(T, 22, 0)
    S *= T

    def fold(p0, p1, width, amp, fade=True):
        """A crease: a dark valley with a light ridge beside it, fading towards its end."""
        d, side, t = seg_dist(X, Y, p0, p1)
        f = (1 - t * .8) if fade else 1.0
        S[:] += amp * f * (-np.exp(-(d / width) ** 2) + .7 * np.exp(-((d - side * width * 1.6) / (width * 1.3)) ** 2) * (side > 0))

    cy = 264 if kind == 'pants' else 258
    for sx in (-1, 1):
        cx = 256
        # whiskers from the crotch towards the hips
        for ang, ln, amp in ((18, 96, .075), (34, 104, .06), (52, 90, .05), (8, 70, .045)):
            a = np.radians(ang)
            fold((cx + sx * 6, cy - 4), (cx + sx * (6 + ln * np.cos(a)), cy - 4 - ln * np.sin(a)), 4.2, amp)
        # gathers under the waistband
        for gx in (150, 176, 204, 232):
            x0 = gx if sx < 0 else 512 - gx
            fold((x0, 70), (x0 + sx * 3, 104 + (gx % 3) * 8), 2.6, .05)
        if kind == 'pants':
            legx = 256 + sx * 66
            # knee: a few slanted creases
            for dy, amp in ((0, .06), (16, .045), (-14, .04)):
                fold((legx - sx * 44, 404 + dy + 10), (legx + sx * 40, 396 + dy - 8), 4.6, amp, fade=False)
            # the leg stacks above the cuff
            for dy, amp in ((0, .085), (-15, .07), (-31, .05), (-46, .035)):
                fold((256 + sx * 36, 566 + dy + 3), (256 + sx * 100, 566 + dy - 3), 3.6, amp, fade=False)
            # one long soft fold down the shin
            fold((legx + sx * 8, 300), (legx - sx * 2, 560), 9, .035, fade=False)
        else:
            legx = 256 + sx * 72
            fold((legx - sx * 20, 190), (legx + sx * 6, 330), 9, .04, fade=False)
            fold((legx + sx * 34, 210), (legx + sx * 44, 330), 6, .035, fade=False)

    # cloth is never even
    S += noise(46, .018) + noise(14, .012) + noise(3, .008)

    def line(p0, p1, w, dark, light=.035):
        d, side, _ = seg_dist(X, Y, p0, p1)
        S[:] *= 1 - dark * np.exp(-(d / w) ** 2)
        S[:] += light * np.exp(-((d - 2.2 * w) / (1.4 * w)) ** 2) * (side > 0)

    # ribbed waistband (and cuffs), with the seam under it
    rib = 1 + .045 * np.sin(X * 2 * np.pi / 4.6)
    band = (Y > 30) & (Y < 68)
    S[band] *= rib[band] * .985
    line((120, 68.5), (392, 68.5), 1.5, .2)
    for yy in (44, 56):                      # elastic channels
        line((130, yy), (382, yy), .9, .07, .012)
    if kind == 'pants':
        cuff = Y > 578
        S[cuff] *= rib[cuff] * .97
        for sx in (-1, 1):
            line((256 + sx * 30, 578), (256 + sx * 100, 578), 1.6, .22)
    else:
        for sx in (-1, 1):                   # stitched hem
            x0, x1 = 256 + sx * 10, 256 + sx * 146
            line((x0, 322 + (2 if sx else 0)), (x1, 318), .9, .13, .02)
            line((x0, 327), (x1, 323), .8, .08, .0)

    if not back:
        line((256, 70), (256, cy - 6), 1.1, .1, .02)                         # centre seam
        for sx in (-1, 1):                                                     # slanted pocket openings
            line((256 + sx * 104, 72), (256 + sx * 132, 168), 1.5, .2, .05)
        # drawstring: two cords from the eyelets, with their shadows
        for sx, ln in ((-1, 92), (1, 80)):
            x0 = 256 + sx * 13
            pts = [(x0, 52), (x0 + sx * 5, 52 + ln * .5), (x0 + sx * 2, 52 + ln)]
            for p0, p1 in zip(pts, pts[1:]):
                d, _, _ = seg_dist(X, Y, (p0[0] + 2.2, p0[1] + 1.5), (p1[0] + 2.2, p1[1] + 1.5))
                S[:] *= 1 - .22 * np.exp(-(d / 2.4) ** 2)
            for p0, p1 in zip(pts, pts[1:]):
                d, _, _ = seg_dist(X, Y, p0, p1)
                core = np.exp(-(d / 1.9) ** 4)
                S[:] = S * (1 - core) + core * (1.13 - .1 * (d / 1.9) ** 2)
            d = np.hypot(X - x0, Y - 51)                                        # eyelet
            S[:] *= 1 - .5 * np.exp(-(d / 2.6) ** 4)
            d, _, _ = seg_dist(X, Y, (pts[-1][0], pts[-1][1]), (pts[-1][0], pts[-1][1] + 9))   # knotted tip
            core = np.exp(-(d / 2.6) ** 4)
            S[:] = S * (1 - core) + core * 1.05
    else:
        line((256, 70), (256, cy - 4), 1.2, .14, .03)                         # seat seam
        for sx in (-1, 1):                                                     # yoke
            line((256, 104), (256 + sx * 134, 84), 1.1, .1, .02)
        # one patch pocket (on the wearer's right, so on the right as seen from behind)
        x0, y0, w, h = 300, 118, 66, 74
        for p0, p1 in (((x0, y0), (x0 + w, y0)), ((x0, y0), (x0 + 2, y0 + h - 8)), ((x0 + w, y0), (x0 + w - 2, y0 + h - 8)),
                       ((x0 + 2, y0 + h - 8), (x0 + w / 2, y0 + h)), ((x0 + w - 2, y0 + h - 8), (x0 + w / 2, y0 + h))):
            line(p0, p1, 1.1, .14, .03)

    # the cloth rolls away at the outline
    edge = blur(m, 7)
    S *= .8 + .2 * np.clip((edge - .5) * 2.2, 0, 1) ** .7

    lum = np.clip(S * 200, 0, 255).astype(np.uint8)
    alpha = (m * 255).astype(np.uint8)
    name = f"img/{kind}-{'back' if back else 'front'}.png"
    Image.merge('LA', (Image.fromarray(lum), Image.fromarray(alpha))).save(name, optimize=True)
    print(name)


if __name__ == '__main__':
    for kind in ('pants', 'shorts'):
        for back in (False, True):
            build(kind, back)
