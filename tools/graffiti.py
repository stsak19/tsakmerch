"""Turn the owner's snake-and-dagger logo (white on black) into spray paint for the shop wall:
only the paint is kept (as transparency), with soft overspray around it, a grainy edge where the
wall shows through, and a few drips. Output: img/graffiti.webp (white paint + alpha; the page tints it).

Run from the repo root: PYTHONPATH=tools python3 tools/graffiti.py img/src/graffiti-logo.png"""
import sys
import numpy as np
from PIL import Image
from pants import blur

src = Image.open(sys.argv[1]).convert('L')
a = np.asarray(src, dtype=np.float32) / 255
ys, xs = np.where(a > .5)
M = 90
a = np.pad(a, M)[ys.min():ys.max() + 2 * M + 130, xs.min():xs.max() + 2 * M]      # room for halo and drips
H, W = a.shape
rng = np.random.default_rng(3)

# drips: thin runs of paint from a few low points of the shape
solid = a > .5
drips = np.zeros_like(a)
for fx, ln, w in ((.40, 110, 3.2), (.585, 70, 2.6), (.30, 46, 2.2), (.66, 150, 3.0), (.52, 95, 2.4)):
    x = int(W * fx)
    col = np.flatnonzero(solid[:, x])
    if not len(col):
        continue
    y0 = col.max()
    yy = np.arange(H)[:, None].astype(np.float32)
    xx = np.arange(W)[None, :].astype(np.float32)
    t = np.clip((yy - y0) / ln, 0, 1)
    width = w * (1 - .55 * t) + 1.6 * np.exp(-((yy - (y0 + ln)) / 5) ** 2)             # a bead at the end
    drips = np.maximum(drips, ((yy >= y0 - 6) & (yy <= y0 + ln + 4)) * np.exp(-((xx - x) / width) ** 2))

paint = np.maximum(blur(a, 1.6), drips * .92)
halo = blur(np.maximum(a, drips * .5), 16) * .34 + blur(a, 44) * .16                  # overspray mist
grain = rng.random((H, W)).astype(np.float32)
wallgrain = .8 + .2 * blur(rng.random((H, W)).astype(np.float32), 1.2) / .5           # plaster shows through a little
speck = (grain > .55).astype(np.float32)                                              # mist is made of droplets
alpha = np.clip(np.maximum(paint * (.86 + .14 * grain) * np.clip(wallgrain, 0, 1.05), halo * (.35 + .65 * speck)), 0, 1)
# uneven coverage: a spray can never lays a flat coat
alpha *= .88 + .12 * np.clip(blur(rng.random((H, W)).astype(np.float32), 30) * 2, 0, 1)

out = np.zeros((H, W, 4), np.uint8)
out[..., :3] = 255
out[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
im = Image.fromarray(out, 'RGBA')
im = im.resize((560, round(560 * H / W)), Image.LANCZOS)
im.save('img/graffiti.webp', 'WEBP', quality=88, method=6, exact=True)
print(im.size)
