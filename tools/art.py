"""Cut the STATHIS artwork (chest logo, back panther) out of the owner's mockup as white ink with transparency,
so the page can print it on any garment in any colour.  usage: python3 tools/art.py <mockup.png> img/"""
import sys, numpy as np, cv2
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).astype(np.float32); H, W, _ = a.shape
lum = a.mean(2)
def cut(x0, x1, rows, name):
    p = lum[:, x0:x1]
    # the paper background is the bright area connected to the picture's border; ink is bright but enclosed by cloth
    n, cc = cv2.connectedComponents((p > 110).astype(np.uint8))
    border = set(np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))) - {0}
    m = (p > 110) & ~np.isin(cc, list(border)); m[:rows[0]] = False; m[rows[1] + 1:] = False
    paper = cv2.dilate(np.isin(cc, list(border)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.where(m); pad = 6
    y0, y1, xa, xb = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    al = np.clip((p[y0:y1, xa:xb] - 24) / (235 - 24), 0, 1) * ~paper[y0:y1, xa:xb]          # black cloth = 0, white ink = 1
    im = Image.fromarray((al * 255).astype(np.uint8), 'L').resize(((xb - xa) * 3, (y1 - y0) * 3), Image.LANCZOS)
    rgba = Image.merge('RGBA', [Image.new('L', im.size, 255)] * 3 + [im]); rgba.save(out + name, optimize=True)
    print(name, rgba.size, 'centre x in shirt', (xa + xb) / 2 / (x1 - x0), 'width share', (xb - xa) / (x1 - x0))
cut(0, W // 2, (400, 560), 'art-logo.png')
cut(W // 2, W, (320, 700), 'art-panther.png')
