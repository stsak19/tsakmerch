"""Make the .webp copies the page loads from the .png sources in img/ (about 7x smaller).
Run from the repo root after changing any PNG: python3 tools/webp.py"""
import glob
from PIL import Image
for f in sorted(glob.glob('img/*.png')):
    im = Image.open(f).convert('RGBA')
    if f.endswith('art-panther.png'):  # drawn at most ~420px wide
        im = im.resize((520, round(520 * im.height / im.width)), Image.LANCZOS)
    im.save(f[:-4] + '.webp', 'WEBP', quality=90, method=6, exact=True)
