pdflatex -interaction=nonstopmode main.tex > $null
bibtex main > $null
pdflatex -interaction=nonstopmode main.tex > $null
pdflatex -interaction=nonstopmode main.tex | Select-String -Pattern "^! |Undefined|undefined|Output written"
$env:PYTHONIOENCODING = 'utf-8'
python -c @"
import pymupdf
from PIL import Image
d = pymupdf.open('main.pdf')
for i, p in enumerate(d):
    t = ' '.join(p.get_text().split())
    ks = [k for k in ['INTRODUCTION','RELATED WORK','WHY A 16','METHOD','EVALUATION PROTOCOL','EXPERIMENTS','ABLATIONS','OTHER RESOLUTIONS','LIMITATIONS','CONCLUSION','AI USE STATEMENT','REFERENCES'] if k in t.upper()]
    print(i + 1, ks, '|', t[-80:])
ims = []
for i in range(min(11, len(d))):
    pix = d[i].get_pixmap(dpi=45)
    ims.append(Image.frombytes('RGB', [pix.width, pix.height], pix.samples))
w, h = ims[0].size
s = Image.new('RGB', (w * 6, h * 2), 'white')
for i, im in enumerate(ims):
    s.paste(im, ((i % 6) * w, (i // 6) * h))
s.save('_pages.png')
"@
