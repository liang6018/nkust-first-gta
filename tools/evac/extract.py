"""Extract floor-plan structure from NKUST evacuation PDFs.
Per page: walls (black strokes), label words (tesseract), spaces (raster segmentation),
stairs (parallel tread hatching), voids (X boxes), icons (stair exits, WC, evacuation arrows)."""
import sys, json, math, re, glob, os
import numpy as np, cv2, pdfplumber, pypdfium2 as pdfium, pytesseract

K = 2.0          # raster px per pdf unit
OCR_S = 7        # render scale for OCR

def col(c):
    if c is None: return None
    if isinstance(c, (int, float)): return (float(c),) * 3
    c = tuple(float(v) for v in c)
    return c if len(c) == 3 else (c[0],) * 3 if len(c) == 1 else c[:3]

def polyl(o):
    if o['object_type'] == 'rect':
        x0, t, x1, b = o['x0'], o['top'], o['x1'], o['bottom']; return [(x0, t), (x1, t), (x1, b), (x0, b), (x0, t)]
    return [(float(x), float(y)) for x, y in o['pts']]

def page_objs(p):
    out = []
    for o in p.lines + p.curves + p.rects:
        out.append({'pts': polyl(o), 'sc': col(o.get('stroking_color')), 'fc': col(o.get('non_stroking_color')), 'stroke': bool(o.get('stroke')), 'fill': bool(o.get('fill')),
                    'lw': float(o.get('linewidth') or 0), 'b': (o['x0'], o['top'], o['x1'], o['bottom'])})
    return out

def near(c, ref, tol=.05): return c is not None and all(abs(a - b) < tol for a, b in zip(c, ref))

def process(pdfpath, pi, doc=None, pdf=None):
    p = pdf.pages[pi]; objs = page_objs(p)
    # legend top: red legend circle (lw 1.8) at left
    legend = [o['b'][1] for o in objs if o['stroke'] and near(o['sc'], (1, 0, 0)) and abs(o['lw'] - 1.8) < .1 and o['b'][0] < 130 and o['b'][1] > 400]
    lt = min(legend) - 6 if legend else 670
    R = (16, 160, 826, lt + 4)
    def inR(b): return b[2] >= R[0] and b[0] <= R[2] and b[3] >= 171 and b[1] <= R[3] - 4
    blk = [o for o in objs if o['stroke'] and near(o['sc'], (0, 0, 0)) and inR(o['b']) and (o['b'][2] - o['b'][0]) < 700 and not (o['b'][3] < 172)]
    # --- label words: group glyph-sized strokes on a baseline, OCR each group
    gl = [o for o in blk if 3.8 <= o['b'][3] - o['b'][1] <= 7.2 and (o['b'][2] - o['b'][0]) <= 5.4]
    gl.sort(key=lambda o: o['b'][0]); used = set(); groups = []
    for i, g in enumerate(gl):
        if i in used: continue
        grp = [g]; used.add(i); last = g
        while True:
            nx = None
            for j, o in enumerate(gl):
                if j in used: continue
                if abs((o['b'][1] + o['b'][3]) - (last['b'][1] + last['b'][3])) < 1.6 and abs((o['b'][3] - o['b'][1]) - (last['b'][3] - last['b'][1])) < 1.6 and o['b'][0] >= last['b'][0] - .3 and o['b'][0] - last['b'][2] < 2.4:
                    nx = j; break
            if nx is None: break
            used.add(nx); grp.append(gl[nx]); last = gl[nx]
        if len(grp) >= 2: groups.append(grp)
    pg = doc[pi]; words = []
    for grp in groups:
        x0 = min(o['b'][0] for o in grp) - 1.5; y0 = min(o['b'][1] for o in grp) - 1.5; x1 = max(o['b'][2] for o in grp) + 1.5; y1 = max(o['b'][3] for o in grp) + 1.5
        img = pg.render(scale=OCR_S * 2, crop=(x0, p.height - y1, p.width - x1, y0)).to_pil().convert('L')
        from PIL import ImageOps
        img = ImageOps.expand(img, border=20, fill=255)
        t = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip().replace(' ', '')
        words.append([t, round(x0 + 1.5, 1), round(y0 + 1.5, 1), round(x1 - 1.5, 1), round(y1 - 1.5, 1), len(grp)])
    # glyph strokes inside word boxes are not walls
    def in_word(b):
        for w in words:
            if b[0] >= w[1] - .8 and b[2] <= w[3] + .8 and b[1] >= w[2] - .8 and b[3] <= w[4] + .8 and (b[3] - b[1]) < 8: return True
        return False
    walls, diag = [], []
    for o in blk:
        b = o['b']
        if in_word(b): continue
        if len(o['pts']) == 2 and (b[2] - b[0]) > 6 and (b[3] - b[1]) > 6: diag.append(o)
        else: walls.append(o)
    voids = []
    for i in range(len(diag)):
        for j in range(i + 1, len(diag)):
            a, c = diag[i]['b'], diag[j]['b']
            if all(abs(a[k] - c[k]) < 1.5 for k in range(4)): voids.append([round(v, 1) for v in a])
    def isvoiddiag(o): return any(all(abs(o['b'][k] - v[k]) < 1.5 for k in range(4)) for v in voids)
    walls += [o for o in diag if not isvoiddiag(o)]
    # --- raster
    W, H = int((R[2] - R[0]) * K), int((R[3] - R[1]) * K)
    wall = np.zeros((H, W), np.uint8)
    for o in walls:
        pts = np.array([[(x - R[0]) * K, (y - R[1]) * K] for x, y in o['pts']], np.int32)
        cv2.polylines(wall, [pts], False, 255, 2)
    # envelope (closing)
    RD = int(14 * K); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RD + 1, 2 * RD + 1))
    dil = cv2.dilate(wall, k)
    free = (dil == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = np.isin(lab, [b for b in border if b != 0]).astype(np.uint8)
    out = cv2.dilate(out, k)
    inside = ((out == 0) & (wall == 0)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(inside, connectivity=4)
    return dict(R=R, words=words, voids=voids, walls=walls, objs=objs, wall=wall, lab=lab, stats=stats, n=n, building=(out == 0).astype(np.uint8))

if __name__ == '__main__':
    f = sys.argv[1]; pi = int(sys.argv[2])
    pdf = pdfplumber.open(f); doc = pdfium.PdfDocument(f)
    r = process(f, pi, doc, pdf)
    print(len(r['walls']), r['n'], r['words'][:40], r['voids'][:5])
