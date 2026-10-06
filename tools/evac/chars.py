import pdfplumber, glob, numpy as np, cv2, pickle
from extract import page_objs, near
def words_of(objs):
    blk = [o for o in objs if o['stroke'] and near(o['sc'], (0, 0, 0)) and o['b'][1] > 171]
    gl = [o for o in blk if 4.6 <= o['b'][3] - o['b'][1] <= 6.4 and (o['b'][2] - o['b'][0]) <= 5.6]
    small = [o for o in blk if (o['b'][3] - o['b'][1]) <= 6.4 and (o['b'][2] - o['b'][0]) <= 6]
    gl.sort(key=lambda o: o['b'][0]); used = set(); out = []; seen = set()
    for i, g in enumerate(gl):
        if i in used: continue
        grp = [g]; used.add(i); last = g
        while True:
            nx = None
            for j, o in enumerate(gl):
                if j in used: continue
                if abs((o['b'][1] + o['b'][3]) - (last['b'][1] + last['b'][3])) < 1.6 and o['b'][0] >= last['b'][0] - .3 and o['b'][0] - last['b'][2] < 2.4: nx = j; break
            if nx is None: break
            used.add(nx); grp.append(gl[nx]); last = gl[nx]
        if len(grp) < 2: continue
        y0 = min(o['b'][1] for o in grp) - .7; y1 = max(o['b'][3] for o in grp) + .7
        band = [o for o in small if o['b'][1] >= y0 and o['b'][3] <= y1]
        x0 = min(o['b'][0] for o in grp); x1 = max(o['b'][2] for o in grp); ch = True
        while ch:
            ch = False
            for o in band:
                if o['b'][2] >= x0 - 2.4 and o['b'][0] <= x1 + 2.4 and (o['b'][0] < x0 or o['b'][2] > x1): x0 = min(x0, o['b'][0]); x1 = max(x1, o['b'][2]); ch = True
        ws = [o for o in band if o['b'][0] >= x0 - .01 and o['b'][2] <= x1 + .01]
        key = (round(x0, 1), round(y0, 1))
        if key in seen: continue
        seen.add(key)
        # split into chars by x overlap
        ws.sort(key=lambda o: o['b'][0]); cells = []
        for o in ws:
            if cells and o['b'][0] < cells[-1][1] - .4: cells[-1][2].append(o); cells[-1][1] = max(cells[-1][1], o['b'][2])
            else: cells.append([o['b'][0], o['b'][2], [o]])
        top = min(o['b'][1] for o in ws); bot = max(o['b'][3] for o in ws)
        out.append(dict(b=(x0, top, x1, bot), cells=[(c[0], c[1], c[2]) for c in cells]))
    return out
def cellimg(c, top, h, S=4):
    x0 = c[0]; img = np.zeros((int(h * S) + 8, 32), np.uint8)
    for o in c[2]: cv2.polylines(img, [np.array([[(x - x0) * S + 4, (y - top) * S + 4] for x, y in o['pts']], np.int32)], False, 255, 2)
    return cv2.resize(img, (16, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
if __name__ == '__main__':
    allc = []
    for f in sorted(glob.glob('/home/claude/evac/*.pdf')):
        for pi, p in enumerate(pdfplumber.open(f).pages):
            for w in words_of(page_objs(p)):
                h = w['b'][3] - w['b'][1]
                for c in w['cells']: allc.append((cellimg(c, w['b'][1], h), c[1] - c[0], h))
    print(len(allc))
    reps = []; lab = []
    for img, wd, h in allc:
        best, bd = -1, 1e9
        for k, (r, rw) in enumerate(reps):
            d = np.abs(img - r).mean()
            if d < bd and abs(wd / h - rw) < .15: bd, best = d, k
        if best >= 0 and bd < .05: lab.append(best)
        else: reps.append((img, wd / h)); lab.append(len(reps) - 1)
    cnt = np.bincount(lab); order = np.argsort(-cnt)
    print(len(reps))
    n = len(order); cols = 14; sheet = np.full((44 * ((n + cols - 1) // cols), cols * 52), 255, np.uint8)
    for k, c in enumerate(order):
        r, cc = divmod(k, cols); im = 255 - (cv2.resize(reps[c][0], (20, 30)) * 255).astype(np.uint8)
        sheet[r * 44 + 2:r * 44 + 32, cc * 52:cc * 52 + 20] = im
        cv2.putText(sheet, str(c), (cc * 52 + 22, r * 44 + 14), cv2.FONT_HERSHEY_SIMPLEX, .38, 0, 1)
        cv2.putText(sheet, str(cnt[c]), (cc * 52 + 22, r * 44 + 28), cv2.FONT_HERSHEY_SIMPLEX, .3, 110, 1)
    cv2.imwrite('/home/claude/work/char_sheet.png', sheet)
    pickle.dump([(r, a) for r, a in reps], open('/home/claude/work/char_reps.pkl', 'wb'))
