import sys, json, math, glob, os, pickle
import numpy as np, cv2, pdfplumber
from extract import page_objs, near
from chars import words_of, cellimg
K = 2.0
REPS = pickle.load(open('/home/claude/work/char_reps.pkl', 'rb'))
CH = {}
for ch, ids in {'0': [79, 46, 59, 15, 73, 239, 209, 235], '1': [44, 8, 74], '2': [14, 119, 19, 45, 75, 230, 233, 226, 83], '3': [43, 98, 85, 21], '4': [99, 22, 9, 64, 18],
                '5': [86, 60, 100, 20, 77], '6': [58, 101, 10], '7': [57, 157, 236, 187, 38, 248, 56, 80], '8': [103, 51, 17], '9': [122, 49, 16, 113], 'C': [121, 106],
                'F': [7, 63], 'L': [36, 39, 160], 'D': [42], '-': [34, 65, 50, 29, 47]}.items():
    for i in ids: CH[i] = ch
def classify(img, asp):
    best, bd = -1, 1e9
    for k, (r, ra) in enumerate(REPS):
        if abs(asp - ra) > .2: continue
        d = np.abs(img - r).mean()
        if d < bd: bd, best = d, k
    return CH.get(best, '?') if bd < .07 else '?'
FLOORS = {0: '1 2 3 4 5 B1', 1: '1 2 3 4 5 6 B1 B2', 2: '1 2 3 4 5 R B1', 3: '1 2 3 4 5 B1', 4: '1 2 3 4 5 R B1', 5: '1 2 3 4 5 6 B1', 6: '1 2 3 4 5 6 7 B1 B2', 7: '1 2 3 4 5 R B1', 8: '1', 9: '1 2 3 4 5', 10: '1 2 3 4',
          11: '1 2 3 B1', 12: '1 2 3', 13: '1 2', 14: '1 2 3', 15: '1 2 3 4', 16: '1 2 R', 17: '1 2 3', 18: '1 2 3 4 5 6 7', 19: '1 2 3 4 5 6 7 8 B1', 20: '1 2 3 4 5 R', 21: '1 2 3 4 5 R B1', 22: '1 2 3 4', 23: '1 2 3 4', 24: '1 2 3 4', 25: '1 2 3 4'}
def process(f, bi, pi, p, dbg=None):
    objs = page_objs(p)
    legend = [o['b'][1] for o in objs if o['stroke'] and near(o['sc'], (1, 0, 0)) and abs(o['lw'] - 1.8) < .1 and o['b'][0] < 130 and o['b'][1] > 400]
    lt = min(legend) - 6 if legend else 670
    R = (16, 160, 826, lt + 4)
    def inR(b): return b[2] >= R[0] and b[0] <= R[2] and b[3] >= 171 and b[1] <= R[3] - 4
    blk = [o for o in objs if o['stroke'] and near(o['sc'], (0, 0, 0)) and inR(o['b']) and (o['b'][2] - o['b'][0]) < 700 and not (o['b'][3] < 172)]
    # words
    words = []; wordobj = set()
    for w in words_of(objs):
        if not inR(w['b']): continue
        h = w['b'][3] - w['b'][1]
        txt = ''.join(classify(cellimg(c, w['b'][1], h), (c[1] - c[0]) / h) for c in w['cells'])
        for c in w['cells']:
            for o in c[2]: wordobj.add(id(o))
        known = sum(ch != '?' for ch in txt)
        if known >= 2 and known >= .6 * len(txt): words.append([txt.replace('?', '')] + [round(v, 1) for v in w['b']])
    # map object identities: words_of used page_objs again -> match by bbox instead
    wb = [w[1:5] for w in words]
    def in_word(b): return any(b[0] >= x0 - .5 and b[2] <= x1 + .5 and b[1] >= y0 - .5 and b[3] <= y1 + .5 for x0, y0, x1, y1 in wb)
    walls, diag, treads = [], [], []
    for o in blk:
        b = o['b']; w, h = b[2] - b[0], b[3] - b[1]
        if in_word(b): continue
        if len(o['pts']) == 2 and w > 6 and h > 6: diag.append(o); continue
        walls.append(o)
        if len(o['pts']) == 2 and (w < .3 or h < .3) and 3 <= max(w, h) <= 40: treads.append(o)
    voids = []
    for i in range(len(diag)):
        for j in range(i + 1, len(diag)):
            a, c = diag[i]['b'], diag[j]['b']
            if all(abs(a[k] - c[k]) < 1.5 for k in range(4)): voids.append([round(v, 1) for v in a])
    walls += [o for o in diag if not any(all(abs(o['b'][k] - v[k]) < 1.5 for k in range(4)) for v in voids)]
    # stair flights: >=5 parallel equal treads, spacing 1-3.5
    flights = []
    for vert in (True, False):
        T = [o for o in treads if ((o['b'][2] - o['b'][0]) < .3) == vert]
        key = lambda o: (o['b'][0] if vert else o['b'][1])
        T.sort(key=key); used = [False] * len(T)
        for i, t in enumerate(T):
            if used[i]: continue
            L0, L1 = (t['b'][1], t['b'][3]) if vert else (t['b'][0], t['b'][2])
            grp = [i]; last = key(t)
            for j in range(i + 1, len(T)):
                if used[j]: continue
                o = T[j]; M0, M1 = (o['b'][1], o['b'][3]) if vert else (o['b'][0], o['b'][2])
                if abs(M0 - L0) < 1 and abs(M1 - L1) < 1 and .8 < key(o) - last < 3.6: grp.append(j); last = key(o)
            if len(grp) >= 5:
                for j in grp: used[j] = True
                xs = [T[j]['b'] for j in grp]
                flights.append([round(min(b[0] for b in xs), 1), round(min(b[1] for b in xs), 1), round(max(b[2] for b in xs), 1), round(max(b[3] for b in xs), 1), 'v' if vert else 'h', len(grp)])
    # stair cores: merge flights within 5 units
    cores = []
    for fl in flights:
        b = fl[:4]; merged = False
        for c in cores:
            if b[0] < c['b'][2] + 5 and b[2] > c['b'][0] - 5 and b[1] < c['b'][3] + 5 and b[3] > c['b'][1] - 5:
                c['b'] = [min(c['b'][0], b[0]), min(c['b'][1], b[1]), max(c['b'][2], b[2]), max(c['b'][3], b[3])]; c['f'].append(fl); merged = True; break
        if not merged: cores.append({'b': list(b), 'f': [fl]})
    # raster + envelope
    W, H = int((R[2] - R[0]) * K), int((R[3] - R[1]) * K)
    wall = np.zeros((H, W), np.uint8)
    for o in walls: cv2.polylines(wall, [np.array([[(x - R[0]) * K, (y - R[1]) * K] for x, y in o['pts']], np.int32)], False, 255, 2)
    RD = int(14 * K); ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RD + 1, 2 * RD + 1))
    free = (cv2.dilate(wall, ker) == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    outside = cv2.dilate(np.isin(lab, list(border)).astype(np.uint8), ker)
    bld = (outside == 0).astype(np.uint8)
    inside = ((bld == 1) & (wall == 0)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(inside, connectivity=4)
    # icons
    icons = {'stair': [], 'wcM': [], 'wcF': [], 'arrow': []}
    for o in objs:
        if not o['fill'] or not inR(o['b']): continue
        b = o['b']; w, h = b[2] - b[0], b[3] - b[1]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        fc = o['fc']
        if near(fc, (.2, .6, .2)) and 9 < w < 12 and 9 < h < 12 and len(o['pts']) >= 12: icons['stair'].append([round(cx, 1), round(cy, 1)])
        elif near(fc, (.2, .6, .2)) and (5 < w < 13 and 5 < h < 13) and len(o['pts']) in (8, 9, 10): icons['arrow'].append([round(cx, 1), round(cy, 1)])
        elif near(fc, (0, 0, 1)) and 10 < w < 14: icons['wcM'].append([round(cx, 1), round(cy, 1)])
        elif near(fc, (1, 0, 0)) and 10 < w < 14 and h > 10: icons['wcF'].append([round(cx, 1), round(cy, 1)])
    def at(x, y):
        X, Y = int((x - R[0]) * K), int((y - R[1]) * K)
        for r in range(0, 7):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if 0 <= Y + dy < H and 0 <= X + dx < W and lab[Y + dy, X + dx] > 0: return int(lab[Y + dy, X + dx])
        return 0
    tags = {}
    for k, pts in icons.items():
        for x, y in pts:
            l = at(x, y)
            if l: tags.setdefault(l, {}); tags[l][k] = tags[l].get(k, 0) + 1
    wl = {}
    for i, w in enumerate(words):
        l = at((w[1] + w[3]) / 2, (w[2] + w[4]) / 2)
        if l: wl.setdefault(l, []).append(w[0])
    # spaces -> rects at 1 unit cells (0.15 m)
    C2 = 2
    spaces = []
    for l in range(1, n):
        x, y, w, h, a = stats[l]
        aU = a / K / K
        if aU < 60 or min(w, h) < 5 * K: continue
        sub = (lab[y:y + h, x:x + w] == l).astype(np.uint8)
        gh, gw = (h + C2 - 1) // C2, (w + C2 - 1) // C2
        pad = np.zeros((gh * C2, gw * C2), np.uint8); pad[:h, :w] = sub
        m = (pad.reshape(gh, C2, gw, C2).mean(axis=(1, 3)) >= .5).astype(np.uint8)
        rects = []
        for yy in range(gh):
            for xx in range(gw):
                if not m[yy, xx]: continue
                x2 = xx
                while x2 + 1 < gw and m[yy, x2 + 1]: x2 += 1
                y2 = yy
                while y2 + 1 < gh and m[y2 + 1, xx:x2 + 1].all(): y2 += 1
                m[yy:y2 + 1, xx:x2 + 1] = 0
                rects.append([round((x + xx * C2) / K + R[0], 1), round((y + yy * C2) / K + R[1], 1), round((x + (x2 + 1) * C2) / K + R[0], 1), round((y + (y2 + 1) * C2) / K + R[1], 1)])
        spaces.append({'id': l, 'a': round(aU), 'bb': [round(x / K + R[0], 1), round(y / K + R[1], 1), round((x + w) / K + R[0], 1), round((y + h) / K + R[1], 1)], 'r': rects, 't': tags.get(l, {}), 'w': wl.get(l, [])})
    os.makedirs('/home/claude/work/walls', exist_ok=True)
    cv2.imwrite(f'/home/claude/work/walls/{bi:02d}_{pi}.png', cv2.resize(wall, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    # building footprint mask at 2 units for alignment
    E = 4; eh, ew = H // E, W // E
    env = (bld[:eh * E, :ew * E].reshape(eh, E, ew, E).mean(axis=(1, 3)) > .5).astype(np.uint8)
    res = dict(bi=bi, page=pi, R=R, K=K, words=words, voids=voids, cores=[{'b': [round(v, 1) for v in c['b']], 'f': c['f']} for c in cores], spaces=spaces, icons=icons,
               env=[''.join('1' if v else '0' for v in row) for row in env], envC=E / K)
    if dbg is not None:
        im = np.full((H, W, 3), 40, np.uint8)
        rng = np.random.default_rng(1)
        for s in spaces:
            c = (220, 215, 200) if s['t'].get('arrow') else tuple(int(v) for v in rng.integers(80, 230, 3))
            im[lab == s['id']] = c
        im[wall > 0] = (0, 0, 0)
        for c in cores:
            b = c['b']; cv2.rectangle(im, (int((b[0] - R[0]) * K), int((b[1] - R[1]) * K)), (int((b[2] - R[0]) * K), int((b[3] - R[1]) * K)), (0, 0, 255), 2)
        for v in voids: cv2.rectangle(im, (int((v[0] - R[0]) * K), int((v[1] - R[1]) * K)), (int((v[2] - R[0]) * K), int((v[3] - R[1]) * K)), (255, 255, 0), 1)
        for w in words: cv2.putText(im, w[0], (int((w[1] - R[0]) * K), int((w[4] - R[1]) * K)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 120), 1)
        for x, y in icons['stair']: cv2.circle(im, (int((x - R[0]) * K), int((y - R[1]) * K)), 6, (0, 160, 0), -1)
        cv2.imwrite(dbg, im)
    return res
if __name__ == '__main__':
    out = []
    files = sorted(glob.glob('/home/claude/evac/*.pdf'))
    sel = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else range(len(files))
    os.makedirs('/home/claude/work/dbg', exist_ok=True)
    for bi in sel:
        f = files[bi]; pdf = pdfplumber.open(f); fl = FLOORS[bi].split()
        for pi, p in enumerate(pdf.pages):
            r = process(f, bi, pi, p, dbg=f'/home/claude/work/dbg/{bi:02d}_{fl[pi]}.png'); r['floor'] = fl[pi]; r['name'] = os.path.basename(f)[3:-4]
            out.append(r); print(bi, fl[pi], len(r['spaces']), 'cores', len(r['cores']), 'words', [w[0] for w in r['words']][:14], flush=True)
    json.dump(out, open('/home/claude/work/plans_%s.json' % ('all' if len(sys.argv) == 1 else '_'.join(sys.argv[1:])), 'w'))
