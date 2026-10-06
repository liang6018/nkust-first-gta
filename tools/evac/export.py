"""Build per-building, per-floor 0.5 m room grids in world space from the registered evacuation plans."""
import json, math, gzip, base64, numpy as np, cv2
from fit import S, plans
place = json.load(open('place.json'))
CELL = .5
KIND = {'游泳館': 'gym', '產業實驗': 'office', '樂群樓': 'dorm', '敬業樓': 'dorm', '財金': 'class', '圖書資訊': 'library', '學生活動中心': 'club', '外語學院': 'class', '管理': 'class', '行政': 'office',
        '電機資訊': 'computer', '工學院': 'lab', '教師研究': 'faculty', '創業園區': 'office', '跨領域': 'workshop', '智慧製造': 'workshop', '職務宿舍': 'home'}
SKIP = ['實作工場']
def kind_of(n): return next((v for k, v in KIND.items() if k in n), 'class')
def floor_raster(q):
    """space index raster (1 px per pdf unit) for page q; walls filled by nearest space within 2.5 units; -1 = none"""
    R = q['R']; H, W = int(R[3] - R[1]) + 1, int(R[2] - R[0]) + 1
    lab = np.full((H, W), -1, np.int32)
    for i, sp in enumerate(q['spaces']):
        for x0, y0, x1, y1 in sp['r']: lab[int(y0 - R[1]):int(math.ceil(y1 - R[1])), int(x0 - R[0]):int(math.ceil(x1 - R[0]))] = i
    empty = (lab < 0).astype(np.uint8)
    d, idx = cv2.distanceTransformWithLabels(empty, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(empty == 0); src = np.zeros(idx.max() + 1, np.int32) - 1
    src[idx[ys, xs]] = lab[ys, xs]
    fill = src[idx]; out = np.where((empty == 1) & (d <= 2.5), fill, lab)
    return out
def classify(sp, kind, aM2, in_core):
    t = sp['t']
    if t.get('arrow') or in_core: return 'corr'
    if t.get('wcM') or t.get('wcF'): return 'wc'
    if kind in ('library', 'workshop', 'gym'): return 'open' if aM2 > 60 else ('store' if aM2 < 6 else 'office')
    if aM2 < 5: return 'store'
    if aM2 > 260 and not sp['w']: return 'open'
    if sp['w']:
        if kind in ('dorm', 'home'): return kind
        if aM2 < 22: return 'faculty' if kind != 'office' else 'office'
        return kind
    if kind in ('dorm', 'home'): return kind if aM2 < 40 else 'open'
    return 'office' if aM2 < 60 else 'open'
def rle(arr):
    out = []; prev = arr[0]; n = 0
    for v in arr:
        if v == prev: n += 1
        else: out.append(f'{prev}*{n}' if n > 1 else str(prev)); prev = v; n = 1
    out.append(f'{prev}*{n}' if n > 1 else str(prev)); return ','.join(out)
res = {}
for b in S['bld']:
    if b['n'] not in place or any(k in b['n'] for k in SKIP): continue
    pl = place[b['n']]
    if pl['iou'] < .45: continue
    th, sc, t = pl['th'], pl['sc'], pl['t']; c, s_ = math.cos(th), math.sin(th)
    U = np.array([c, s_]); V = np.array([-s_, c])
    kind = kind_of(b['n'])
    ps = sorted([q for q in plans if q['bi'] == pl['bi'] and not q['floor'].startswith('B') and q['floor'] != 'R'], key=lambda q: int(q['floor']))
    ref = [q for q in plans if q['bi'] == pl['bi'] and q['floor'] == pl['ref']][0]
    # grid frame over the OSM footprint
    P = np.array(b['f']); o = P.mean(0)
    uu = (P - o) @ U; vv = (P - o) @ V
    u0, v0 = math.floor(uu.min() / CELL) * CELL - CELL, math.floor(vv.min() / CELL) * CELL - CELL
    nu, nv = int(math.ceil((uu.max() - u0) / CELL)) + 2, int(math.ceil((vv.max() - v0) / CELL)) + 2
    ii, jj = np.meshgrid(np.arange(nu), np.arange(nv))
    cu = u0 + (ii + .5) * CELL; cv_ = v0 + (jj + .5) * CELL
    WX = o[0] + U[0] * cu + V[0] * cv_; WZ = o[1] + U[1] * cu + V[1] * cv_
    foot = np.zeros((nv, nu), np.uint8)
    poly_g = np.stack([((P - o) @ U - u0) / CELL, ((P - o) @ V - v0) / CELL], 1)
    cv2.fillPoly(foot, [np.round(poly_g * 8).astype(np.int32)], 1, shift=3)
    # world -> ref plan units
    dx, dz = WX - t[0], WZ - t[1]
    pref_x = (c * dx + s_ * dz) / sc; pref_y = (-s_ * dx + c * dz) / sc
    # stair cores from the reference floor
    cores = []
    for co in ref['cores']:
        x0, y0, x1, y1 = co['b']
        vert = sum(f[5] for f in co['f'] if f[4] == 'v') >= sum(f[5] for f in co['f'] if f[4] == 'h')
        # world corners -> grid (u, v)
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        W_ = np.stack([sc * (c * pts[:, 0] - s_ * pts[:, 1]) + t[0], sc * (s_ * pts[:, 0] + c * pts[:, 1]) + t[1]], 1)
        gu = (W_ - o) @ U; gv = (W_ - o) @ V
        cores.append([round(float(gu.min()), 2), round(float(gv.min()), 2), round(float(gu.max()), 2), round(float(gv.max()), 2), 'u' if vert else 'v'])
    # merge cores closer than 2.5 m
    merged = True
    while merged:
        merged = False
        for i in range(len(cores)):
            for j in range(i + 1, len(cores)):
                a, b2 = cores[i], cores[j]
                if a[0] - 2.5 < b2[2] and b2[0] - 2.5 < a[2] and a[1] - 2.5 < b2[3] and b2[1] - 2.5 < a[3]:
                    cores[i] = [min(a[0], b2[0]), min(a[1], b2[1]), max(a[2], b2[2]), max(a[3], b2[3]), a[4]]; del cores[j]; merged = True; break
            if merged: break
    cores = [c4 for c4 in cores if (c4[2] - c4[0]) * (c4[3] - c4[1]) > 2]
    if not cores: print('  no stair cores, skip', b['n']); continue
    floors = []
    for q in ps:
        fr = pl['floors'][q['floor']]
        px = (pref_x - fr['ox']) / fr['s']; py = (pref_y - fr['oy']) / fr['s']
        L = floor_raster(q); R = q['R']
        ix = np.round(px - R[0]).astype(int); iy = np.round(py - R[1]).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < L.shape[1]) & (iy < L.shape[0])
        sid = np.full((nv, nu), -1, np.int32); sid[ok] = L[iy[ok], ix[ok]]
        rooms = []; code = np.zeros((nv, nu), np.int32)
        ids, counts = np.unique(sid[(sid >= 0) & (foot > 0)], return_counts=True)
        for i, n in zip(ids, counts):
            sp = q['spaces'][i]; aM2 = n * CELL * CELL
            m = (sid == i) & (foot > 0)
            cc = np.stack([cu[m], cv_[m]], 1)
            core_hit = any(((cc[:, 0] >= a - .5) & (cc[:, 0] <= b2 + .5) & (cc[:, 1] >= v_ - .5) & (cc[:, 1] <= d + .5)).mean() > .35 for a, v_, b2, d, _ in cores)
            ty = classify(sp, kind, aM2, core_hit)
            if ty in ('corr', 'open'): code[m] = 1 if ty == 'corr' else 2; continue
            if n < 8: code[m] = 1; continue
            rooms.append([ty, (sp['w'][0] if sp['w'] else '')]); code[m] = len(rooms) + 2
        code[(foot > 0) & (sid < 0)] = 2   # inside the shell but not on this floor's plan: open floor
        code[foot == 0] = 0
        floors.append({'f': q['floor'], 'g': rle(code.ravel().tolist()), 'rooms': rooms})
    res[b['n']] = {'o': [round(float(o[0]), 3), round(float(o[1]), 3)], 'U': [round(c, 6), round(s_, 6)], 'u0': u0, 'v0': v0, 'nu': nu, 'nv': nv, 'kind': kind, 'cores': cores, 'floors': floors, 'iou': pl['iou']}
    print(b['n'], kind, len(ps), 'floors', 'cores', len(cores), 'rooms/floor', [len(f['rooms']) for f in floors], flush=True)
raw = json.dumps(res, ensure_ascii=False, separators=(',', ':')).encode()
gz = gzip.compress(raw, 9); b64 = base64.b64encode(gz).decode()
open('plandata.b64', 'w').write(b64); json.dump(res, open('plandata.json', 'w'), ensure_ascii=False)
print(len(raw), len(gz), len(b64))
