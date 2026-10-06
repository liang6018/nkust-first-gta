import json, numpy as np, cv2, math, re
s = open('/home/claude/nkust-first-gta/index.html').read(); i = s.index('const SURVEY = ') + 15
S, _ = json.JSONDecoder().raw_decode(s[i:])
plans = json.load(open('plans_all.json'))
MAP = {'游泳館': 11, '產業實驗園區': [9, 10], '樂群樓': 19, '敬業樓 A': 20, '敬業樓 B': 21, '財金': 2, '圖書資訊': 6, '學生活動中心': 7, '外語學院 圓形': None, '外語學院': 1, '管理': 4, '行政': 5,
       '電機資訊': 3, '工學院': 0, '教師研究': 14, '創業園區': 12, '跨領域': 17, '智慧製造': 16, '實作工場': [15, 18], '職務宿舍 1': 22, '職務宿舍 2': 23, '職務宿舍 3': 24, '職務宿舍 4': 25}
def union_env(ps):
    es = [np.array([[c == '1' for c in row] for row in p['env']], np.uint8) for p in ps]
    H = max(e.shape[0] for e in es); W = max(e.shape[1] for e in es); env = np.zeros((H, W), np.uint8)
    for e in es: env[:e.shape[0], :e.shape[1]] |= e
    return env
def plan_pts(bi, floor='1'):
    ps = [q for q in plans if q['bi'] == bi and not q['floor'].startswith('B') and q['floor'] != 'R']
    env = union_env(ps); C = ps[0]['envC']; R = ps[0]['R']
    ys, xs = np.nonzero(env)
    return np.stack([R[0] + (xs + .5) * C, R[1] + (ys + .5) * C], 1), C
def game_mask(poly, res=.5, pad=40):
    P = np.array(poly); x0, z0 = P.min(0) - pad; x1, z1 = P.max(0) + pad
    W, H = int((x1 - x0) / res), int((z1 - z0) / res); m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [((P - [x0, z0]) / res).astype(np.int32)], 1); return m, (x0, z0, res)
def score(pts, C, th, sc, t, m, org):
    c, s_ = math.cos(th), math.sin(th)
    X = sc * (c * pts[:, 0] - s_ * pts[:, 1]) + t[0]; Z = sc * (s_ * pts[:, 0] + c * pts[:, 1]) + t[1]
    x0, z0, res = org; ix = ((X - x0) / res).astype(int); iz = ((Z - z0) / res).astype(int)
    ok = (ix >= 0) & (iz >= 0) & (ix < m.shape[1]) & (iz < m.shape[0])
    inter = m[iz[ok], ix[ok]].sum() * (C * sc) ** 2 * (len(pts) / len(pts))
    a_plan = len(pts) * (C * sc) ** 2; a_game = m.sum() * res * res
    return inter / (a_plan + a_game - inter)
def fit(bi, poly):
    pts, C = plan_pts(bi); m, org = game_mask(poly)
    sub = pts[::max(1, len(pts) // 2500)]; Cs = C * math.sqrt(len(pts) / len(sub))
    gc = np.array(poly).mean(0); gA = m.sum() * .25
    best = (-1, 0, .1, gc)
    P = np.array(poly); E = np.roll(P, -1, 0) - P; ang = np.degrees(np.arctan2(E[:, 1], E[:, 0])) % 90; L = np.hypot(E[:, 0], E[:, 1])
    hist = np.zeros(90)
    for a, l in zip(ang, L): hist[int(a) % 90] += l
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5), 'valid')[1:91]
    alpha = float(np.argmax(hist))
    for th in np.radians([alpha + k * 90 + d for k in range(4) for d in (-2, -1, 0, 1, 2)]):
        for sc in np.exp(np.linspace(math.log(.04), math.log(.32), 60)):
            if abs(len(pts) * (C * sc) ** 2 / gA - 1) > .8: continue
            c, s_ = math.cos(th), math.sin(th); pc = pts.mean(0)
            t = gc - sc * np.array([c * pc[0] - s_ * pc[1], s_ * pc[0] + c * pc[1]])
            v = score(sub, Cs, th, sc, t, m, org)
            if v > best[0]: best = (v, th, sc, t)
    v, th, sc, t = best
    for it in range(4):  # local refine
        for dth in np.radians([-1, -.5, -.25, 0, .25, .5, 1]):
            for dsc in [.97, .985, 1, 1.015, 1.03]:
                for dx in [-1.5, -.5, 0, .5, 1.5]:
                    for dz in [-1.5, -.5, 0, .5, 1.5]:
                        tt = t + [dx, dz]; vv = score(sub, Cs, th + dth, sc * dsc, tt, m, org)
                        if vv > v: v, th2, sc2, t2 = vv, th + dth, sc * dsc, tt; best = (v, th2, sc2, t2)
        v, th, sc, t = best
    return dict(iou=round(float(v), 3), th=float(th), sc=float(sc), t=[float(t[0]), float(t[1])])
if __name__ == '__main__':
    res = {}
    for b in S['bld']:
        key = next((k for k in MAP if k in b['n']), None)
        if key is None or MAP[key] is None: print('skip', b['n']); continue
        cands = MAP[key] if isinstance(MAP[key], list) else [MAP[key]]
        best = None
        for bi in cands:
            r = fit(bi, b['f']); r['bi'] = bi
            if best is None or r['iou'] > best['iou']: best = r
        res[b['n']] = best; print(b['n'], best['bi'], best['iou'], round(math.degrees(best['th']), 1), round(best['sc'], 4), flush=True)
    json.dump(res, open('fit.json', 'w'), ensure_ascii=False, indent=0)
