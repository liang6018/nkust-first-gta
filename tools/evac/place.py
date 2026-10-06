"""Robust placement: per-floor solid masks (labeled rooms, corridors, stairs, small rooms), floors registered by blurred-wall NCC with a
near-identity prior, union -> OSM footprint fit (rotation alpha+k90, scale, shift by template matching)."""
import json, math, numpy as np, cv2
from fit import S, plans, MAP
U = 1.0  # mask resolution: 1 px per pdf unit
def wimg(q):
    w = cv2.imread(f"/home/claude/work/walls/{q['bi']:02d}_{q['page']}.png", 0).astype(np.float32) / 255
    return cv2.GaussianBlur(w, (0, 0), 2.0)
def solid(q, shape):
    m = np.zeros(shape, np.uint8); R = q['R']
    for sp in q['spaces']:
        keep = sp['w'] or sp['t'].get('arrow') or sp['t'].get('stair') or sp['t'].get('wcM') or sp['t'].get('wcF') or sp['a'] < 700
        if not keep: continue
        for x0, y0, x1, y1 in sp['r']: cv2.rectangle(m, (int(x0 - R[0]), int(y0 - R[1])), (int(x1 - R[0]), int(y1 - R[1])), 1, -1)
    for c in q['cores']:
        b = c['b']; cv2.rectangle(m, (int(b[0] - R[0]) - 3, int(b[1] - R[1]) - 3), (int(b[2] - R[0]) + 3, int(b[3] - R[1]) + 3), 1, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
def reg(WA, WB, Rr, Rq):
    best = None
    for s in np.exp(np.linspace(math.log(.9), math.log(1.1), 21)):
        B = cv2.resize(WB, None, fx=s, fy=s)
        # place B so that page origins coincide up to a +-60 unit window
        H, W = WA.shape; pad = 70
        canvas = np.zeros((H + 2 * pad, W + 2 * pad), np.float32); canvas[pad:pad + H, pad:pad + W] = WA
        bx, by = int(round(Rq[0] * s - Rr[0])), int(round(Rq[1] * s - Rr[1]))  # B pixel (0,0) lands here in A coords when shift=0
        h2, w2 = min(B.shape[0], H), min(B.shape[1], W)
        tpl = B[:h2, :w2]
        x0, y0 = pad + bx - 60, pad + by - 60
        win = canvas[max(0, y0):y0 + h2 + 120, max(0, x0):x0 + w2 + 120]
        if win.shape[0] < h2 or win.shape[1] < w2: continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCORR_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        dx, dy = loc[0] + max(0, x0) - pad - bx, loc[1] + max(0, y0) - pad - by
        v = mx - .0008 * math.hypot(dx, dy)
        if best is None or v > best[0]: best = (v, s, dx, dy)
    v, s, dx, dy = best
    # u_ref = u * s + (Rr... ) : u_ref - Rr0 = (u - Rq0) * s + bx + dx  =>  u_ref = u*s + (Rr0 + bx + dx - Rq0*s)
    bx, by = Rq[0] * s - Rr[0], Rq[1] * s - Rr[1]
    return dict(score=round(float(v), 3), s=float(s), ox=float(Rr[0] + bx + dx - Rq[0] * s), oy=float(Rr[1] + by + dy - Rq[1] * s))
def footprint_raster(poly, res=.5, pad=60):
    P = np.array(poly); x0, z0 = P.min(0) - pad; x1, z1 = P.max(0) + pad
    m = np.zeros((int((z1 - z0) / res), int((x1 - x0) / res)), np.float32)
    cv2.fillPoly(m, [((P - [x0, z0]) / res).astype(np.int32)], 1.0); return m, (x0, z0, res)
def main():
    out = {}
    for b in S['bld']:
        key = next((k for k in MAP if k in b['n']), None)
        if key is None or MAP[key] is None: continue
        bi = (MAP[key] if isinstance(MAP[key], list) else [MAP[key]])[0]
        ps = [q for q in plans if q['bi'] == bi]
        above = [q for q in ps if not q['floor'].startswith('B') and q['floor'] != 'R']
        ref = next((q for q in above if q['floor'] == '2'), above[0]); Rr = ref['R']; WA = wimg(ref)
        shape = (int(Rr[3] - Rr[1]) + 1, int(Rr[2] - Rr[0]) + 1)
        floors = {}; Un = np.zeros(shape, np.float32)
        for q in ps:
            fr = dict(score=1, s=1, ox=0, oy=0) if q is ref else reg(WA, wimg(q), Rr, q['R'])
            floors[q['floor']] = fr
            if q in above:
                m = solid(q, (int(q['R'][3] - q['R'][1]) + 1, int(q['R'][2] - q['R'][0]) + 1)).astype(np.float32)
                s = fr['s']; M = np.float32([[s, 0, q['R'][0] * s + fr['ox'] - Rr[0]], [0, s, q['R'][1] * s + fr['oy'] - Rr[1]]])
                Un = np.maximum(Un, cv2.warpAffine(m, M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST))
        # full envelope union as an alternative mask
        Ue = np.zeros(shape, np.float32)
        for q in above:
            fr = floors[q['floor']]; e = np.array([[c == '1' for c in row] for row in q['env']], np.float32); C = q['envC']
            e = cv2.resize(e, None, fx=C, fy=C, interpolation=cv2.INTER_NEAREST); s = fr['s']
            M = np.float32([[s, 0, q['R'][0] * s + fr['ox'] - Rr[0]], [0, s, q['R'][1] * s + fr['oy'] - Rr[1]]])
            Ue = np.maximum(Ue, cv2.warpAffine(e, M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST))
        results = []
        for kind, Un in (('solid', Un), ('env', Ue)):
            # fit union to OSM footprint
            G, org = footprint_raster(b['f']); P = np.array(b['f'])
            E = np.roll(P, -1, 0) - P; ang = np.degrees(np.arctan2(E[:, 1], E[:, 0])) % 90; L = np.hypot(E[:, 0], E[:, 1]); hist = np.zeros(90)
            for a, l in zip(ang, L): hist[int(a) % 90] += l
            alpha = float(np.argmax(np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5), 'valid')[1:91]))
            ys, xs = np.nonzero(Un > .5); pts = np.stack([xs + Rr[0] + .5, ys + Rr[1] + .5], 1)[::4]; aU = (Un > .5).sum(); gA = G.sum() * org[2] ** 2
            best = None
            for k in range(4):
                for d in np.arange(-3, 3.01, .5):
                    th = math.radians(alpha + k * 90 + d); c, s_ = math.cos(th), math.sin(th)
                    sc0 = math.sqrt(gA / aU)
                    for sc in sc0 * np.exp(np.linspace(math.log(.75), math.log(1.3), 36)):
                        X = sc * (c * pts[:, 0] - s_ * pts[:, 1]); Z = sc * (s_ * pts[:, 0] + c * pts[:, 1])
                        # rasterize rotated plan into a small image, then template-match over the footprint raster
                        res = org[2]; mx, mz = X.min(), Z.min()
                        T = np.zeros((int((Z.max() - mz) / res) + 2, int((X.max() - mx) / res) + 2), np.float32)
                        T[((Z - mz) / res).astype(int), ((X - mx) / res).astype(int)] = 1
                        T = cv2.dilate(T, np.ones((3, 3), np.uint8))
                        if T.shape[0] >= G.shape[0] or T.shape[1] >= G.shape[1]: continue
                        r = cv2.matchTemplate(G, T, cv2.TM_CCORR); _, v, _, loc = cv2.minMaxLoc(r)
                        iou = v / (T.sum() + G.sum() - v)
                        if best is None or iou > best[0]: best = (iou, th, sc, loc[0] * res + org[0] - mx, loc[1] * res + org[1] - mz, k)
            results.append((best[0], kind, best, Un))
        iou0, kind, best, Un = max(results, key=lambda x: x[0])
        np.save(f'solid_{bi}.npy', Un)
        iou, th, sc, tx, tz, k = best
        alts = []
        for k2 in range(4):
            pass
        out[b['n']] = dict(kind=kind, bi=bi, ref=ref['floor'], floors=floors, iou=round(float(iou), 3), th=th, sc=sc, t=[float(tx), float(tz)], k=k)
        print(b['n'], kind, 'iou', round(float(iou), 3), 'k', k, 'deg', round(math.degrees(th), 1), 'sc', round(sc, 4), {f: v['score'] for f, v in floors.items()}, flush=True)
    json.dump(out, open('place.json', 'w'), ensure_ascii=False, indent=0)
if __name__ == '__main__': main()
