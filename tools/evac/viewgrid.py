import json, sys, numpy as np, cv2
D = json.load(open('plandata.json')); n = [k for k in D if sys.argv[1] in k][0]; d = D[n]
tiles = []
for f in d['floors']:
    vals = []
    for tok in f['g'].split(','):
        v, _, k = tok.partition('*'); vals += [int(v)] * (int(k) if k else 1)
    g = np.array(vals).reshape(d['nv'], d['nu'])
    rng = np.random.default_rng(3); pal = rng.integers(60, 230, (400, 3)); pal[0] = (30, 30, 30); pal[1] = (215, 215, 200); pal[2] = (120, 120, 120)
    im = pal[g].astype(np.uint8)
    for i, (ty, lab) in enumerate(f['rooms']):
        ys, xs = np.nonzero(g == i + 3)
        if len(xs): cv2.putText(im, ty[:3], (int(xs.mean()) - 6, int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 0, 0), 1)
    for u0, v0, u1, v1, o in d['cores']:
        cv2.rectangle(im, (int((u0 - d['u0']) / .5), int((v0 - d['v0']) / .5)), (int((u1 - d['u0']) / .5), int((v1 - d['v0']) / .5)), (0, 0, 255), 1)
    im = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST); cv2.putText(im, f['f'] + 'F', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    tiles.append(im)
W = max(t.shape[1] for t in tiles); cv2.imwrite('grid.png', np.vstack([np.pad(t, ((0, 6), (0, W - t.shape[1]), (0, 0))) for t in tiles[:3]]))
