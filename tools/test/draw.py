import json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
d = json.load(open('plan.json')); G = d
fig, ax = plt.subplots(figsize=(14, 9))
P = np.array(d['poly'] + [d['poly'][0]]); ax.plot(P[:, 0], P[:, 1], 'k-', lw=2)
U = d['F']['U']; V = d['F']['V']; ox, oz = d['F']['ox'], d['F']['oz']
cols = plt.cm.tab20(np.arange(20))
for k, (l, r) in enumerate(zip(d['lab'], d['room'])):
    i, j = k % d['nu'], k // d['nu']
    if l < 0: continue
    u = d['u0'] + (i + .5) * .5; v = d['v0'] + (j + .5) * .5
    x = ox + U[0] * u + V[0] * v; z = oz + U[1] * u + V[1] * v
    c = 'red' if l == 1 else (cols[r % 20] if r >= 0 else '#eeeeee')
    ax.plot(x, z, 's', ms=2.2, color=c, alpha=.5 if r >= 0 else .8)
for w in d['walls']: ax.plot([w[0], w[2]], [w[1], w[3]], 'b-', lw=.8)
if d['inst']: I = np.array(d['inst']); ax.plot(I[:, 0], I[:, 1], 'g.', ms=1)
for r in d['rooms']:
    for p in r['doors']: ax.plot(p[0], p[1], 'm^', ms=4)
ax.set_aspect('equal'); ax.invert_yaxis(); plt.tight_layout(); plt.savefig(sys.argv[1], dpi=90)
