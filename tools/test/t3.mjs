import { open } from './h.mjs';
const { b, pg, logs } = await open();
const only = process.argv[2];
const r = await pg.evaluate((only) => {
  const out = [];
  const goTo = (x, z, max = 900) => { let last = 1e9, stuck = 0; for (let f = 0; f < max; f += 6) { const p = GTA.player.pos, dx = x - p.x, dz = z - p.z, d = Math.hypot(dx, dz); if (d < .3) return true; if (d > last - .02) { if (++stuck > 12) return false; } else stuck = 0; last = Math.min(last, d); GTA.sim({ KeyW: true }, 6, Math.atan2(-dx, -dz)); } return false; };
  for (const I of GTA.interiors) {
    if (only && !I.name.includes(only)) continue;
    const res = { n: I.name, real: I.plan.real };
    const G = I.plan.grids ? I.plan.grids[0] : I.plan.gr;
    const [dx0, dz0] = I.doorsXZ[0], B = GTA.built.find(b => b.interior === I), E = B.entrance;
    // outside the main door -> inside
    GTA.tp(dx0 + E.nx * 2.5, dz0 + E.nz * 2.5);
    let ok = goTo(dx0 - E.nx * 1.2, dz0 - E.nz * 1.2);
    res.enter = ok;
    // BFS on corridor cells to the first core's front
    const c = I.cores[0]; const q = (a, bb) => [c.ox + c.A[0] * a + c.B[0] * bb, c.oz + c.A[1] * a + c.B[1] * bb];
    const N = G.nu * G.nv, free1 = k => k >= 0 && k < N && G.lab[k] === 0 && G.room[k] < 0, strict = k => free1(k) && G.dist[k] >= .4 && free1(k - 1) && free1(k + 1) && free1(k - G.nu) && free1(k + G.nu); let walk = strict;
    const cellOf = (x, z) => { const dx = x - G.F.ox, dz = z - G.F.oz; const u = dx * G.F.U[0] + dz * G.F.U[1], v = dx * G.F.V[0] + dz * G.F.V[1]; const i = Math.floor((u - G.u0) / .5), j = Math.floor((v - G.v0) / .5); return i < 0 || j < 0 || i >= G.nu || j >= G.nv ? -1 : i + j * G.nu; };
    const toW = k => { const u = G.u0 + (k % G.nu + .5) * .5, v = G.v0 + (Math.floor(k / G.nu) + .5) * .5; return [G.F.ox + G.F.U[0] * u + G.F.V[0] * v, G.F.oz + G.F.U[1] * u + G.F.V[1] * v]; };
    const s = cellOf(GTA.player.pos.x, GTA.player.pos.z), [fx, fz] = q(-1, 0), t = cellOf(fx, fz);
    let prev, Q, found = false;
    for (const wf of [strict, k => free1(k) && G.dist[k] >= .4]) { walk = wf;     prev = new Int32Array(N).fill(-2); Q = [s]; prev[s] = -1; found = false;
    for (let h = 0; h < Q.length && s >= 0; h++) { const k = Q[h]; if (k === t) { found = true; break; } const i = k % G.nu; for (const m of [i > 0 ? k - 1 : -1, i < G.nu - 1 ? k + 1 : -1, k - G.nu, k + G.nu]) if (m >= 0 && m < N && prev[m] === -2 && walk(m)) { prev[m] = k; Q.push(m); } }
 if (found) break; res.lenient = walk !== strict; }
    res.path = found; if (!found) res.dbg = [walk(s), walk(t), Q.length, G.lab[t], G.room[t], +G.dist[t].toFixed(2)].join(",");
    if (found) { const p = []; for (let k = t; k >= 0; k = prev[k]) p.push(k); p.reverse(); let at = null; for (let i = 2; i < p.length && ok; i += 3) { ok = goTo(...toW(p[i]), 300); if (!ok) at = toW(p[i]); } ok = ok && goTo(fx, fz, 300); res.walk = ok; if (!ok) { const P = GTA.player.pos; res.dbg = JSON.stringify({ p: [+P.x.toFixed(1), +P.z.toFixed(1), +P.y.toFixed(1)], to: at && at.map(v => +v.toFixed(1)), near: GTA.near(P.x, P.z, P.y + .5) }).slice(0, 300); } }
    else { GTA.tp(fx, fz); }
    // climb
    const W4 = c.W / 4 + .05; let fl = 0;
    GTA.tp(...q(-1, -W4)); GTA.player.pos.y = I.levels[0] + (I.levels[0] ? 0 : 0);
    for (let k = 0; k < I.levels.length - 1; k++) {
      if (!goTo(...q(.6, -W4), 300) || !goTo(...q(c.CL - .6, -W4), 600) || !goTo(...q(c.CL - .6, W4), 200) || !goTo(...q(.6, W4), 600)) break;
      if (Math.abs(GTA.player.pos.y - I.levels[k + 1]) < .3) fl = k + 1; else break;
    }
    res.climb = `${fl}/${I.levels.length - 1}`; res.y = +GTA.player.pos.y.toFixed(2);
    // rooms with doors
    let nd = 0, tot = 0; for (const g of (I.plan.grids || [I.plan.gr, I.plan.up])) for (const r of g.rooms) { tot++; if (!r.doors || !r.doors.length) nd++; }
    res.noDoor = `${nd}/${tot}`;
    out.push(res);
  }
  return out;
}, only);
console.table(r);
console.log(logs.filter(l => l.startsWith('ERR')).join('\n'));
await b.close();
