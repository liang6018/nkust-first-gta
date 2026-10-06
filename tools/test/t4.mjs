import { open } from './h.mjs';
import fs from 'fs';
const { b, pg, logs } = await open();
const name = process.argv[2], fk = +(process.argv[3] || 0);
const r = await pg.evaluate(([name, fk]) => {
  const I = GTA.interiors.find(i => i.name.includes(name)), G = I.plan.grids[fk];
  const cellOf = (x, z) => { const dx = x - G.F.ox, dz = z - G.F.oz; const u = dx * G.F.U[0] + dz * G.F.U[1], v = dx * G.F.V[0] + dz * G.F.V[1]; const i = Math.floor((u - G.u0) / .5), j = Math.floor((v - G.v0) / .5); return i + j * G.nu; };
  const mark = {}; I.doorsXZ.forEach(([x, z], i) => mark[cellOf(x, z)] = 'D');
  I.cores.forEach((c, i) => { mark[cellOf(c.ox - c.A[0], c.oz - c.A[1])] = 'T'; });
  const B = GTA.built.find(b => b.interior === I), E = B.entrance, [dx0, dz0] = I.doorsXZ[0];
  const N = G.nu * G.nv, walk = k => k >= 0 && G.lab[k] === 0 && G.room[k] < 0 && G.dist[k] >= .45, s0 = cellOf(dx0 - E.nx * 1.2, dz0 - E.nz * 1.2), seen = new Uint8Array(N), Q = [s0]; seen[s0] = 1;
  for (let h = 0; h < Q.length; h++) { const k = Q[h], i = k % G.nu; for (const m of [i > 0 ? k - 1 : -1, i < G.nu - 1 ? k + 1 : -1, k - G.nu, k + G.nu]) if (m >= 0 && m < N && !seen[m] && walk(m)) { seen[m] = 1; Q.push(m); } }
  const rows = [];
  for (let j = 0; j < G.nv; j++) { let s = ''; for (let i = 0; i < G.nu; i++) { const k = i + j * G.nu; s += mark[k] || (G.lab[k] < 0 ? ' ' : G.lab[k] === 1 ? '#' : G.room[k] >= 0 ? 'abcdefghijklmnopqrstuvwxyz'[G.room[k] % 26] : seen[k] ? '+' : G.dist[k] < .45 ? ',' : G.code[k] === 1 ? '.' : ':'); } rows.push(s); }
  return rows.join('\n');
}, [name, fk]);
fs.writeFileSync('grid.txt', r); await b.close();
