import { open } from './h.mjs';
const { b, pg, logs } = await open();
const r = await pg.evaluate(() => {
  const out = [];
  const I = GTA.interiors.find(i => i.name === '管理學院'); const G = I.plan.grids[1];
  // a corridor cell on floor 2
  let k0 = -1; for (let k = 0; k < G.lab.length; k++) if (G.lab[k] === 0 && G.room[k] < 0 && G.dist[k] > 3) { k0 = k; break; }
  const u = G.u0 + (k0 % G.nu + .5) * .5, v = G.v0 + (Math.floor(k0 / G.nu) + .5) * .5, x = G.F.ox + G.F.U[0] * u + G.F.V[0] * v, z = G.F.oz + G.F.U[1] * u + G.F.V[1] * v;
  GTA.tp(x, z); GTA.player.pos.y = I.levels[1];
  for (const t of [10 * 60 + 30, 10 * 60 + 2, 12 * 60 + 30]) { GTA.setTime(t); out.push(GTA.life()); for (let i = 0; i < 120; i++) GTA.life(); out.push(GTA.life()); }
  return out;
});
console.log(JSON.stringify(r, null, 0).slice(0, 3000));
console.log(logs.filter(l => l.startsWith('ERR') || l.startsWith('error')).join('\n'));
await b.close();
