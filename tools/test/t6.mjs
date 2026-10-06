import { open } from './h.mjs';
import fs from 'fs';
const { b, pg } = await open();
const [name, k] = [process.argv[2], +process.argv[3]];
const r = await pg.evaluate(([n, k]) => { const I = GTA.interiors.find(i => i.name.includes(n)); return GTA.plan(I.name, k); }, [name, k]);
fs.writeFileSync('plan.json', JSON.stringify(r)); await b.close();
