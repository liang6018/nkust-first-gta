import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
import fs from 'fs';
const T = '/tmp/claude-0/tj/t/';
export async function open(file = '/home/claude/nkust-first-gta/index.html') {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const pg = await b.newPage({ viewport: { width: 900, height: 500 } });
  const logs = []; pg.on('console', m => logs.push(m.type() + ': ' + m.text())); pg.on('pageerror', e => logs.push('ERR ' + e.message));
  await pg.route('**/*', r => {
    const u = r.request().url();
    if (u.startsWith('file:')) return r.continue();
    const m = u.match(/three@0\.165\.0\/(.*)$/);
    if (m) return r.fulfill({ body: fs.readFileSync(T + m[1]), contentType: 'application/javascript' });
    return r.abort();
  });
  await pg.goto('file://' + file);
  await pg.waitForFunction(() => window.GTA && window.GTA.built && window.GTA.built.length > 10, null, { timeout: 180000 }).catch(e => logs.push('TIMEOUT ' + e.message));
  return { b, pg, logs };
}
