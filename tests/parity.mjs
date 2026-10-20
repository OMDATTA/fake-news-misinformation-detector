// Checks that the browser model (web/assets/js/model.js) reproduces the
// Python pipeline's calibrated probabilities on 25 held-out test articles.
// Run:  node tests/parity.mjs        (after scripts/05_calibrate_export.py)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, '../web/assets/js/model.js'), 'utf8');
const { Detector } = await import('data:text/javascript,' + encodeURIComponent(src));
const model = JSON.parse(readFileSync(join(here, '../web/assets/data/model.json'), 'utf8'));
const fixtures = JSON.parse(readFileSync(join(here, 'fixtures.json'), 'utf8'));

const det = new Detector(model);
let worst = 0, worstLogit = 0, flips = 0;
for (const f of fixtures) {
  const r = det.analyze(f.text);
  worst = Math.max(worst, Math.abs(r.p - f.p));
  worstLogit = Math.max(worstLogit, Math.abs(r.logit - f.logit));
  if ((r.p >= 0.5) !== (f.p >= 0.5)) flips++;
}
console.log(`fixtures=${fixtures.length} max|dp|=${worst.toExponential(2)} max|dlogit|=${worstLogit.toExponential(2)} label flips=${flips}`);
if (worst > 1e-3 || flips > 0) { console.error('PARITY FAILED'); process.exit(1); }
console.log('PARITY OK');
