import { Detector, segmentForHighlight } from './model.js';

const $ = (s) => document.querySelector(s);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const SAMPLES = {
  wire: `City council approves $48 million plan to replace aging water mains.
The Harbor City council voted 7-2 on Tuesday to approve a $48 million program to replace roughly 30 miles of water pipes installed before 1960, officials said. The plan, which will be funded through a mix of state loans and a phased rate increase, is expected to take six years to complete, according to a statement from the city's public works department. "These pipes are well past their design life," said Maria Chen, the department's director, adding that the city recorded 214 main breaks last year. Two council members who voted against the measure said they supported the repairs but objected to the size of the rate increase, which the department estimated at about $4 a month for a typical household. A public hearing on the rate schedule is scheduled for March 12.`,
  sensational: `SHOCKING: They Don't Want You To Know What's REALLY In Your Tap Water!!!
You won't believe what the so-called "experts" at city hall are hiding from you. This is absolutely outrageous. They're spending YOUR money on a massive pipe scam while your family drinks poison every single day. Wake up, people! Just look at what they did. The corrupt politicians are lying to your face and the mainstream media is totally silent. Why won't anyone ask the obvious question? Who profits? We all know the answer. Share this everywhere before they delete it!!! The truth is finally exposed and they are terrified. This is a disgrace and every single one of them should be thrown out.`,
  mixed: `Opinion: Our city's water plan is overdue, and still not enough.
For years we have been told that fixing the water system can wait. It can't. The council's vote this week to replace 30 miles of old pipe is a welcome step, but I worry it is far too little and far too slow. Six years is a long time to ask families to trust pipes that are older than most of the people who will pay for them. I spoke with neighbors who have had brown water three times this winter. They are not asking for miracles. They are asking for a plan that matches the scale of the problem. If the council is serious, it should publish the full list of streets, the timeline for each, and a clear explanation of how the new rates were calculated. We deserve to see the numbers.`,
};

const BAND = [
  [0.85, 'Strongly resembles misleading examples', true],
  [0.65, 'Leans toward misleading examples', true],
  [0.35, 'Mixed signal — inconclusive', false],
  [0.15, 'Leans toward reliable examples', false],
  [-1, 'Strongly resembles reliable examples', false],
];

let det = null;
let report = null;

function fmtLing(name, v) {
  switch (name) {
    case 'log_words': return Math.round(Math.expm1(v)).toLocaleString();
    case 'type_token_ratio': return v.toFixed(2);
    case 'caps_ratio': case 'digit_ratio': return pct(v);
    case 'avg_word_len': case 'avg_sentence_len': return v.toFixed(1);
    case 'flesch': return v.toFixed(0);
    default: return v.toFixed(2);
  }
}

// Never display 0% or 100%: the model cannot be certain.
function shownP(p, d = 1) {
  if (p > 0.99) return '>99%';
  if (p < 0.01) return '<1%';
  return pct(p, d);
}

function wordCount(t) { return (t.match(/[\p{L}\p{N}']+/gu) || []).length; }

function renderResult(text) {
  const r = det.analyze(text);
  const p = r.p;
  const covMin = det.m.train_vocab_coverage_p05 || 0.9;
  // Refuse to give a verdict when there is too little familiar text to judge.
  const insufficient = r.nWords < 40 || r.coverage < covMin * 0.8;
  let [, band, isMis] = BAND.find(([t]) => p >= t);
  if (insufficient) { band = 'Not enough familiar text to estimate'; isMis = false; }
  $('#band').textContent = band;
  $('#band').classList.toggle('is-mis', isMis);
  $('#body').classList.toggle('is-insufficient', insufficient);
  $('#prob').textContent = shownP(p);
  $('#marker').style.left = `${(p * 100).toFixed(1)}%`;
  $('#markerLabel').textContent = shownP(p, 0);

  // what the calibrated number means, using the test-set reliability bins
  const bins = report?.calibration?.reliability?.platt || [];
  const bin = bins.find((b) => Math.floor(Math.min(p, 0.9999) * 10) === b.bin);
  let ex = '';
  if (bin) {
    ex = `Among held-out test articles that received a similar score (${bin.bin * 10}–${bin.bin * 10 + 10}%), ${pct(bin.frac_pos, 0)} came from misleading sources (n = ${bin.count.toLocaleString()}${bin.count < 50 ? ', a small sample' : ''}). `;
  }
  const oodAcc = report ? pct(report.ood.scores.accuracy, 0) : '74%';
  ex += `That calibration holds for the kinds of outlets it was trained on; on unfamiliar outlets the model was right about ${oodAcc} of the time.`;
  $('#explain').textContent = ex;

  // warnings
  const warns = [];
  if (insufficient) warns.push('The score above is shown for transparency only. Paste a longer English news article for a meaningful estimate.');
  if (r.nWords < 80) warns.push(`Short text (${r.nWords} words). Estimates on fewer than ~80 words are unstable.`);
  if (r.coverage < covMin) warns.push(`Only ${pct(r.coverage, 0)} of the words are in the model's vocabulary (typical news text: ${pct(covMin, 0)}+). The text may be non-English or far from 2015–2018 news, so treat the score as unreliable.`);
  if (r.nWords >= 80 && r.coverage >= covMin && Math.abs(p - 0.5) > 0.45) {
    warns.push('High confidence reflects style similarity, not verified facts.');
  }
  $('#warnings').innerHTML = warns.map((w) => `<li>${esc(w)}</li>`).join('');

  // highlighted text
  const segs = segmentForHighlight(text, r.tokenScore);
  const maxAbs = Math.max(1e-9, ...segs.map((s) => Math.abs(s.score)));
  $('#highlight').innerHTML = segs.map((s) => {
    if (!s.word || Math.abs(s.score) < maxAbs * 0.2) return esc(s.text);
    const a = Math.min(1, Math.abs(s.score) / maxAbs) * 0.8 + 0.15;
    const cls = s.score > 0 ? 'm' : `r${a > 0.55 ? ' strong' : ''}`;
    const dir = s.score > 0 ? 'misleading' : 'reliable';
    return `<mark class="${cls}" style="--a:${a.toFixed(2)}" title="${s.score > 0 ? '+' : ''}${s.score.toFixed(3)} toward ${dir}">${esc(s.text)}</mark>`;
  }).join('');

  // top phrases
  const pos = r.terms.filter((t) => t.contrib > 0).sort((a, b) => b.contrib - a.contrib).slice(0, 8);
  const neg = r.terms.filter((t) => t.contrib < 0).sort((a, b) => a.contrib - b.contrib).slice(0, 8);
  const m = Math.max(1e-9, ...[...pos, ...neg].map((t) => Math.abs(t.contrib)));
  const bar = (list) => list.length
    ? list.map((t) => `<li><i class="fill" style="width:${(Math.abs(t.contrib) / m * 100).toFixed(1)}%"></i><span>${esc(t.term)}${t.count > 1 ? ` <span class="muted">×${t.count}</span>` : ''}</span><span class="v">${t.contrib > 0 ? '+' : ''}${t.contrib.toFixed(3)}</span></li>`).join('')
    : '<li class="none">No strong phrases.</li>';
  $('#toMis').innerHTML = bar(pos);
  $('#toRel').innerHTML = bar(neg);

  // linguistic table
  const lings = [...r.ling].sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
  const lm = Math.max(1e-9, ...lings.map((l) => Math.abs(l.contrib)));
  $('#ling tbody').innerHTML = lings.map((l) => `
    <tr>
      <td>${esc(l.label)}</td>
      <td class="num">${fmtLing(l.name, l.value)}</td>
      <td class="num muted">${fmtLing(l.name, l.typical)}</td>
      <td><div class="eff"><i class="${l.contrib < 0 ? 'rel' : ''}" style="width:${Math.max(2, Math.abs(l.contrib) / lm * 56).toFixed(0)}px"></i><small title="${l.contrib >= 0 ? 'toward misleading' : 'toward reliable'}">${l.contrib > 0 ? '+' : ''}${l.contrib.toFixed(2)}</small></div></td>
    </tr>`).join('');

  $('#empty').hidden = true;
  const body = $('#body');
  body.hidden = false;
  body.classList.remove('is-fresh'); void body.offsetWidth; body.classList.add('is-fresh');
}

// ------------------------------------------------------------- report UI
function fillNumbers(rep) {
  const deployed = rep.models.find((m) => m.name.includes('deployed'));
  const set = (key, val) => document.querySelectorAll(`[data-fill="${key}"]`).forEach((el) => { el.textContent = val; });
  set('marker_acc', pct(rep.marker_only_cv_accuracy, 0));
  set('n_test', rep.models.length ? (rep.data.split_counts.test).toLocaleString() : '');
  set('ood_n', rep.ood.n.toLocaleString());
  set('acc_in', pct(deployed.accuracy));
  set('acc_ood', pct(rep.ood.scores.accuracy));
  set('emb_model', rep.embeddings.model.split('/').pop());
  set('prev', pct(rep.imbalance.prevalence, 0));
  set('ece_in', pct(rep.calibration.platt.ece));
  set('ece_ood', pct(rep.ood.scores.ece, 0));
  document.querySelector('[data-stat="indist"]').textContent = pct(deployed.accuracy);
  document.querySelector('[data-stat="ood"]').textContent = pct(rep.ood.scores.accuracy);

  const rows = [...rep.models].sort((a, b) => b.macro_f1 - a.macro_f1);
  $('#cmp tbody').innerHTML = rows.map((m) => `
    <tr class="${m.name.includes('deployed') ? 'deployed' : ''}">
      <td>${esc(m.name.replace(' (deployed)', ''))}</td><td class="fam">${esc(m.family)}</td>
      <td class="num">${pct(m.accuracy)}</td><td class="num">${m.macro_f1.toFixed(3)}</td>
      <td class="num">${m.roc_auc.toFixed(3)}</td><td class="num">${m.brier.toFixed(4)}</td><td class="num">${pct(m.ece, 2)}</td>
    </tr>`).join('');

  const e = rep.embeddings;
  $('#emb tbody').innerHTML = Object.keys(e.in_dist).map((k) => `
    <tr><td>${esc(k.replace(' (same 10k)', ''))}</td><td class="num">${pct(e.in_dist[k].accuracy)}</td>
    <td class="num">${e.ood[k] ? pct(e.ood[k].accuracy) : '—'}</td><td class="num">${e.ood[k] ? e.ood[k].roc_auc.toFixed(3) : '—'}</td></tr>`).join('');

  $('#imb tbody').innerHTML = Object.entries(rep.imbalance.results).map(([k, v]) => `
    <tr><td>${esc(k)}</td><td class="num">${v.precision.toFixed(3)}</td><td class="num">${v.recall.toFixed(3)}</td>
    <td class="num">${v.f1.toFixed(3)}</td><td class="num">${v.pr_auc.toFixed(3)}</td></tr>`).join('');

  const c = rep.calibration;
  $('#calText').textContent = `Expected calibration error: ${pct(c.uncalibrated.ece, 2)} raw → ${pct(c.platt.ece, 2)} after Platt scaling (isotonic: ${pct(c.isotonic.ece, 2)}). Brier score ${c.platt.brier.toFixed(4)}.`;
  $('#oodText').textContent = `ROC-AUC falls to ${rep.ood.scores.roc_auc.toFixed(2)} and calibration error rises to ${pct(rep.ood.scores.ece, 0)}. ${pct(rep.ood.share_confident, 0)} of these predictions were still above 90% or below 10%: the model is overconfident on writing it has not seen.`;
}

function charts(rep) {
  if (!window.Chart) return;
  const C = window.Chart;
  C.defaults.color = '#a9a49e';
  C.defaults.font.family = "'Satoshi', 'Inter', sans-serif";
  C.defaults.font.size = 12;
  C.defaults.borderColor = 'rgba(255,255,255,0.08)';
  C.defaults.plugins.legend.labels.boxWidth = 12;
  const RED = '#e5202e', LIGHT = '#d9d5cf', GREY = '#6f6b67';
  const pts = (rb) => rb.map((b) => ({ x: b.mean_pred, y: b.frac_pos }));
  const diag = { label: 'Perfect calibration', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: GREY, borderDash: [5, 5], pointRadius: 0, showLine: true, borderWidth: 1 };
  const axes = {
    x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Predicted P(misleading)' }, ticks: { callback: (v) => `${v * 100}%` } },
    y: { min: 0, max: 1, title: { display: true, text: 'Observed share misleading' }, ticks: { callback: (v) => `${v * 100}%` } },
  };
  new C($('#chartCal'), {
    type: 'scatter',
    data: { datasets: [diag,
      { label: 'Raw logistic output', data: pts(rep.calibration.reliability.uncalibrated), borderColor: LIGHT, backgroundColor: LIGHT, showLine: true, borderWidth: 1.2, pointRadius: 3 },
      { label: 'Platt-calibrated', data: pts(rep.calibration.reliability.platt), borderColor: RED, backgroundColor: RED, showLine: true, borderWidth: 2.4, pointRadius: 4 }] },
    options: { maintainAspectRatio: false, scales: axes },
  });
  new C($('#chartOod'), {
    type: 'scatter',
    data: { datasets: [diag,
      { label: 'Unfamiliar sources (WELFake)', data: pts(rep.ood.reliability), borderColor: RED, backgroundColor: RED, showLine: true, borderWidth: 2.4, pointRadius: 4 }] },
    options: { maintainAspectRatio: false, scales: axes },
  });
  const sm = rep.source_markers;
  const SHORT = {
    "Mentions 'Reuters'": 'Mentions Reuters',
    "Image credit ('Featured image', 'Photo by')": 'Image credit',
    'Embedded tweet / handle': 'Tweet / @handle',
    'Contains URL': 'URL',
    'Curly quotes / apostrophes': 'Curly quotes',
    'Title in ALL CAPS words (3+)': 'ALL-CAPS title',
    'Title tag like [VIDEO] / (IMAGES)': '[VIDEO]-style tag',
  };
  new C($('#chartSrc'), {
    type: 'bar',
    data: { labels: sm.map((s) => SHORT[s.marker] || s.marker), datasets: [
      { label: 'Reliable', data: sm.map((s) => s.reliable_pct), backgroundColor: LIGHT },
      { label: 'Misleading', data: sm.map((s) => s.misleading_pct), backgroundColor: RED }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, scales: { x: { max: 100, ticks: { callback: (v) => `${v}%` } }, y: { ticks: { autoSkip: false } } } },
  });
  const tp = [...rep.topics].sort((a, b) => b.misleading_share - a.misleading_share);
  new C($('#chartTopic'), {
    type: 'bar',
    data: { labels: tp.map((t) => t.top_words.slice(0, 3).join(', ')), datasets: [
      { label: 'Misleading share', data: tp.map((t) => t.misleading_share * 100), backgroundColor: tp.map((t) => (t.misleading_share > 0.5 ? RED : '#8c1119')) }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (ctx) => `n = ${tp[ctx.dataIndex].n.toLocaleString()}` } } }, scales: { x: { max: 100, ticks: { callback: (v) => `${v}%` } }, y: { ticks: { autoSkip: false } } } },
  });
  const ho = rep.topic_holdout;
  new C($('#chartHold'), {
    type: 'bar',
    data: { labels: ho.map((h) => h.label.split(', ').slice(0, 3).join(', ')), datasets: [
      { label: 'Topic seen in training', data: ho.map((h) => h.acc_seen * 100), backgroundColor: LIGHT },
      { label: 'Topic held out', data: ho.map((h) => h.acc_held_out * 100), backgroundColor: RED }] },
    options: { indexAxis: 'y', maintainAspectRatio: false, scales: { x: { min: 90, max: 100, ticks: { callback: (v) => `${v}%` } }, y: { ticks: { autoSkip: false } } } },
  });
}

// ------------------------------------------------------------------ boot
async function boot() {
  const text = $('#text');
  const run = $('#run');
  const update = () => { $('#wordcount').textContent = `${wordCount(text.value).toLocaleString()} words`; };
  text.addEventListener('input', update);
  document.querySelectorAll('[data-sample]').forEach((b) => b.addEventListener('click', () => {
    text.value = SAMPLES[b.dataset.sample]; update();
    if (det) renderResult(text.value);
  }));
  $('#clear').addEventListener('click', () => {
    text.value = ''; update(); $('#body').hidden = true; $('#empty').hidden = false; text.focus();
  });
  $('#form').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!det || !text.value.trim()) return;
    renderResult(text.value);
    if (window.innerWidth < 960) $('#result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });

  try {
    const [m, r] = await Promise.all([
      fetch('assets/data/model.json').then((x) => x.json()),
      fetch('assets/data/report.json').then((x) => x.json()),
    ]);
    det = new Detector(m);
    report = r;
    run.disabled = false; run.textContent = 'Analyze';
    fillNumbers(r);
    const draw = () => charts(r);
    if (window.Chart) draw(); else window.addEventListener('load', draw, { once: true });
  } catch (err) {
    console.error(err);
    run.textContent = 'Model failed to load';
  }
}
boot();
