// In-browser inference for the Fake News & Misinformation Detector.
// Mirrors src/fnd/text.py and src/fnd/features.py exactly (see tests/parity.mjs).
// No text ever leaves the browser.

// ---------------------------------------------------------------- scrubbing
const SCRUB_I = [
  /https?:\/\/\S+/gi,
  /www\.\S+/gi,
  /pic\.twitter\.com\/\S+/gi,
  /@\w+/gi,
  /\(reuters\)/gi,
  /\breuters\b/gi,
  /21st century wire/gi,
  /featured image[^\n.]*/gi,
  /photo by[^\n.]*/gi,
  /image via[^\n.]*/gi,
  /getty images/gi,
  /\bvia\s+(?:twitter|youtube|facebook)\b/gi,
  /[\[(](?:video|videos|image|images|tweet|tweets|watch|photo|photos)[\])]/gi,
  /\bbit\.ly\/?\S*/gi,
  /\b\d{1,4}\s?(?:edt|est|gmt|bst|cet)\b/gi,
  /\bfactbox\b/gi,
  /\bread more\b:?/gi,
  /\bvia\b/gi,
];
const DATELINE = /^\s*[A-Z][A-Z .,\/'-]{2,40}\s+-\s+/;

export function asciiFold(text) {
  return text
    .replace(/[\u2019\u2018]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, '-')
    .normalize('NFKD')
    .replace(/[^\x00-\x7F]/g, '');
}

export function scrub(text) {
  let t = asciiFold(text || '');
  t = t.replace(DATELINE, ' ');
  for (const rx of SCRUB_I) t = t.replace(rx, ' ');
  t = t.replace(/'/g, ' ');
  return t.replace(/\s+/g, ' ').trim();
}

// ------------------------------------------------------------- tokenising
export function forTfidf(clean) {
  return clean.toLowerCase().replace(/[0-9]/g, '0');
}
export function tokenize(clean) {
  return forTfidf(clean).match(/[a-z0-9_]{2,}/g) || [];
}

// ---------------------------------------------------- linguistic features
const FIRST_PERSON = new Set(['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']);
const SECOND_PERSON = new Set(['you', 'your', 'yours', 'ya']);
const SENSATIONAL = new Set([
  'shocking', 'shocked', 'outrageous', 'unbelievable', 'insane', 'disgusting',
  'destroy', 'destroys', 'destroyed', 'slam', 'slams', 'slammed', 'bombshell',
  'horrifying', 'epic', 'hilarious', 'pathetic', 'idiot', 'idiots', 'stupid',
  'moron', 'lies', 'liar', 'lying', 'disgrace', 'disgraceful', 'breaking',
  'exposed', 'exposes', 'busted', 'meltdown', 'furious', 'rant', 'brutal',
  'brutally', 'humiliating', 'humiliated', 'wow', 'absolutely', 'totally',
  'literally', 'completely', 'massive', 'huge', 'hysterical', 'crazy',
]);
const HEDGES = new Set([
  'reportedly', 'allegedly', 'may', 'might', 'could', 'possibly', 'perhaps',
  'appears', 'appeared', 'suggests', 'suggested', 'likely', 'unlikely',
  'apparently', 'estimated', 'approximately',
]);
const ATTRIBUTION = new Set([
  'said', 'says', 'told', 'according', 'stated', 'spokesman', 'spokeswoman',
  'spokesperson', 'statement', 'announced', 'confirmed', 'official',
  'officials', 'added',
]);

const count = (s, ch) => s.split(ch).length - 1;

export function linguisticFeatures(clean) {
  const words = clean.match(/[A-Za-z0-9]+/g) || [];
  const n = words.length;
  const nz = Math.max(n, 1);
  const lower = words.map((w) => w.toLowerCase());
  const nSent = Math.max(clean.split(/[.!?]+/).filter((s) => s.trim()).length, 1);
  const head = lower.slice(0, 300);
  const ttr = new Set(head).size / Math.max(head.length, 1);
  let caps = 0, syl = 0, digits = 0, chars = 0;
  for (const w of words) {
    chars += w.length;
    if (w.length >= 2 && w === w.toUpperCase() && /[A-Z]/.test(w)) caps++;
    if (/[0-9]/.test(w)) digits++;
  }
  for (const w of lower) syl += Math.max(1, (w.match(/[aeiouy]+/g) || []).length);
  const per100 = 100 / nz;
  const rate = (lex) => lower.reduce((a, w) => a + (lex.has(w) ? 1 : 0), 0) * per100;
  let flesch = 206.835 - 1.015 * (n / nSent) - 84.6 * (syl / nz);
  flesch = Math.min(Math.max(flesch, -50), 120);
  return [
    Math.log1p(n),
    chars / nz,
    ttr,
    Math.min(n / nSent, 120),
    count(clean, '!') * per100,
    count(clean, '?') * per100,
    caps / nz,
    count(clean, '"') * per100,
    rate(FIRST_PERSON),
    rate(SECOND_PERSON),
    rate(SENSATIONAL),
    rate(HEDGES),
    rate(ATTRIBUTION),
    digits / nz,
    flesch,
  ];
}

// ------------------------------------------------------------------ model
export class Detector {
  constructor(model) {
    this.m = model;
    this.index = new Map();
    model.tfidf.terms.forEach((t, i) => this.index.set(t, i));
  }

  analyze(rawText) {
    const m = this.m;
    const clean = scrub(rawText);
    const toks = tokenize(clean);

    // n-gram counts restricted to the vocabulary
    const counts = new Map();
    const bump = (term) => {
      const i = this.index.get(term);
      if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
    };
    toks.forEach(bump);
    for (let k = 0; k + 1 < toks.length; k++) bump(toks[k] + ' ' + toks[k + 1]);

    const weights = new Map();
    let norm = 0;
    for (const [i, c] of counts) {
      const w = (1 + Math.log(c)) * m.tfidf.idf[i];
      weights.set(i, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;

    let logit = m.intercept;
    const termContrib = [];
    for (const [i, w] of weights) {
      const c = m.tfidf.coef[i] * (w / norm);
      logit += c;
      termContrib.push({ term: m.tfidf.terms[i], contrib: c, count: counts.get(i) });
    }
    const tfidfLogit = logit - m.intercept;

    const raw = linguisticFeatures(clean);
    const L = m.ling;
    const ling = raw.map((x, j) => {
      let z = (x - L.mean[j]) / L.scale[j];
      z = Math.max(-L.clip, Math.min(L.clip, z));
      const c = L.coef[j] * z * L.weight;
      logit += c;
      return { name: L.names[j], label: L.labels[j], value: x, z, contrib: c, typical: L.mean[j] };
    });

    const { a, b } = m.calibration;
    const p = 1 / (1 + Math.exp(-(a * logit + b)));
    const pRaw = 1 / (1 + Math.exp(-logit));

    // per-token-type score for highlighting: unigram + half of each bigram
    const tokenScore = new Map();
    const tokenCount = new Map();
    toks.forEach((t) => tokenCount.set(t, (tokenCount.get(t) || 0) + 1));
    for (const tc of termContrib) {
      const parts = tc.term.split(' ');
      const share = parts.length === 1 ? tc.contrib : tc.contrib / 2;
      for (const part of parts) tokenScore.set(part, (tokenScore.get(part) || 0) + share);
    }
    for (const [t, s] of tokenScore) tokenScore.set(t, s / (tokenCount.get(t) || 1));

    const uni = toks.length;
    const covered = toks.filter((t) => this.index.has(t)).length;
    termContrib.sort((x, y) => Math.abs(y.contrib) - Math.abs(x.contrib));

    return {
      p, pRaw, logit, tfidfLogit,
      nWords: (clean.match(/[A-Za-z0-9]+/g) || []).length,
      coverage: uni ? covered / uni : 0,
      terms: termContrib,
      ling,
      tokenScore,
      clean,
    };
  }
}

// Turn the per-token scores into highlighted segments of the ORIGINAL text.
export function segmentForHighlight(rawText, tokenScore) {
  const out = [];
  const rx = /[\p{L}\p{N}_'\u2019]+/gu;
  let last = 0, mt;
  while ((mt = rx.exec(rawText)) !== null) {
    if (mt.index > last) out.push({ text: rawText.slice(last, mt.index), score: 0, word: false });
    const word = mt[0];
    const subs = forTfidf(asciiFold(word).replace(/'/g, ' ')).match(/[a-z0-9_]{2,}/g) || [];
    let s = 0;
    for (const sub of subs) s += tokenScore.get(sub) || 0;
    out.push({ text: word, score: s, word: true });
    last = mt.index + word.length;
  }
  if (last < rawText.length) out.push({ text: rawText.slice(last), score: 0, word: false });
  return out;
}
