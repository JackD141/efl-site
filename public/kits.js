// Club shirts drawn from the club's colours (EFL squad data: abbr / color / textColor in the plan JSONs) plus a table of
// well-known patterns. Used by the homepage (via xp-core.js) and the Player Picks tables.
/* ---------- kits: club colours from the EFL squad data, plus patterns / fixes for well-known kits (approximate) ---------- */
const KITS = {
  LIN: ['#e10613', '#ffffff', 'stripes'], SHU: ['#ed1c24', '#ffffff', 'stripes'], STO: ['#d7172f', '#ffffff', 'stripes'],
  SOU: ['#e3051b', '#ffffff', 'stripes'], EXE: ['#e1211c', '#ffffff', 'stripes'], SHW: ['#0971ce', '#ffffff', 'stripes'],
  HUD: ['#0971ce', '#ffffff', 'stripes'], WIG: ['#00539e', '#ffffff', 'stripes'], COL: ['#005eb8', '#ffffff', 'stripes'],
  GRI: ['#111111', '#ffffff', 'stripes'], NOT: ['#111111', '#ffffff', 'stripes'], BRA: ['#72253d', '#f2b51c', 'stripes'],
  WBA: ['#122f67', '#ffffff', 'stripes'], CLT: ['#e1231b', '#ffffff', 'stripes'],
  BLA: ['#014898', '#ffffff', 'halves'], BRR: ['#1a51a0', '#ffffff', 'quarters'], WYC: ['#55b1e2', '#0b1f4b', 'quarters'],
  QPR: ['#0054a2', '#ffffff', 'hoops'], REA: ['#0133a0', '#ffffff', 'hoops'], DON: ['#e2211c', '#ffffff', 'hoops'],
  BUR: ['#6c1d45', '#99d6ea', 'sleeves'], WHU: ['#7a263a', '#1bb1e7', 'sleeves'], FLE: ['#e1211c', '#ffffff', 'sleeves'],
  ROT: ['#e1211c', '#ffffff', 'sleeves'], WAT: ['#fbee23', '#111111', 'sleeves'], NOR: ['#fff200', '#00a650', 'sleeves'],
  WOL: ['#fdb913', '#231f20', 'plain'], PNE: ['#ffffff', '#0e1d49', 'plain'], BOL: ['#ffffff', '#06205c', 'plain'],
  TRA: ['#ffffff', '#001489', 'plain'], PVL: ['#ffffff', '#111111', 'plain'], MKD: ['#ffffff', '#e30613', 'plain'],
  DER: ['#ffffff', '#111111', 'plain'], SWA: ['#ffffff', '#111111', 'plain'], BRO: ['#ffffff', '#111111', 'plain'],
};
function kitForClub(c) {
  c = c || {};
  const k = KITS[c.abbr];
  return k ? { a: k[0], b: k[1], pattern: k[2] } : { a: c.color || '#1f3d7a', b: c.textColor || '#ffffff', pattern: 'plain' };
}
const SHIRT_PATH = 'M21 3 L7 11 L2 26 L13 29 L13 57 L51 57 L51 29 L62 26 L57 11 L43 3 Q32 12 21 3 Z';
let shirtSeq = 0;
function shirtSvgFor(c, keeper) {
  let { a, b, pattern } = kitForClub(c);
  if (keeper) { b = a; a = '#c6e33a'; pattern = 'sleeves'; } // keepers: a generic keeper kit with club-colour sleeves
  const id = 'sh' + (++shirtSeq);
  const over = {
    plain: '',
    stripes: [16, 28, 40].map(x => `<rect x="${x}" y="0" width="7" height="60" fill="${b}"/>`).join(''),
    hoops: [14, 28, 42].map(y => `<rect x="0" y="${y}" width="64" height="7" fill="${b}"/>`).join(''),
    halves: `<rect x="32" y="0" width="32" height="60" fill="${b}"/>`,
    quarters: `<rect x="32" y="0" width="32" height="30" fill="${b}"/><rect x="0" y="30" width="32" height="30" fill="${b}"/>`,
    sleeves: `<path d="M7 11 L2 26 L13 29 L16 14 Z M57 11 L62 26 L51 29 L48 14 Z" fill="${b}"/>`,
  }[pattern] || '';
  return `<svg viewBox="0 0 64 60" aria-hidden="true"><defs><clipPath id="${id}"><path d="${SHIRT_PATH}"/></clipPath></defs>
    <g clip-path="url(#${id})"><rect x="0" y="0" width="64" height="60" fill="${a}"/>${over}</g>
    <path d="${SHIRT_PATH}" fill="none" stroke="rgba(0,0,0,0.35)" stroke-width="1.5"/></svg>`;
}
// a small shirt for tables: <span class="pp-shirt">svg</span>
const shirtIcon = (c, keeper) => `<span class="pp-shirt" title="${(c && c.name) || ''}">${shirtSvgFor(c, keeper)}</span>`;
// last-5-games minutes as small chips (dark = 60+, light = 1-59, grey = 0), oldest first
function recentHtml(list) {
  if (!list || !list.length) return '<span class="cp-muted cp-small">–</span>';
  return '<span class="pp-mins">' + list.map(m => `<span class="pp-m ${m >= 60 ? 'pp-m-full' : m > 0 ? 'pp-m-part' : 'pp-m-zero'}">${m}</span>`).join('') + '</span>';
}
// set-piece tags as a small vertical list next to the shirt; the reason shows on hover (title) and tap
const SP_SHORT = { Pens: 'PEN', Corners: 'CRN', FKs: 'FK' };
function spList(tags) {
  if (!tags || !tags.length) return '';
  return '<span class="pp-sp-list">' + tags.map(x => {
    const t = typeof x === 'string' ? x : x.t, why = typeof x === 'string' ? '' : x.why;
    return `<span class="pp-sp-tag" title="${t}: ${why}">${SP_SHORT[t] || t}</span>`;
  }).join('') + '</span>';
}
const spText = tags => (tags || []).map(x => typeof x === 'string' ? x : `${x.t} (${x.why})`).join('; ');
// opponent name: full on wide screens, the club's 3-letter code on phones (pages fill ABBR from their club list)
const ABBR = {};
const fxName = (name, short) => `<span class="fx-long">${short}</span><span class="fx-abbr" title="${short}">${ABBR[name] || short.slice(0, 3).toUpperCase()}</span>`;
