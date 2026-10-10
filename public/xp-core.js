// Shared by the Optimal Picks homepage (optimal.js) and the Player Picks "All" tab (allpicks.js): loads the position and
// club plans and computes each player's expected points with the same maths as the position pages (keepers.js /
// defenders.js / attackers.js), including minutes edited on those pages (saved in this browser).
// Functions use the page's global `state` (plans, clubs, week, players, shortByName, abbrByName, kitById).
const POSITIONS = ['GK', 'DEF', 'MID', 'FWD'];
const POS_NAME = { GK: 'Goalkeeper', DEF: 'Defender', MID: 'Midfielder', FWD: 'Forward' };
const POS_PAGE = { GK: 'picks.html?pos=GK', DEF: 'picks.html?pos=DEF', MID: 'picks.html?pos=MID', FWD: 'picks.html?pos=FWD' };
const MINS_KEYS = { GK: 'efl_keeper_mins_v1', DEF: 'efl_defender_mins_v1', MID: 'efl_mid_mins_v1', FWD: 'efl_fwd_mins_v1' };

const shortOf = name => state.shortByName[name] || name;
function abbr(name) {
  if (state.abbrByName[name]) return state.abbrByName[name];
  const w = shortOf(name).replace(/[^A-Za-z ]/g, '').split(' ').filter(Boolean);
  return (w.length === 1 ? w[0].slice(0, 3) : w[0].slice(0, 2) + w[1][0]).toUpperCase();
}

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
function kitOf(clubId) {
  const c = state.kitById[clubId] || {};
  const k = KITS[c.abbr];
  return k ? { a: k[0], b: k[1], pattern: k[2] } : { a: c.color || '#1f3d7a', b: c.textColor || '#ffffff', pattern: 'plain' };
}
const SHIRT_PATH = 'M21 3 L7 11 L2 26 L13 29 L13 57 L51 57 L51 29 L62 26 L57 11 L43 3 Q32 12 21 3 Z';
let shirtSeq = 0;
function shirtSvg(clubId, keeper) {
  let { a, b, pattern } = kitOf(clubId);
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
function readJson(key, fallback) {
  try { const v = JSON.parse(localStorage.getItem(key) || 'null'); return v && typeof v === 'object' ? v : fallback; } catch (e) { return fallback; }
}
/* ---------- negative binomial helper (same as the position pages) ---------- */
function nb(mu, r, per, steps) {
  const atLeast = {};
  if (mu <= 0) { steps.forEach(s => { atLeast[s] = 0; }); return { pts: 0, atLeast }; }
  const q = mu / (r + mu);
  let pmf = Math.pow(r / (r + mu), r), cdf = 0, pts = 0;
  for (let k = 0; k < 200; k++) {
    for (const s of steps) if (k === s) atLeast[s] = 1 - cdf;
    pts += pmf * Math.floor(k / per);
    cdf += pmf;
    if (cdf > 1 - 1e-10 && k > Math.max(...steps)) break;
    pmf *= (k + r) / (k + 1) * q;
  }
  for (const s of steps) if (atLeast[s] === undefined) atLeast[s] = 0;
  return { pts, atLeast };
}
function glm(m, vals, floor) {
  let z = m.intercept;
  m.features.forEach((name, i) => {
    const v = name === 'home' ? vals[name] : Math.log(Math.max(vals[name], floor));
    z += m.coef[i] * (v - m.mean[i]) / m.sd[i];
  });
  return Math.exp(z);
}

/* ---------- xP per fixture, per position (mirrors keepers.js / defenders.js / attackers.js) ---------- */
const DEF_STATS = { clr: { per: 4, steps: [4] }, blk: { per: 2, steps: [2] }, tkl: { per: 2, steps: [2] } };
function defenderFixtureXp(d, f) {
  const p = state.plans.DEF, sc = p.scoring, club = state.clubs.DEF[d.club], opp = state.clubs.DEF[f.oppId];
  let total = sc.appearance + sc.cleanSheet * f.cs + f.gcPts;
  for (const [stat, cfg] of Object.entries(DEF_STATS)) {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_opp: f.lamOpp, lam_own: f.lamOwn, home: f.home };
    total += nb(glm(p.model[stat], vals, 1e-3), p.model[stat].r, cfg.per, cfg.steps).pts;
  }
  const scale = f.lamOwn / p.lamAvg;
  total += sc.goal * d.rates.g90 * scale + sc.assist * d.rates.a90 * scale + sc.yellow * d.rates.y90 + sc.red * d.rates.r90 + p.other;
  return total; // if he plays 90; scaled by minutes / 90 like the Defender page
}
function curveAt(c, mins, key) {
  if (mins <= 0) return 0;
  const i = Math.min(c.mins.length - 2, Math.floor(mins / (c.mins[1] - c.mins[0])));
  const t = (mins - c.mins[i]) / (c.mins[i + 1] - c.mins[i]);
  return c[key][i] + t * (c[key][i + 1] - c[key][i]);
}
// v2 defenders (same as defenders.js fixtureXpV2): appearance from the minutes curve, clean sheet x P(60+ minutes), the
// rest x minutes / 90, goals / assists from the npxG / xA models
function defenderFixtureXpV2(d, f, mins) {
  const p = state.plans.DEF, sc = p.scoring, club = state.clubs.DEF[d.club], opp = state.clubs.DEF[f.oppId], t = mins / 90;
  if (mins <= 0) return 0;
  const val = stat => {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_opp: f.lamOpp, lam_own: f.lamOwn, home: f.home };
    Object.assign(vals, d.fm || {});
    if (d.mates) vals.mates = d.mates[stat];
    return glm(p.model[stat], vals, p.model[stat].floor || 1e-3);
  };
  let total = curveAt(p.appCurve, mins, 'pts') + sc.cleanSheet * f.cs * curveAt(p.p60Curve, mins, 'p') + f.gcPts * t;
  for (const [stat, cfg] of Object.entries(DEF_STATS)) total += nb(val(stat) * t, p.model[stat].r, cfg.per, cfg.steps).pts;
  const mg = p.model.goals, fm = d.fm || {};
  total += sc.goal * (val('goals') * mg.scale + (mg.pens ? fm.fm_pxg || 0 : 0)) * t + sc.assist * val('assists') * p.model.assists.scale * t;
  return total + t * (sc.yellow * d.rates.y90 + sc.red * d.rates.r90 + p.other);
}
// expected appearance points for expected minutes: the minutes model's curve where the plan has one (a 45 can be a 50%
// chance of starting), else the plain rule (2 for 60+, 1 below)
function appPoints(plan, mins) {
  const sc = plan.scoring, c = plan.appCurve;
  if (!c) return mins >= 60 ? sc.appearance60 : mins > 0 ? sc.appearance : 0;
  if (mins <= 0) return 0;
  const i = Math.min(c.mins.length - 2, Math.floor(mins / (c.mins[1] - c.mins[0])));
  const t = (mins - c.mins[i]) / (c.mins[i + 1] - c.mins[i]);
  return c.pts[i] + t * (c.pts[i + 1] - c.pts[i]);
}
function attackerFixtureXp(pos, d, f, mins) {
  const p = state.plans[pos], sc = p.scoring, club = state.clubs[pos][d.club], opp = state.clubs[pos][f.oppId], t = mins / 90;
  if (mins <= 0) return 0;
  const mu = {};
  for (const stat of Object.keys(p.model)) {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_own: f.lamOwn, lam_opp: f.lamOpp, home: f.home };
    Object.assign(vals, d.fm || {}); // FotMob history rates (midfielders): fm_* last 20, fm40_* last 40 appearances
    if (d.mates) vals.mates = d.mates[stat]; // team-mates' rate excluding him
    const m = p.model[stat];
    mu[stat] = (glm(m, vals, 1e-4) * (m.scale || 1) + (m.pens && d.fm ? d.fm.fm_pxg : 0)) * t;
  }
  const g = nb(mu.goals, p.model.goals.r, 1, [3]);
  const r = d.rates;
  return appPoints(p, mins) + sc.goal * mu.goals + sc.hatTrick * g.atLeast[3] + sc.assist * mu.assists
    + sc.sot * mu.sot + nb(mu.kp, p.model.kp.r, sc.keyPassPer, [2]).pts + (p.model.int ? sc.interception * mu.int : 0)
    + t * (sc.yellow * r.y90 + sc.red * r.r90 + sc.penMiss * r.pm90 + sc.ownGoal * r.og90);
}
// [{f, xp}] for a player's fixtures in a gameweek, at his expected minutes
function playerWeek(pl, gw) {
  const wk = state.week[pl.pos][pl.club] && state.week[pl.pos][pl.club][gw];
  if (!wk || !wk.games || pl.mins <= 0) return { wk, fx: (wk && wk.fx || []).map(f => ({ f, xp: 0 })) };
  const s = pl.mins / 90;
  let fx;
  if (pl.pos === 'GK') fx = wk.fx.map(f => ({ f, xp: s * f.xp }));
  else if (pl.pos === 'DEF') fx = wk.fx.map(f => ({ f, xp: state.plans.DEF.v2 ? defenderFixtureXpV2(pl.raw, f, pl.mins) : s * defenderFixtureXp(pl.raw, f) }));
  else fx = wk.fx.map(f => ({ f, xp: attackerFixtureXp(pl.pos, pl.raw, f, pl.mins) }));
  return { wk, fx };
}


async function loadXpData() {
  const files = { CLUB: 'club_plan.json', GK: 'keeper_plan.json', DEF: 'defender_plan.json', MID: 'mid_plan.json', FWD: 'fwd_plan.json' };
  const got = await Promise.all(Object.entries(files).map(async ([k, f]) => {
    const res = await fetch('data/' + f, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load ${f} (${res.status})`);
    return [k, await res.json()];
  }));
  state.plans = Object.fromEntries(got);
  for (const c of state.plans.CLUB.clubs) {
    state.shortByName[c.name] = c.short || c.name;
    if (c.abbr) state.abbrByName[c.name] = c.abbr;
    state.kitById[c.id] = c;
  }
  for (const k of ['CLUB', 'GK', 'DEF', 'MID', 'FWD']) {
    for (const c of state.plans[k].clubs) {
      state.week[k][c.id] = Object.fromEntries(c.weeks.map(w => [w.gw, w]));
      (state.clubs[k] = state.clubs[k] || {})[c.id] = c;
    }
  }
  const clubShort = id => { const c = state.plans.CLUB.clubs.find(q => q.id === id); return c ? (c.short || c.name) : '?'; };
  for (const pos of POSITIONS) {
    const plan = state.plans[pos];
    const minsEdits = readJson(MINS_KEYS[pos], {});
    for (const raw of (pos === 'GK' ? plan.keepers : pos === 'DEF' ? plan.defenders : plan.players)) {
      if (!state.week[pos][raw.club]) continue;
      const edited = minsEdits[raw.id] !== undefined;
      state.players.push({ id: raw.id, name: raw.name, pos, club: raw.club, clubShort: clubShort(raw.club), raw,
        mins: edited ? minsEdits[raw.id] : raw.xMins, minsEdited: edited && minsEdits[raw.id] !== raw.xMins });
    }
  }
}
