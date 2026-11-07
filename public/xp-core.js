// Shared by the Optimal Picks homepage (optimal.js) and the Player Picks "All" tab (allpicks.js): loads the position and
// club plans and computes each player's expected points with the same maths as the position pages (keepers.js /
// defenders.js / attackers.js), including start chances edited on those pages (saved in this browser; starts.js).
// Functions use the page's global `state` (plans, clubs, week, players, shortByName, abbrByName, kitById).
const POSITIONS = ['GK', 'DEF', 'MID', 'FWD'];
const POS_NAME = { GK: 'Goalkeeper', DEF: 'Defender', MID: 'Midfielder', FWD: 'Forward' };
const POS_PAGE = { GK: 'picks.html?pos=GK', DEF: 'picks.html?pos=DEF', MID: 'picks.html?pos=MID', FWD: 'picks.html?pos=FWD' };

const shortOf = name => state.shortByName[name] || name;
function abbr(name) {
  if (state.abbrByName[name]) return state.abbrByName[name];
  const w = shortOf(name).replace(/[^A-Za-z ]/g, '').split(' ').filter(Boolean);
  return (w.length === 1 ? w[0].slice(0, 3) : w[0].slice(0, 2) + w[1][0]).toUpperCase();
}

// kits: see kits.js (loaded first)
const kitOf = clubId => kitForClub(state.kitById[clubId]);
const shirtSvg = (clubId, keeper) => shirtSvgFor(state.kitById[clubId], keeper);

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
// v2 defenders (same as defenders.js fixtureXpV2): P(full game) x his usual full game (clean sheet counts) + P(part game)
// x a typical part game, for a start chance ps (starts.js)
function defenderFixtureXpV2(d, f, ps) {
  const p = state.plans.DEF, sc = scenarioProbs(p, d, ps);
  return sc.pFull * defenderAt(d, f, d.mFull || p.startModel.full, 2, 1) + sc.pPart * defenderAt(d, f, p.startModel.part, 1, 0);
}
// one scenario: plays `mins` with `app` appearance points; the clean sheet counts with weight csW
function defenderAt(d, f, mins, app, csW) {
  const p = state.plans.DEF, sc = p.scoring, club = state.clubs.DEF[d.club], opp = state.clubs.DEF[f.oppId], t = mins / 90;
  const val = stat => {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_opp: f.lamOpp, lam_own: f.lamOwn, home: f.home };
    Object.assign(vals, d.fm || {});
    if (d.mates) vals.mates = d.mates[stat];
    return glm(p.model[stat], vals, p.model[stat].floor || 1e-3);
  };
  let total = app + sc.cleanSheet * f.cs * csW + f.gcPts * t;
  for (const [stat, cfg] of Object.entries(DEF_STATS)) total += nb(val(stat) * t, p.model[stat].r, cfg.per, cfg.steps).pts;
  const mg = p.model.goals, fm = d.fm || {};
  total += sc.goal * (val('goals') * mg.scale + (mg.pens ? fm[mg.penCol || 'fm_pxg'] || 0 : 0)) * t + sc.assist * val('assists') * p.model.assists.scale * t;
  return total + t * (sc.yellow * d.rates.y90 + sc.red * d.rates.r90 + p.other);
}
function attackerFixtureXp(pos, d, f, ps) {
  const p = state.plans[pos], sc = scenarioProbs(p, d, ps);
  return sc.pFull * attackerAt(pos, d, f, d.mFull || p.startModel.full, p.scoring.appearance60)
    + sc.pPart * attackerAt(pos, d, f, p.startModel.part, p.scoring.appearance);
}
function attackerAt(pos, d, f, mins, app) {
  const p = state.plans[pos], sc = p.scoring, club = state.clubs[pos][d.club], opp = state.clubs[pos][f.oppId], t = mins / 90;
  const mu = {};
  for (const stat of Object.keys(p.model)) {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_own: f.lamOwn, lam_opp: f.lamOpp, home: f.home };
    Object.assign(vals, d.fm || {}); // FotMob history rates (midfielders): fm_* last 20, fm40_* last 40 appearances
    if (d.mates) vals.mates = d.mates[stat]; // team-mates' rate excluding him
    const m = p.model[stat];
    mu[stat] = (glm(m, vals, 1e-4) * (m.scale || 1) + (m.pens && d.fm ? d.fm[m.penCol || 'fm_pxg'] || 0 : 0)) * t;
  }
  const g = nb(mu.goals, p.model.goals.r, 1, [3]);
  const r = d.rates;
  return app + sc.goal * mu.goals + sc.hatTrick * g.atLeast[3] + sc.assist * mu.assists
    + sc.sot * mu.sot + nb(mu.kp, p.model.kp.r, sc.keyPassPer, [2]).pts + (p.model.int ? sc.interception * mu.int : 0)
    + t * (sc.yellow * r.y90 + sc.red * r.r90 + sc.penMiss * r.pm90 + sc.ownGoal * r.og90);
}
// [{f, xp}] for a player's fixtures in a gameweek at his start chance (pl.start); asStarter: "if he starts" (his club's
// first game of the gameweek at 100%, a double's second game at his chance of starting it given he started the first)
function playerWeek(pl, gw, asStarter) {
  const wk = state.week[pl.pos][pl.club] && state.week[pl.pos][pl.club][gw];
  if (!wk || !wk.games || (pl.start <= 0 && !asStarter)) return { wk, fx: (wk && wk.fx || []).map(f => ({ f, xp: 0 })) };
  const fx = wk.fx.map((f, j) => {
    const ps = startChanceAt(pl.raw, j, pl.start, asStarter), plan = state.plans[pl.pos];
    if (pl.pos === 'GK') {
      const sc = scenarioProbs(plan, pl.raw, ps);
      return { f, xp: sc.pFull * f.xp + sc.pPart };
    }
    if (pl.pos === 'DEF') return { f, xp: defenderFixtureXpV2(pl.raw, f, ps) };
    return { f, xp: attackerFixtureXp(pl.pos, pl.raw, f, ps) };
  });
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
    const edits = readStore(START_KEYS[pos]);
    for (const raw of (pos === 'GK' ? plan.keepers : pos === 'DEF' ? plan.defenders : plan.players)) {
      if (!state.week[pos][raw.club]) continue;
      const edited = edits[raw.id] !== undefined;
      state.players.push({ id: raw.id, name: raw.name, pos, club: raw.club, clubShort: clubShort(raw.club), raw,
        start: edited ? edits[raw.id] : raw.pStart || 0, startEdited: edited && Math.abs(edits[raw.id] - (raw.pStart || 0)) > 1e-9 });
    }
  }
}
