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
function curveAt(c, mins, key) {
  if (mins <= 0) return 0;
  const i = Math.min(c.mins.length - 2, Math.floor(mins / (c.mins[1] - c.mins[0])));
  const t = (mins - c.mins[i]) / (c.mins[i + 1] - c.mins[i]);
  return c[key][i] + t * (c[key][i + 1] - c[key][i]);
}
// v2 defenders (same as defenders.js fixtureXpV2): appearance from the minutes curve, clean sheet x P(60+ minutes), the
// rest x minutes / 90, goals / assists from the npxG / xA models
function defenderFixtureXpV2(d, f, mins) {
  const p = state.plans.DEF;
  if (mins <= 0) return 0;
  if (p.minsMix) {
    const sc = scenarios(p, mins);
    return sc.p60 * defenderAt(d, f, d.mFull || p.minsMix.full, 2, 1) + sc.part * defenderAt(d, f, p.minsMix.part, 1, 0);
  }
  return defenderAt(d, f, mins, curveAt(p.appCurve, mins, 'pts'), curveAt(p.p60Curve, mins, 'p'));
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
// minutes scenarios for an expected-minutes value (same as the position pages): P(60+), P(1-59)
function scenarios(p, mins) {
  const p60 = curveAt(p.p60Curve, mins, 'p');
  return { p60, part: Math.max(0, curveAt(p.appCurve, mins, 'pts') - 2 * p60) };
}
function attackerFixtureXp(pos, d, f, mins) {
  const p = state.plans[pos];
  if (mins <= 0) return 0;
  if (p.minsMix && p.p60Curve) {
    const sc = scenarios(p, mins);
    return sc.p60 * attackerAt(pos, d, f, d.mFull || p.minsMix.full, p.scoring.appearance60)
      + sc.part * attackerAt(pos, d, f, p.minsMix.part, p.scoring.appearance);
  }
  return attackerAt(pos, d, f, mins, appPoints(p, mins));
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
// [{f, xp}] for a player's fixtures in a gameweek, at his expected minutes
function playerWeek(pl, gw) {
  const wk = state.week[pl.pos][pl.club] && state.week[pl.pos][pl.club][gw];
  if (!wk || !wk.games || pl.mins <= 0) return { wk, fx: (wk && wk.fx || []).map(f => ({ f, xp: 0 })) };
  const s = pl.mins / 90;
  let fx;
  const gk = state.plans.GK;
  if (pl.pos === 'GK') fx = wk.fx.map(f => {
    if (!gk.minsMix || !gk.p60Curve) return { f, xp: s * f.xp };
    const sc = scenarios(gk, pl.mins);
    return { f, xp: sc.p60 * f.xp + sc.part * 1 };
  });
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
