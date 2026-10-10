// Midfielder / forward picks: one script for both pages (<body data-pos="MID|FWD">)
const statusEl = document.getElementById('status');
const root = document.getElementById('kp-root');
const POS = document.body.dataset.pos;
const LABEL = POS === 'MID' ? { one: 'Midfielder', many: 'Midfielders', file: 'mid_plan.json' } : { one: 'Forward', many: 'Forwards', file: 'fwd_plan.json' };
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];
const MINS_KEY = `efl_${POS.toLowerCase()}_mins_v1`;
const STATS = { goals: 'Goals', assists: 'Assists', sot: 'Shots on target', kp: 'Key passes', int: 'Interceptions' };

const state = { plan: null, clubs: {}, clubWeek: {}, shortByName: {}, gw: null, league: 'All', query: '', showAll: false, mins: {}, cache: {} };

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmt = n => n.toFixed(1);
const fmt2 = n => n.toFixed(2);
const pct = p => (p * 100).toFixed(0) + '%';
function fmtDate(iso, withDay) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-GB', withDay ? { weekday: 'short', day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short' });
}
const weekOf = (clubId, gw) => state.clubWeek[clubId] && state.clubWeek[clubId][gw];
// a club is locked for a gameweek once one of its games that week has kicked off (Fantasy EFL locks game by game)
const kickedOff = wk => !!(wk && wk.fx && wk.fx.some(f => f.ko && new Date(f.ko).getTime() <= Date.now()));
const gwMeta = gw => state.plan.gameweeks.find(g => g.gw === gw);
const shortOf = name => state.shortByName[name] || name;

/* expected minutes: 90 if he played 60+ in his club's latest game, else 0; edits saved in this browser */
function loadMins() { try { state.mins = JSON.parse(localStorage.getItem(MINS_KEY) || '{}') || {}; } catch (e) { state.mins = {}; } }
function saveMins() { try { localStorage.setItem(MINS_KEY, JSON.stringify(state.mins)); } catch (e) { /* ignore */ } }
const minsOf = d => (state.mins[d.id] !== undefined ? state.mins[d.id] : d.xMins);
const edited = d => state.mins[d.id] !== undefined && state.mins[d.id] !== d.xMins;

/* ---------- the model (same maths as ml/team_strengths/attackers.py / attack_model.py) ---------- */
function ratePer90(stat, d, club, f) {
  const m = state.plan.model[stat];
  const opp = state.clubs[f.oppId];
  const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_own: f.lamOwn, lam_opp: f.lamOpp, home: f.home };
  Object.assign(vals, d.fm || {}); // FotMob history rates (midfielders): fm_* last 20, fm40_* last 40 appearances
  if (d.mates) vals.mates = d.mates[stat]; // team-mates' rate excluding him
  let z = m.intercept;
  m.features.forEach((name, i) => {
    const v = name === 'home' ? vals[name] : Math.log(Math.max(vals[name], 1e-4));
    z += m.coef[i] * (v - m.mean[i]) / m.sd[i];
  });
  return Math.exp(z);
}
// negative binomial with mean mu and size r: E[floor(X / per)] and P(X >= k) for each k in steps
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
function curveAt(c, mins, key) {
  if (!c || mins <= 0) return 0;
  const i = Math.min(c.mins.length - 2, Math.floor(mins / (c.mins[1] - c.mins[0])));
  const t = (mins - c.mins[i]) / (c.mins[i + 1] - c.mins[i]);
  return c[key][i] + t * (c[key][i + 1] - c[key][i]);
}
// expected points for one fixture if he plays exactly `mins` minutes with `app` appearance points
function compsAt(d, f, mins, app) {
  const p = state.plan, sc = p.scoring, club = state.clubs[d.club], t = mins / 90;
  const out = { mins, app, mu: {}, rate: {} };
  // goals / assists come from predicted npxG / xA x conversion (+ his penalty term for goals)
  for (const stat of Object.keys(p.model)) {
    const m = p.model[stat];
    out.rate[stat] = ratePer90(stat, d, club, f) * (m.scale || 1) + (m.pens && d.fm ? d.fm[m.penCol || 'fm_pxg'] || 0 : 0);
    out.mu[stat] = out.rate[stat] * t;
  }
  const g = nb(out.mu.goals, p.model.goals.r, 1, [1, 2, 3]);
  const kp = nb(out.mu.kp, p.model.kp.r, sc.keyPassPer, [2, 4]);
  out.pGoal = g.atLeast[1]; out.pHat = g.atLeast[3]; out.kpSteps = kp.atLeast;
  out.goalPts = sc.goal * out.mu.goals + sc.hatTrick * out.pHat;
  out.assistPts = sc.assist * out.mu.assists;
  out.sotPts = sc.sot * out.mu.sot;
  out.kpPts = kp.pts;
  out.intPts = p.model.int ? sc.interception * out.mu.int : 0;
  const r = d.rates;
  out.cardPts = t * (sc.yellow * r.y90 + sc.red * r.r90 + sc.penMiss * r.pm90 + sc.ownGoal * r.og90);
  out.xp = out.app + out.goalPts + out.assistPts + out.sotPts + out.kpPts + out.intPts + out.cardPts;
  return out;
}
// chance of each minutes scenario for an expected-minutes value: P(60+) from the minutes model's curve, P(1-59) from the
// expected appearance points (= 2 P(60+) + 1 P(1-59)); the rest is not playing
function scenarios(mins) {
  const p = state.plan;
  const p60 = curveAt(p.p60Curve, mins, 'p');
  return { p60, part: Math.max(0, appPoints(p, mins) - 2 * p60) };
}
// expected points for one fixture with `mins` expected minutes. With the minutes-scenario mixture (minsMix in the plan):
// P(60+) x points if he plays his usual full game + P(1-59) x points from a typical part game. Thresholds (key passes
// per 2, hat-tricks) are then evaluated on the minutes he would actually play, not on the average.
function fixtureXp(d, f, mins) {
  const key = `${d.id}|${f.oppId}|${f.date}|${mins}`;
  if (state.cache[key]) return state.cache[key];
  const p = state.plan;
  let out;
  if (p.minsMix && p.p60Curve && mins > 0) {
    const s = scenarios(mins);
    const full = compsAt(d, f, d.mFull || p.minsMix.full, p.scoring.appearance60);
    const part = compsAt(d, f, p.minsMix.part, p.scoring.appearance);
    const mix = (a, b) => s.p60 * a + s.part * b;
    out = { mins, p60: s.p60, pPart: s.part, rate: full.rate, mu: {}, kpSteps: {} };
    for (const k of ['app', 'goalPts', 'assistPts', 'sotPts', 'kpPts', 'intPts', 'cardPts', 'xp', 'pGoal', 'pHat']) out[k] = mix(full[k], part[k]);
    for (const k of Object.keys(full.mu)) out.mu[k] = mix(full.mu[k], part.mu[k]);
    for (const k of Object.keys(full.kpSteps)) out.kpSteps[k] = mix(full.kpSteps[k], part.kpSteps[k]);
  } else {
    out = compsAt(d, f, mins, appPoints(p, mins));
  }
  state.cache[key] = out;
  return out;
}
function weekXp(d, gw, mins) {
  const w = weekOf(d.club, gw);
  if (!w || !w.games) return { xp: 0, fx: [] };
  const fx = w.fx.map(f => ({ f, r: fixtureXp(d, f, mins) }));
  return { xp: fx.reduce((a, x) => a + x.r.xp, 0), fx };
}
const pxp = (d, gw) => weekXp(d, gw, minsOf(d)).xp;

/* ---------- tooltip ---------- */
const tip = document.createElement('div');
tip.className = 'cp-tip';
tip.style.display = 'none';
document.body.appendChild(tip);

function breakdownRows(items) {
  const sc = state.plan.scoring;
  const sum = fn => items.reduce((a, x) => a + fn(x), 0);
  const one = items.length === 1 ? items[0].r : null;
  const rows = [
    ['Appearance', state.plan.appCurve ? `expected, from ${items[0].r.mins} expected minutes` : one ? (one.mins >= 60 ? '60+ minutes' : one.mins > 0 ? 'under 60 minutes' : 'does not play') : `per game, ${items.length} games`, sum(x => x.r.app)],
    ['Goals', `${sc.goal} × ${sum(x => x.r.mu.goals).toFixed(2)} exp.${one ? ` · P(scores) ${pct(one.pGoal)}` : ''} + hat-trick ${sc.hatTrick} × P(3+)`, sum(x => x.r.goalPts)],
    ['Assists', `${sc.assist} × ${sum(x => x.r.mu.assists).toFixed(2)} exp.`, sum(x => x.r.assistPts)],
    ['Shots on target', `${sc.sot} × ${sum(x => x.r.mu.sot).toFixed(2)} exp.`, sum(x => x.r.sotPts)],
    ['Key passes', one ? `[P(2+) ${pct(one.kpSteps[2])} + P(4+) ${pct(one.kpSteps[4])} + …] · exp. ${one.mu.kp.toFixed(2)}` : `+1 per 2 · exp. ${sum(x => x.r.mu.kp).toFixed(2)}`, sum(x => x.r.kpPts)],
  ];
  if (state.plan.model.int) rows.push(['Interceptions', `${sc.interception} × ${sum(x => x.r.mu.int).toFixed(2)} exp.`, sum(x => x.r.intPts)]);
  rows.push(['Cards and other', 'his card / missed pen / own goal rates', sum(x => x.r.cardPts)]);
  return rows;
}
function showTip(el, x, y) {
  const d = state.plan.players.find(q => q.id === +el.dataset.player);
  if (!d) return;
  const mins = minsOf(d);
  const wk = weekXp(d, +el.dataset.gw, mins);
  if (!wk.fx.length) return;
  const items = el.dataset.tip === 'fx' ? [wk.fx[+el.dataset.i]] : wk.fx;
  const club = state.clubs[d.club];
  const rows = breakdownRows(items);
  const total = items.reduce((a, it) => a + it.r.xp, 0);
  const title = items.length === 1 ? `${club.short} v ${shortOf(items[0].f.opp)} (${items[0].f.ha}) · ${fmtDate(items[0].f.date, true)}` : `${d.name} · GW ${el.dataset.gw}`;
  let foot = '';
  if (items.length === 1) {
    const f = items[0].f, opp = state.clubs[f.oppId];
    foot = `<div class="cp-tip-foot">Odds: ${esc(club.short)} expected goals ${f.lamOwn.toFixed(2)}, ${esc(opp ? opp.short : '?')} ${f.lamOpp.toFixed(2)}.
      His role vs ${POS === 'MID' ? 'midfield' : 'forward'} team-mates: goals ${d.role.goals.toFixed(2)}×, shots on target ${d.role.sot.toFixed(2)}×, key passes ${d.role.kp.toFixed(2)}×${d.role.int ? `, interceptions ${d.role.int.toFixed(2)}×` : ''}.</div>`;
  }
  tip.innerHTML = `<div class="cp-tip-title">${esc(title)} <span>${esc(d.name)}: expected points with ${mins} expected minutes</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">Total</td><td class="cp-tip-num">${fmt2(total)}</td></tr></table>${foot}`;
  tip.style.display = 'block';
  moveTip(x, y);
}
function moveTip(x, y) {
  if (window.innerWidth > 0) tip.style.maxWidth = Math.min(520, window.innerWidth - 16) + 'px';
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let left = x + pad, top = y + pad;
  if (left + w > window.innerWidth - 8) left = Math.max(8, x - w - pad);
  if (top + h > window.innerHeight - 8) top = Math.max(8, y - h - pad);
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}
document.addEventListener('mouseover', e => {
  const el = e.target.closest && e.target.closest('[data-tip]');
  if (el) showTip(el, e.clientX, e.clientY); else tip.style.display = 'none';
});
document.addEventListener('mousemove', e => { if (tip.style.display === 'block') moveTip(e.clientX, e.clientY); });
document.addEventListener('scroll', () => { tip.style.display = 'none'; }, true);

/* ---------- page ---------- */
function tags(d) {
  const t = [];
  if (d.injury || d.status === 'injured') t.push(`<span class="cp-tag kp-tag-out">${esc(d.injury || 'injured')}</span>`);
  if (d.suspended) t.push('<span class="cp-tag kp-tag-out">suspended</span>');
  if (d.startedLast) t.push('<span class="cp-tag cp-tag-picked" title="Played 60+ minutes in his club\'s latest game">started last</span>');
  if (kickedOff(weekOf(d.club, state.gw))) t.push('<span class="cp-tag" title="His club\'s game this gameweek has kicked off, so he is locked">locked</span>');
  return t.join(' ');
}
function render() {
  const { plan } = state;
  const scrollY = window.scrollY;
  const gw = state.gw;
  const meta = gwMeta(gw);
  const q = state.query.trim().toLowerCase();
  const hasInt = !!plan.model.int;
  const next5 = plan.gameweeks.filter(g => g.gw >= gw).slice(0, 5).map(g => g.gw);
  const list = plan.players.filter(d => {
    const c = state.clubs[d.club];
    return c && (state.showAll || minsOf(d) >= (state.plan.appCurve ? 30 : 1)) && (state.league === 'All' || c.league === state.league)
      && (!q || d.name.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q));
  }).map(d => {
    const wk = weekXp(d, gw, minsOf(d));
    return { d, c: state.clubs[d.club], wk, xp: wk.xp, n5: next5.reduce((a, g) => a + pxp(d, g), 0) };
  }).sort((a, b) => b.xp - a.xp);
  const shown = list.slice(0, 150);
  const options = plan.gameweeks.map(g => `<option value="${g.gw}" ${g.gw === gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  const stale = plan.latestCompletedGw > plan.startersFromGw
    ? `<div class="cp-warn">⚠ Expected starters use lineups up to GW ${plan.startersFromGw}, but GW ${plan.latestCompletedGw} has finished. Refresh after exporting the latest stats.</div>` : '';
  const rows = shown.map(({ d, c, wk, xp, n5 }, i) => {
    const sumMu = s => wk.fx.reduce((a, x) => a + x.r.mu[s], 0);
    const cell = s => `<td class="cp-center opt">${wk.fx.length ? sumMu(s).toFixed(2) : '–'}</td>`;
    return `<tr class="${minsOf(d) <= 0 ? 'cp-dim' : ''}">
      <td><span class="cp-rank ${i < 3 ? 'cp-rank-' + (i + 1) : 'cp-rank-n'}">${i + 1}</span></td>
      <td><strong>${esc(d.name)}</strong> ${tags(d)}<div class="cp-muted cp-small">${d.starts60} games of 60+ this season · ${d.totalPoints} pts</div></td>
      <td class="pp-club">${shirtIcon(c, false)}${spList(d.tags)}<div>${esc(c.short)}<div class="cp-muted cp-small">${esc(c.league)}</div></div></td>
      <td class="cp-center opt2">${recentHtml(d.recent)}</td>
      <td class="cp-center"><input type="number" class="kp-mins ${edited(d) ? 'kp-mins-edited' : ''}" data-player="${d.id}" min="0" max="90" step="5" value="${minsOf(d)}" title="Expected minutes per game. Default ${d.xMins}." />${state.plan.p60Curve && minsOf(d) > 0 ? `<div class="cp-muted cp-small" title="Chance he plays 60+ minutes (the minutes model); the expected minutes are an average that includes the chance he misses out or comes off early">60+: ${pct(curveAt(state.plan.p60Curve, minsOf(d), 'p'))}</div>` : ''}</td>
      <td>${wk.fx.length ? wk.fx.map((x, j) => `<span class="cp-fx" data-tip="fx" data-player="${d.id}" data-gw="${gw}" data-i="${j}"><span class="cp-dot ${x.f.src === 'market' ? 'cp-dot-market' : 'cp-dot-model'}"></span>${fxName(x.f.opp, esc(shortOf(x.f.opp)))} (${x.f.ha}) <span class="cp-fx-xp">${fmt(x.r.xp)}</span></span>`).join('') : '<span class="cp-muted">No fixture</span>'}</td>
      <td class="cp-center opt">${wk.fx.length ? wk.fx.map(x => pct(x.r.pGoal)).join(' / ') : '–'}</td>
      ${cell('sot')}${cell('kp')}${hasInt ? cell('int') : ''}
      <td class="cp-xp" data-tip="total" data-player="${d.id}" data-gw="${gw}">${fmt(xp)}</td>
      <td class="cp-right cp-muted opt">${fmt(n5)}</td></tr>`;
  }).join('');
  const t = s => plan.model[s].test;
  const err = s => !t(s).model ? '' : `${t(s).model.mse.toFixed(3)} vs ${t(s).own_rate.mse.toFixed(3)} (own rate) vs ${t(s).last10.mse.toFixed(3)} (last 10)`;
  const scoring = POS === 'MID'
    ? 'appearance, goals (+6), assists (+3), shots on target (+1 each), key passes (+1 per 2), interceptions (+2 each), hat-tricks (+5) and cards'
    : 'appearance, goals (+5), assists (+3), shots on target (+1 each), key passes (+1 per 2), hat-tricks (+5) and cards';
  root.innerHTML = `
    <p class="cp-intro">Expected points (xP) for ${LABEL.many.toLowerCase()}, built from the parts that score: ${scoring}.
      Each count comes from a model of the player's role in his team, his club's style, the opponent's style and the match odds.
      Edit expected minutes to change a prediction. Hover a fixture or an xP for the breakdown.</p>
    <div class="cp-controls">
      <button id="kp-prev" class="cp-nav-btn">&#8592;</button><select id="kp-gw">${options}</select><button id="kp-next" class="cp-nav-btn">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="kp-search" type="text" placeholder="Search ${LABEL.one.toLowerCase()} or club..." value="${esc(state.query)}" autocomplete="off" />
      <label class="kp-toggle"><input type="checkbox" id="kp-all" ${state.showAll ? 'checked' : ''}/> ${state.plan.appCurve ? 'Show players under 30 expected minutes' : 'Show non-starters and injured'}</label>
      ${Object.keys(state.mins).length ? '<button class="cp-link-btn" id="kp-reset">Reset minutes</button>' : ''}
    </div>
    ${stale}
    <h2 class="cp-h2">${LABEL.many} for GW ${gw} <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.marketGames} of ${meta.games} games priced from odds · ${list.length} shown${list.length > shown.length ? ' (top 150 listed)' : ''}</span></h2>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th>#</th><th>${LABEL.one}</th><th>Club</th><th class="cp-center opt2" title="Minutes in his last 5 games for this club, oldest first (dark = 60+, light = came on or off, grey = did not play)">Last 5 (mins)</th><th class="cp-center" title="Editable. Minutes he plays in each game; counts scale with minutes, 60+ earns the 2-point appearance. Saved in this browser.">Exp. mins</th>
        <th>Fixtures (xP)</th><th class="cp-center opt">P(scores)</th><th class="cp-center opt">Exp. shots on target</th><th class="cp-center opt">Exp. key passes</th>${hasInt ? '<th class="cp-center opt">Exp. interceptions</th>' : ''}
        <th class="cp-right">xP</th><th class="cp-right opt" title="Expected points over this and the next 4 gameweeks">Next 5 GWs</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="${hasInt ? 12 : 11}" class="cp-muted">No ${LABEL.many.toLowerCase()} match.</td></tr>`}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      ${plan.appCurve
        ? `<li><strong>Expected minutes</strong> come from our minutes model: his recent minutes, starts and appearances at his club, how long since he last played,
        whether he is new to the club, rest days and doubles (0 if injured or suspended). It is an average, so 45 can mean a 50% chance of starting.
        Goals, shots and passes scale with expected minutes; appearance points follow the model's curve (e.g. 90 → 2.0, 60 → about 1.4, 30 → about 0.9).
        Edit the box to override; edits are saved in this browser.</li>`
        : `<li><strong>Expected minutes</strong> default to 90 for players who played 60+ minutes in their club's latest game, else 0 (and 0 if injured or suspended).
        Edit the box for the minutes he will play in each game: goals, shots and passes scale with minutes, and 60+ earns 2 appearance points (1 below 60). Edits are saved in this browser.</li>`}
      <li><strong>Each count</strong> = his role (his rate compared with ${POS === 'MID' ? 'midfield' : 'forward'} team-mates in the same games, so it moves with him between clubs)
        × his club's recent style × what this week's opponent concedes × the match odds (his team's and the opponent's expected goals) and home advantage.
        Rolling club and opponent styles use recent gameweeks; everything is shrunk towards the league average.</li>
      ${t('goals').v2
        ? `<li><strong>Goals and assists come from expected goals (xG) and expected assists (xA)</strong>: the model predicts his non-penalty xG and xA
        per 90 (from FotMob data: his chances relative to team-mates, his club's and the opponent's style, the odds), which are much less noisy than goals and
        assists, then converts them (goals ≈ ${state.plan.model.goals.scale.toFixed(2)} × npxG + his penalty xG; assists ≈ ${state.plan.model.assists.scale.toFixed(2)} × xA).
        Shots and key passes also use his FotMob shots, chances created, xG and xA history.</li>
        <li><strong>Moving clubs or leagues</strong>: numbers from another league are adjusted by how much players' output changes on stepping up or down
        (measured on FotMob data, e.g. xG from League One counts about 0.8× in the Championship), and a club's style carries over
        from last season, adjusted for promotion or relegation.</li>
        <li>Tested on this season's games without having seen them (mean squared error, new model vs previous): goals ${t('goals').v2.mse.toFixed(4)} vs ${t('goals').v1.mse.toFixed(4)};
        assists ${t('assists').v2.mse.toFixed(4)} vs ${t('assists').v1.mse.toFixed(4)}; shots on target ${t('sot').v2.mse.toFixed(3)} vs ${t('sot').v1.mse.toFixed(3)};
        key passes ${t('kp').v2.mse.toFixed(3)} vs ${t('kp').v1.mse.toFixed(3)}${state.plan.model.int ? `; interceptions ${t('int').v2.mse.toFixed(3)} vs ${t('int').v1.mse.toFixed(3)}` : ''}. Single games are still mostly luck.</li>`
        : `<li>Tested on this season's games without having seen them (mean squared error, lower is better): goals ${err('goals')}; shots on target ${err('sot')};
        key passes ${err('kp')}. Mostly level with the player's own rate, better than a plain recent average, and the odds help for goals and shots.
        Whole-game xP beats each player's recent average points (error ${POS === 'MID' ? '2.53 vs 2.64' : '2.65 vs 2.72'}), but single games are mostly luck.</li>
      <li>No expected-goals (xG) data used for ${LABEL.many.toLowerCase()} yet; midfielders already use it, forwards are next.</li>`}
    </ul></div>
    <p class="cp-foot">Updated ${esc(new Date(plan.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }))}.
      <span class="cp-legend"><span class="cp-dot cp-dot-market"></span> priced from odds <span class="cp-dot cp-dot-model"></span> model estimate</span></p>`;
  tip.style.display = 'none';
  bind();
  window.scrollTo(0, scrollY);
}

function bind() {
  document.getElementById('kp-gw').addEventListener('change', e => { state.gw = +e.target.value; render(); });
  const step = d => {
    const i = state.plan.gameweeks.findIndex(g => g.gw === state.gw) + d;
    if (i >= 0 && i < state.plan.gameweeks.length) { state.gw = state.plan.gameweeks[i].gw; render(); }
  };
  document.getElementById('kp-prev').addEventListener('click', () => step(-1));
  document.getElementById('kp-next').addEventListener('click', () => step(1));
  document.querySelectorAll('.cp-chip[data-league]').forEach(b => b.addEventListener('click', () => { state.league = b.dataset.league; render(); }));
  document.getElementById('kp-all').addEventListener('change', e => { state.showAll = e.target.checked; render(); });
  document.querySelectorAll('.kp-mins').forEach(inp => inp.addEventListener('change', () => {
    const v = Math.max(0, Math.min(90, Math.round(+inp.value || 0)));
    const d = state.plan.players.find(q => q.id === +inp.dataset.player);
    if (v === d.xMins) delete state.mins[d.id]; else state.mins[d.id] = v;
    saveMins(); render();
  }));
  const reset = document.getElementById('kp-reset');
  if (reset) reset.addEventListener('click', () => { state.mins = {}; saveMins(); render(); });
  const s = document.getElementById('kp-search');
  s.addEventListener('input', e => {
    state.query = e.target.value;
    const pos = e.target.selectionStart;
    render();
    const n = document.getElementById('kp-search');
    n.focus();
    n.setSelectionRange(pos, pos);
  });
}

async function load() {
  statusEl.textContent = 'Loading...';
  try {
    const res = await fetch('data/' + LABEL.file, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load ${LABEL.one.toLowerCase()} data (${res.status})`);
    state.plan = await res.json();
    for (const c of state.plan.clubs) {
      state.clubs[c.id] = c;
      state.shortByName[c.name] = c.short;
      if (c.abbr) ABBR[c.name] = c.abbr;
      state.clubWeek[c.id] = Object.fromEntries(c.weeks.map(w => [w.gw, w]));
    }
    loadMins();
    state.gw = state.plan.firstGw;
    statusEl.textContent = '';
    render();
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

load();
