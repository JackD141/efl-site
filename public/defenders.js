const statusEl = document.getElementById('status');
const root = document.getElementById('kp-root');
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];
const MINS_KEY = 'efl_defender_mins_v1';
const STATS = { clr: { name: 'Clearances', per: 4, steps: [4, 8, 12] }, blk: { name: 'Blocks', per: 2, steps: [2, 4] }, tkl: { name: 'Tackles', per: 2, steps: [2, 4] } };

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

/* ---------- the model (same maths as ml/team_strengths/defenders.py) ---------- */
function expectedCount(stat, d, club, f) {
  const m = state.plan.model[stat];
  const opp = state.clubs[f.oppId];
  const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_opp: f.lamOpp, lam_own: f.lamOwn, home: f.home };
  Object.assign(vals, d.fm || {}); // FotMob history rates (v2)
  if (d.mates) vals.mates = d.mates[stat]; // team-mates' rate excluding him (v2)
  let z = m.intercept;
  m.features.forEach((name, i) => {
    const v = name === 'home' ? vals[name] : Math.log(Math.max(vals[name], m.floor || 1e-3));
    z += m.coef[i] * (v - m.mean[i]) / m.sd[i];
  });
  return Math.exp(z);
}
// negative binomial: probabilities of reaching each step, and expected points floor(X / per)
function nbSteps(mu, r, per, steps) {
  const q = mu / (r + mu);
  let pmf = Math.pow(r / (r + mu), r), cdf = 0, pts = 0;
  const atLeast = {};
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
// v2: expected appearance points and chance of 60+ minutes from the minutes model's curves
function curveAt(c, mins, key) {
  if (mins <= 0) return 0;
  const i = Math.min(c.mins.length - 2, Math.floor(mins / (c.mins[1] - c.mins[0])));
  const t = (mins - c.mins[i]) / (c.mins[i + 1] - c.mins[i]);
  return c[key][i] + t * (c[key][i + 1] - c[key][i]);
}
// v2 expected points for one fixture with `mins` expected minutes: appearance from the minutes curve, clean sheet x
// P(60+ minutes), everything else scales with minutes / 90; goals / assists from the npxG / xA models
// one minutes scenario: he plays exactly `mins` minutes with `app` appearance points; the clean sheet only counts in the
// 60+ scenario (csOn); goals conceded, counts, goals / assists and cards scale with minutes
function defCompsAt(d, f, mins, app, csOn) {
  const p = state.plan, sc = p.scoring, club = state.clubs[d.club], t = mins / 90;
  const out = { mins, app, cs: f.cs, csPts: csOn ? sc.cleanSheet * f.cs : 0, gcPts: f.gcPts * t, stats: {} };
  let total = out.app + out.csPts + out.gcPts;
  for (const [stat, cfg] of Object.entries(STATS)) {
    const mu = expectedCount(stat, d, club, f) * t;
    const st = nbSteps(mu, p.model[stat].r, cfg.per, cfg.steps);
    out.stats[stat] = { mu, pts: st.pts, atLeast: st.atLeast };
    total += st.pts;
  }
  const mg = p.model.goals, ma = p.model.assists, fm = d.fm || {};
  out.goals = (expectedCount('goals', d, club, f) * mg.scale + (mg.pens ? fm[mg.penCol || 'fm_pxg'] || 0 : 0)) * t;
  out.assists = expectedCount('assists', d, club, f) * ma.scale * t;
  out.attPts = sc.goal * out.goals + sc.assist * out.assists;
  out.cardPts = t * (sc.yellow * d.rates.y90 + sc.red * d.rates.r90 + p.other);
  out.xp = total + out.attPts + out.cardPts;
  return out;
}
// v2 expected points for one fixture with `mins` expected minutes: P(60+) x his usual full game (clean sheet counts) +
// P(1-59) x a typical part game, so the clearance / tackle / block steps are evaluated on minutes he would really play.
// Without minsMix in the plan: the earlier version (appearance curve, clean sheet x P(60+), the rest x minutes / 90)
function fixtureXpV2(d, f, mins) {
  const key = `${d.id}|${f.oppId}|${f.date}|${mins}`;
  if (state.cache[key]) return state.cache[key];
  const p = state.plan;
  const p60 = curveAt(p.p60Curve, mins, 'p');
  let out;
  if (p.minsMix && mins > 0) {
    const part = Math.max(0, curveAt(p.appCurve, mins, 'pts') - 2 * p60);
    const a = defCompsAt(d, f, d.mFull || p.minsMix.full, 2, true), b = defCompsAt(d, f, p.minsMix.part, 1, false);
    const mix = (x, y) => p60 * x + part * y;
    out = { mins, p60, pPart: part, cs: f.cs, stats: {} };
    for (const k of ['app', 'csPts', 'gcPts', 'goals', 'assists', 'attPts', 'cardPts', 'xp']) out[k] = mix(a[k], b[k]);
    for (const st of Object.keys(a.stats)) {
      out.stats[st] = { mu: mix(a.stats[st].mu, b.stats[st].mu), pts: mix(a.stats[st].pts, b.stats[st].pts), atLeast: {} };
      for (const k of Object.keys(a.stats[st].atLeast)) out.stats[st].atLeast[k] = mix(a.stats[st].atLeast[k], b.stats[st].atLeast[k]);
    }
  } else {
    const t = mins / 90;
    out = defCompsAt(d, f, mins, curveAt(p.appCurve, mins, 'pts'), false);
    out.p60 = p60;
    out.csPts = p.scoring.cleanSheet * f.cs * p60;
    out.xp += out.csPts;
  }
  state.cache[key] = out;
  return out;
}
function fixtureXp(d, f) {
  const key = `${d.id}|${f.oppId}|${f.date}`;
  if (state.cache[key]) return state.cache[key];
  const p = state.plan, sc = p.scoring, club = state.clubs[d.club];
  const out = { cs: f.cs, csPts: sc.cleanSheet * f.cs, gcPts: f.gcPts, stats: {} };
  let total = sc.appearance + out.csPts + out.gcPts;
  for (const [stat, cfg] of Object.entries(STATS)) {
    const mu = expectedCount(stat, d, club, f);
    const s = nbSteps(mu, p.model[stat].r, cfg.per, cfg.steps);
    out.stats[stat] = { mu, pts: s.pts, atLeast: s.atLeast };
    total += s.pts;
  }
  const scale = f.lamOwn / p.lamAvg;
  out.goals = d.rates.g90 * scale; out.assists = d.rates.a90 * scale;
  out.attPts = sc.goal * out.goals + sc.assist * out.assists;
  out.cardPts = sc.yellow * d.rates.y90 + sc.red * d.rates.r90;
  total += out.attPts + out.cardPts + p.other;
  out.xp = total;
  state.cache[key] = out;
  return out;
}
function weekXp(d, gw) {
  const w = weekOf(d.club, gw);
  if (!w || !w.games) return { xp90: 0, fx: [] };
  const fx = w.fx.map(f => ({ f, r: state.plan.v2 ? fixtureXpV2(d, f, minsOf(d)) : fixtureXp(d, f) }));
  return { xp90: fx.reduce((a, x) => a + x.r.xp, 0), fx };
}
// v2 fixtures already include expected minutes; the old model is "if he plays 90" x minutes / 90
const dxp = (d, gw) => (state.plan.v2 ? 1 : minsOf(d) / 90) * weekXp(d, gw).xp90;

/* ---------- tooltip ---------- */
const tip = document.createElement('div');
tip.className = 'cp-tip';
tip.style.display = 'none';
document.body.appendChild(tip);

function breakdownRows(items) {
  const sc = state.plan.scoring;
  const n = items.length;
  const sum = fn => items.reduce((a, x) => a + fn(x), 0);
  const one = n === 1 ? items[0] : null;
  const statRow = stat => {
    const cfg = STATS[stat];
    const detail = one
      ? `${cfg.steps.map((s, i) => `${i ? '+ ' : '['}P(${s}+) ${pct(one.r.stats[stat].atLeast[s])}`).join(' ')} + …] · exp. ${one.r.stats[stat].mu.toFixed(1)}`
      : `+1 per ${cfg.per} · exp. ${sum(x => x.r.stats[stat].mu).toFixed(1)}`;
    return [cfg.name, detail, sum(x => x.r.stats[stat].pts)];
  };
  const v2 = state.plan.v2;
  return [
    v2 ? ['Appearance', `expected, from ${items[0].r.mins} expected minutes`, sum(x => x.r.app)] : ['Appearance', `${sc.appearance}${n > 1 ? ' × ' + n : ''}`, sc.appearance * n],
    ['Clean sheet', `${sc.cleanSheet} × ${one ? pct(one.r.cs) : sum(x => x.r.cs).toFixed(2) + ' expected'}${v2 && one ? ` × P(60+ mins) ${pct(one.r.p60)}` : ''}`, sum(x => x.r.csPts)],
    ['Goals conceded', one ? `−[P(2+) ${pct(one.f.pg2)} + P(4+) ${pct(one.f.pg4)} + …] · exp. ${one.f.xgc.toFixed(2)}` : '−1 per 2', sum(x => x.r.gcPts)],
    statRow('clr'), statRow('tkl'), statRow('blk'),
    ['Goals / assists', `${sc.goal} × ${sum(x => x.r.goals).toFixed(3)} + ${sc.assist} × ${sum(x => x.r.assists).toFixed(3)}`, sum(x => x.r.attPts)],
    ['Cards and other', 'his card rate', sum(x => x.r.cardPts + (v2 ? 0 : state.plan.other))],
  ];
}
function showTip(el, x, y) {
  const d = state.plan.defenders.find(q => q.id === +el.dataset.player);
  if (!d) return;
  const wk = weekXp(d, +el.dataset.gw);
  if (!wk.fx.length) return;
  const items = el.dataset.tip === 'fx' ? [wk.fx[+el.dataset.i]] : wk.fx;
  const club = state.clubs[d.club];
  const rows = breakdownRows(items);
  const total = items.reduce((a, it) => a + it.r.xp, 0);
  const mins = minsOf(d);
  const title = items.length === 1 ? `${club.short} v ${shortOf(items[0].f.opp)} (${items[0].f.ha}) · ${fmtDate(items[0].f.date, true)}` : `${d.name} · GW ${el.dataset.gw}`;
  let style = '';
  if (items.length === 1) {
    const opp = state.clubs[items[0].f.oppId];
    style = `<div class="cp-tip-foot">Style, clearances per defender per 90: ${esc(club.short)} defenders ${club.style.clr.team.toFixed(1)}; defenders facing ${esc(opp ? opp.short : '?')} ${opp ? opp.style.clr.opp.toFixed(1) : '?'} (league ${state.plan.model.clr.prior ? state.plan.model.clr.prior.toFixed(1) : '5.6'}). His role: ${d.role.clr.toFixed(2)}× team-mates.</div>`;
  }
  const v2 = state.plan.v2;
  tip.innerHTML = `<div class="cp-tip-title">${esc(title)} <span>${esc(d.name)}: ${v2 ? `expected points with ${mins} expected minutes` : 'expected points if he plays the whole game'}</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">${v2 ? 'Total' : 'If he plays 90'}</td><td class="cp-tip-num">${fmt2(total)}</td></tr>
    ${v2 || el.dataset.tip === 'fx' ? '' : `<tr class="cp-tip-total"><td colspan="2">× expected minutes ${mins}/90</td><td class="cp-tip-num">${fmt2(total * mins / 90)}</td></tr>`}</table>${style}`;
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
  const next5 = plan.gameweeks.filter(g => g.gw >= gw).slice(0, 5).map(g => g.gw);
  const list = plan.defenders.filter(d => {
    const c = state.clubs[d.club];
    return c && (state.showAll || minsOf(d) >= (plan.v2 ? 30 : 1)) && (state.league === 'All' || c.league === state.league)
      && (!q || d.name.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q));
  }).map(d => ({ d, c: state.clubs[d.club], wk: weekXp(d, gw), xp: dxp(d, gw), n5: next5.reduce((a, g) => a + dxp(d, g), 0) }))
    .sort((a, b) => b.xp - a.xp);
  const shown = list.slice(0, 150);
  const options = plan.gameweeks.map(g => `<option value="${g.gw}" ${g.gw === gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  const stale = plan.latestCompletedGw > plan.startersFromGw
    ? `<div class="cp-warn">⚠ Expected starters use lineups up to GW ${plan.startersFromGw}, but GW ${plan.latestCompletedGw} has finished. Refresh after exporting the latest stats.</div>` : '';
  const rows = shown.map(({ d, c, wk, xp, n5 }, i) => {
    const sumStat = s => wk.fx.reduce((a, x) => a + x.r.stats[s].mu, 0);
    return `<tr class="${minsOf(d) <= 0 ? 'cp-dim' : ''}">
      <td><span class="cp-rank ${i < 3 ? 'cp-rank-' + (i + 1) : 'cp-rank-n'}">${i + 1}</span></td>
      <td><strong>${esc(d.name)}</strong> ${tags(d)}<div class="cp-muted cp-small">${d.starts60} games of 60+ this season</div></td>
      <td class="pp-club">${shirtIcon(c, false)}${spList(d.tags)}<div>${esc(c.short)}<div class="cp-muted cp-small">${esc(c.league)}</div></div></td>
      <td class="cp-center opt2">${recentHtml(d.recent)}</td>
      <td class="cp-center"><input type="number" class="kp-mins ${edited(d) ? 'kp-mins-edited' : ''}" data-player="${d.id}" min="0" max="90" step="5" value="${minsOf(d)}" title="Expected minutes: 90 = plays the whole game; 45 = a 50% chance. Default ${d.xMins}." />${state.plan.p60Curve && minsOf(d) > 0 ? `<div class="cp-muted cp-small" title="Chance he plays 60+ minutes (the minutes model); expected minutes are an average that includes the chance he misses out">60+: ${pct(curveAt(state.plan.p60Curve, minsOf(d), 'p'))}</div>` : ''}</td>
      <td>${wk.fx.length ? wk.fx.map((x, j) => `<span class="cp-fx" data-tip="fx" data-player="${d.id}" data-gw="${gw}" data-i="${j}"><span class="cp-dot ${x.f.src === 'market' ? 'cp-dot-market' : 'cp-dot-model'}"></span>${fxName(x.f.opp, esc(shortOf(x.f.opp)))} (${x.f.ha}) <span class="cp-fx-xp">${fmt(x.r.xp)}</span></span>`).join('') : '<span class="cp-muted">No fixture</span>'}</td>
      <td class="cp-center opt">${wk.fx.length ? wk.fx.map(x => pct(x.r.cs)).join(' / ') : '–'}</td>
      <td class="cp-center opt">${wk.fx.length ? sumStat('clr').toFixed(1) : '–'}</td>
      <td class="cp-center opt">${wk.fx.length ? sumStat('tkl').toFixed(1) : '–'}</td>
      <td class="cp-xp" data-tip="total" data-player="${d.id}" data-gw="${gw}">${fmt(xp)}</td>
      <td class="cp-right cp-muted opt">${fmt(n5)}</td></tr>`;
  }).join('');
  const t = s => plan.model[s].test;
  root.innerHTML = `
    <p class="cp-intro">Expected points (xP) for defenders, built from the parts that score: appearance, clean sheet (+5), goals conceded (−1 per 2),
      clearances (+1 per 4), tackles (+1 per 2), blocks (+1 per 2), goals (+7), assists (+3) and cards. Clean sheets and goals conceded come from the match odds
      (or our team-strength model); clearances, tackles and blocks come from a model of the player's role, his club's style and the opponent's style.
      Edit expected minutes to change a prediction. Hover a fixture or an xP for the breakdown.</p>
    <div class="cp-controls">
      <button id="kp-prev" class="cp-nav-btn">&#8592;</button><select id="kp-gw">${options}</select><button id="kp-next" class="cp-nav-btn">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="kp-search" type="text" placeholder="Search defender or club..." value="${esc(state.query)}" autocomplete="off" />
      <label class="kp-toggle"><input type="checkbox" id="kp-all" ${state.showAll ? 'checked' : ''}/> Show non-starters and injured</label>
      ${Object.keys(state.mins).length ? '<button class="cp-link-btn" id="kp-reset">Reset minutes</button>' : ''}
    </div>
    ${stale}
    <h2 class="cp-h2">Defenders for GW ${gw} <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.marketGames} of ${meta.games} games priced from odds · ${list.length} shown${list.length > shown.length ? ' (top 150 listed)' : ''}</span></h2>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th>#</th><th>Defender</th><th>Club</th><th class="cp-center opt2" title="Minutes in his last 5 games for this club, oldest first (dark = 60+, light = came on or off, grey = did not play)">Last 5 (mins)</th><th class="cp-center" title="Editable. 90 = plays the whole game; 45 = a 50% chance he plays. Saved in this browser.">Exp. mins</th>
        <th>Fixtures (xP if he plays 90)</th><th class="cp-center opt">Clean sheet</th><th class="cp-center opt">Exp. clearances</th><th class="cp-center opt">Exp. tackles</th>
        <th class="cp-right">xP</th><th class="cp-right opt" title="Expected points over this and the next 4 gameweeks">Next 5 GWs</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="11" class="cp-muted">No defenders match.</td></tr>'}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      ${plan.v2 ? `<li><strong>Expected minutes</strong> come from our minutes model (recent minutes, starts and appearances at his club, how long since
        he last played, rest days, doubles; 0 if injured or suspended). Appearance points and the chance of reaching 60 minutes (needed for the clean sheet)
        follow from it; everything else scales with minutes. Edit the box to override; edits are saved in this browser.</li>
      <li><strong>Clearances, tackles, blocks</strong> = his share relative to defensive team-mates × his team-mates' current rate × how much this week's
        opponent makes defenders work × the odds, plus his FotMob history over his last 40 games (any EFL club). Numbers from another league are adjusted by
        the measured step-up / step-down change. Points use the full chance of reaching each step (clearance points = P(4+) + P(8+) + …).</li>
      <li><strong>Goals and assists</strong> come from predicted xG and xA (FotMob), converted to goals and assists, rather than his raw goal record.</li>
      <li>Tested on this season's games without having seen them (log-likelihood, closer to 0 is better; new model vs previous): clearances
        ${t('clr').v2.loglik.toFixed(3)} vs ${t('clr').v1.loglik.toFixed(3)}, blocks ${t('blk').v2.loglik.toFixed(3)} vs ${t('blk').v1.loglik.toFixed(3)},
        tackles ${t('tkl').v2.loglik.toFixed(3)} vs ${t('tkl').v1.loglik.toFixed(3)}, goals ${t('goals').v2.loglik.toFixed(3)} vs ${t('goals').v1.loglik.toFixed(3)},
        assists ${t('assists').v2.loglik.toFixed(3)} vs ${t('assists').v1.loglik.toFixed(3)}. Single games are still mostly luck.</li>`
      : `<li><strong>Expected minutes</strong> default to 90 for defenders who played 60+ minutes in their club's latest game, else 0 (and 0 if injured or suspended).
        Edit the box: xP scales with minutes/90, so 45 means a 50% chance he plays. Edits are saved in this browser; "Reset minutes" clears them.</li>`}
      <li>Later gameweeks use the team-strength model and today's expected starters and club styles, so they are less certain.</li>
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
    const d = state.plan.defenders.find(q => q.id === +inp.dataset.player);
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
    const res = await fetch('data/defender_plan.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load defender data (${res.status})`);
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
