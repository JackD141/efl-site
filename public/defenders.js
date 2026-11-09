const statusEl = document.getElementById('status');
const root = document.getElementById('kp-root');
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];
const STATS = { clr: { name: 'Clearances', per: 4, steps: [4, 8, 12] }, blk: { name: 'Blocks', per: 2, steps: [2, 4] }, tkl: { name: 'Tackles', per: 2, steps: [2, 4] } };

// stat columns are hidden unless "Show stats" is ticked (remembered in this browser, shared by all Player Picks tabs)
const STATS_KEY = 'efl_picks_show_stats';
function loadShowStats() { try { return localStorage.getItem(STATS_KEY) === '1'; } catch (e) { return false; } }
const state = { plan: null, clubs: {}, clubWeek: {}, shortByName: {}, gw: null, league: 'All', query: '', showAll: false, start: {}, cache: {}, sortBy: 'starts', showStats: loadShowStats() };

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

/* start chance in his club's next game: the start model (starts.js), or your edit (saved in this browser) */
const startOf = d => (state.start[d.id] !== undefined ? state.start[d.id] : d.pStart || 0);
const edited = d => state.start[d.id] !== undefined && Math.abs(state.start[d.id] - (d.pStart || 0)) > 1e-9;

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
// v2 expected points for one fixture when he starts with chance ps: P(full game) x his usual full game (clean sheet
// counts) + P(part game) x a typical part game (starts.js), so the clearance / tackle / block steps are evaluated on
// minutes he would really play
const fullMins = d => d.mFull || state.plan.startModel.full;
function fixtureXpV2(d, f, ps) {
  const key = `${d.id}|${f.oppId}|${f.date}|${ps}`;
  if (state.cache[key]) return state.cache[key];
  const p = state.plan, s = scenarioProbs(p, d, ps);
  const a = defCompsAt(d, f, fullMins(d), 2, true), b = defCompsAt(d, f, p.startModel.part, 1, false);
  const mix = (x, y) => s.pFull * x + s.pPart * y;
  const out = { ps, mins: s.pFull * fullMins(d) + s.pPart * p.startModel.part, p60: s.pFull, pPart: s.pPart, cs: f.cs, stats: {} };
  for (const k of ['app', 'csPts', 'gcPts', 'goals', 'assists', 'attPts', 'cardPts', 'xp']) out[k] = mix(a[k], b[k]);
  for (const st of Object.keys(a.stats)) {
    out.stats[st] = { mu: mix(a.stats[st].mu, b.stats[st].mu), pts: mix(a.stats[st].pts, b.stats[st].pts), atLeast: {} };
    for (const k of Object.keys(a.stats[st].atLeast)) out.stats[st].atLeast[k] = mix(a.stats[st].atLeast[k], b.stats[st].atLeast[k]);
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
// asStarter: "if he starts" (his club's first game of the gameweek at 100%; a double's second game uses his chance of
// starting it given he started the first). The old model (no start model in the plan): "if he plays 90" x start chance.
function weekXp(d, gw, asStarter) {
  const w = weekOf(d.club, gw);
  if (!w || !w.games) return { xp90: 0, fx: [] };
  const fx = w.fx.map((f, j) => {
    const ps = startChanceAt(d, j, startOf(d), asStarter);
    if (state.plan.v2) return { f, r: fixtureXpV2(d, f, ps) };
    const r = fixtureXp(d, f);
    return { f, r: Object.assign({}, r, { xp: ps * r.xp }) };
  });
  return { xp90: fx.reduce((a, x) => a + x.r.xp, 0), fx };
}
const dxp = (d, gw) => weekXp(d, gw, false).xp90;
// null if he is not expected to be available (injured / suspended: start and bench chances 0)
const startsXp = (d, gw) => ((d.pStart || 0) > 0 || edited(d) ? weekXp(d, gw, true).xp90 : null);

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
    v2 ? ['Appearance', one ? `starts ${pct(one.r.ps)} · 60+ ${pct(one.r.p60)} · part game ${pct(one.r.pPart)}` : `per game, ${n} games`, sum(x => x.r.app)] : ['Appearance', `${sc.appearance}${n > 1 ? ' × ' + n : ''}`, sc.appearance * n],
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
  const asStarter = el.dataset.mode === 'starts';
  const wk = weekXp(d, +el.dataset.gw, asStarter);
  if (!wk.fx.length) return;
  const items = el.dataset.tip === 'fx' ? [wk.fx[+el.dataset.i]] : wk.fx;
  const club = state.clubs[d.club];
  const rows = breakdownRows(items);
  const total = items.reduce((a, it) => a + it.r.xp, 0);
  const title = items.length === 1 ? `${club.short} v ${shortOf(items[0].f.opp)} (${items[0].f.ha}) · ${fmtDate(items[0].f.date, true)}` : `${d.name} · GW ${el.dataset.gw}`;
  let style = '';
  if (items.length === 1) {
    const opp = state.clubs[items[0].f.oppId];
    style = `<div class="cp-tip-foot">Style, clearances per defender per 90: ${esc(club.short)} defenders ${club.style.clr.team.toFixed(1)}; defenders facing ${esc(opp ? opp.short : '?')} ${opp ? opp.style.clr.opp.toFixed(1) : '?'} (league ${state.plan.model.clr.prior ? state.plan.model.clr.prior.toFixed(1) : '5.6'}). His role: ${d.role.clr.toFixed(2)}× team-mates.</div>`;
  }
  const v2 = state.plan.v2;
  tip.innerHTML = `<div class="cp-tip-title">${esc(title)} <span>${esc(d.name)}: ${asStarter ? 'expected points if he starts' : `expected points with a ${pct(startOf(d))} chance he starts`}</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">Total</td><td class="cp-tip-num">${fmt2(total)}</td></tr></table>${style}`;
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
  if (d.injury || d.status === 'injured') t.push(`<span class="cp-tag kp-tag-out" title="${esc(d.injuryStatus || '')}">${esc(d.injury || 'injured')}${/\((\d+)%\)/.test(d.injuryStatus || '') && !/\(0%\)/.test(d.injuryStatus) ? ' ?' : ''}</span>`);
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
    return c && (state.showAll || startOf(d) >= 0.3) && (state.league === 'All' || c.league === state.league)
      && (!q || d.name.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q));
  }).map(d => ({ d, c: state.clubs[d.club], wk: weekXp(d, gw, false), xp: dxp(d, gw), sx: startsXp(d, gw), n5: next5.reduce((a, g) => a + dxp(d, g), 0) }))
    .sort((a, b) => (state.sortBy === 'starts' ? (b.sx ?? -1) - (a.sx ?? -1) : b.xp - a.xp));
  const shown = list.slice(0, 150);
  const options = plan.gameweeks.map(g => `<option value="${g.gw}" ${g.gw === gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  const stale = plan.latestCompletedGw > plan.startersFromGw
    ? `<div class="cp-warn">⚠ Expected starters use lineups up to GW ${plan.startersFromGw}, but GW ${plan.latestCompletedGw} has finished. Refresh after exporting the latest stats.</div>` : '';
  const rows = shown.map(({ d, c, wk, xp, sx, n5 }, i) => {
    const sumStat = s => wk.fx.reduce((a, x) => a + x.r.stats[s].mu, 0);
    return `<tr class="${startOf(d) <= 0 ? 'cp-dim' : ''}">
      <td><span class="cp-rank ${i < 3 ? 'cp-rank-' + (i + 1) : 'cp-rank-n'}">${i + 1}</span></td>
      <td><strong>${esc(d.name)}</strong> ${tags(d)}<div class="cp-muted cp-small">${d.starts60} games of 60+ this season</div></td>
      <td class="pp-club">${shirtIcon(c, false)}${spList(d.tags)}<div>${esc(c.short)}<div class="cp-muted cp-small">${esc(c.league)}</div></div></td>
      <td class="cp-xp" data-tip="total" data-player="${d.id}" data-gw="${gw}">${fmt(xp)}</td>
      <td class="cp-xp cp-xp-starts" data-tip="total" data-mode="starts" data-player="${d.id}" data-gw="${gw}">${sx === null ? '–' : fmt(sx)}</td>
      <td class="cp-center">${plan.startModel ? startCellHtml(plan, d, startOf(d), edited(d), `data-player="${d.id}"`, wk.fx.length) : ''}</td>
      <td>${wk.fx.length ? wk.fx.map((x, j) => `<span class="cp-fx" data-tip="fx" data-player="${d.id}" data-gw="${gw}" data-i="${j}"><span class="cp-dot ${x.f.src === 'market' ? 'cp-dot-market' : 'cp-dot-model'}"></span>${fxName(x.f.opp, esc(shortOf(x.f.opp)))} (${x.f.ha}) <span class="cp-fx-xp">${fmt(x.r.xp)}</span></span>`).join('') : '<span class="cp-muted">No fixture</span>'}</td>
      <td class="cp-center opt2">${recentHtml(d.recent)}</td>
      <td class="cp-center opt st">${wk.fx.length ? wk.fx.map(x => pct(x.r.cs)).join(' / ') : '–'}</td>
      <td class="cp-center opt st">${wk.fx.length ? sumStat('clr').toFixed(1) : '–'}</td>
      <td class="cp-center opt st">${wk.fx.length ? sumStat('tkl').toFixed(1) : '–'}</td>
      <td class="cp-right cp-muted opt">${fmt(n5)}</td></tr>`;
  }).join('');
  const t = s => plan.model[s].test;
  root.innerHTML = `
    <p class="cp-intro">Expected points (xP) for defenders, built from the parts that score: appearance, clean sheet (+5), goals conceded (−1 per 2),
      clearances (+1 per 4), tackles (+1 per 2), blocks (+1 per 2), goals (+7), assists (+3) and cards. Clean sheets and goals conceded come from the match odds
      (or our team-strength model); clearances, tackles and blocks come from a model of the player's role, his club's style and the opponent's style.
      Change a start chance after team news. Hover a fixture or an xP for the breakdown.</p>
    <div class="cp-controls">
      <button id="kp-prev" class="cp-nav-btn">&#8592;</button><select id="kp-gw">${options}</select><button id="kp-next" class="cp-nav-btn">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="kp-search" type="text" placeholder="Search defender or club..." value="${esc(state.query)}" autocomplete="off" />
      <label class="kp-toggle"><input type="checkbox" id="kp-all" ${state.showAll ? 'checked' : ''}/> Show players under 30% to start</label>
      <label class="kp-toggle" title="Per-game stat columns (scoring chance, shots, passes, clean sheets...)"><input type="checkbox" id="kp-stats" ${state.showStats ? 'checked' : ''}/> Show stats</label>
      ${Object.keys(state.start).length ? '<button class="cp-link-btn" id="kp-reset">Reset start chances</button>' : ''}
    </div>
    ${stale}
    <h2 class="cp-h2">Defenders for GW ${gw} <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.marketGames} of ${meta.games} games priced from odds · ${list.length} shown${list.length > shown.length ? ' (top 150 listed)' : ''}</span></h2>
    <div class="cp-table-wrap"><table class="cp-table ${state.showStats ? '' : 'cp-hide-stats'}">
      <thead><tr><th>#</th><th>Defender</th><th>Club</th><th class="cp-right cp-sort ${state.sortBy === 'xp' ? 'cp-sort-on' : ''}" data-sort="xp" title="Expected points before team news: includes the chance he does not start. Click to sort.">xP</th><th class="cp-right cp-sort ${state.sortBy === 'starts' ? 'cp-sort-on' : ''}" data-sort="starts" title="Expected points if he starts his club's first game this gameweek (picks can be changed after team news): that game at his usual full-game minutes. In a double, the second game still uses his expected minutes, since whether he starts it is not known yet. Click to sort.">xP if starts</th><th class="cp-center" title="Editable. Chance he starts his club's next game (the start model); change it after team news. Below: his minutes if he starts and, in a double, his chance of starting the second game. Saved in this browser.">Start %</th>
        <th>Fixtures (xP)</th><th class="cp-center opt2" title="Minutes in his last 5 games for this club, oldest first (dark = 60+, light = came on or off, grey = did not play)">Last 5 (mins)</th><th class="cp-center opt st" title="Clean-sheet chance, per game">CS</th><th class="cp-center opt st" title="Expected clearances this gameweek">Clr</th><th class="cp-center opt st" title="Expected tackles this gameweek">Tkl</th><th class="cp-right opt" title="Expected points over this and the next 4 gameweeks">Next 5 GWs</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="12" class="cp-muted">No defenders match.</td></tr>'}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      <li><strong>xP if starts</strong> (the default sort): Fantasy EFL lets you change a player until his game kicks off, so after team news the
        question is what he scores <em>if he starts</em>: his club's first game of the gameweek at 100% (his usual full game, clean sheet counting),
        a double's second game at his chance of starting it given he started the first. <strong>xP</strong> is the same before team news, with his
        start chance. Click either heading to sort.</li>
      ${plan.v2 ? `<li><strong>Start %</strong> is the chance he starts his club's next game, from a model of his recent starts, minutes and appearances
        at his club, how long since he last played, rest days and doubles. A club starts ten outfield players, so each club's chances add up to ten: they rise when team-mates are out (0 if EFL lists him injured or suspended, halved if "questionable"; games he missed through a ban do not
        count as being dropped). Below it: his minutes if he starts and, in a double, his chance of starting the second game. Change it after team news
        (100 = in the line-up); edits are saved in this browser.</li>
      <li><strong>Clearances, tackles, blocks</strong> = his share relative to defensive team-mates × his team-mates' current rate × how much this week's
        opponent makes defenders work × the odds, plus his FotMob history over his last 40 games (any EFL club). Numbers from another league are adjusted by
        the measured step-up / step-down change. Points use the full chance of reaching each step (clearance points = P(4+) + P(8+) + …).</li>
      <li><strong>Goals and assists</strong> come from predicted xG and xA (FotMob), converted to goals and assists, rather than his raw goal record.</li>
      <li>Checked on this season's games with the model fitted on last season only (log-likelihood, closer to 0 is better; the model vs its plainest version): clearances
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
    const v = Math.max(0, Math.min(100, Math.round(+inp.value || 0))) / 100;
    const d = state.plan.defenders.find(q => q.id === +inp.dataset.player);
    if (Math.round((d.pStart || 0) * 100) / 100 === v) delete state.start[d.id]; else state.start[d.id] = v;
    writeStore(START_KEYS.DEF, state.start); render();
  }));
  document.getElementById('kp-stats').addEventListener('change', e => {
    state.showStats = e.target.checked;
    try { localStorage.setItem(STATS_KEY, state.showStats ? '1' : '0'); } catch (err) { /* ignore */ }
    render();
  });
  document.querySelectorAll('th[data-sort]').forEach(th => th.addEventListener('click', () => { state.sortBy = th.dataset.sort; render(); }));
  const reset = document.getElementById('kp-reset');
  if (reset) reset.addEventListener('click', () => { state.start = {}; writeStore(START_KEYS.DEF, state.start); render(); });
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
    state.start = readStore(START_KEYS.DEF);
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
