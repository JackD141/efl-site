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
  let z = m.intercept;
  m.features.forEach((name, i) => {
    const v = name === 'home' ? vals[name] : Math.log(Math.max(vals[name], 1e-3));
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
  const fx = w.fx.map(f => ({ f, r: fixtureXp(d, f) }));
  return { xp90: fx.reduce((a, x) => a + x.r.xp, 0), fx };
}
const dxp = (d, gw) => (minsOf(d) / 90) * weekXp(d, gw).xp90;

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
  return [
    ['Appearance', `${sc.appearance}${n > 1 ? ' × ' + n : ''}`, sc.appearance * n],
    ['Clean sheet', `${sc.cleanSheet} × ${one ? pct(one.r.cs) : sum(x => x.r.cs).toFixed(2) + ' expected'}`, sum(x => x.r.csPts)],
    ['Goals conceded', one ? `−[P(2+) ${pct(one.f.pg2)} + P(4+) ${pct(one.f.pg4)} + …] · exp. ${one.f.xgc.toFixed(2)}` : '−1 per 2', sum(x => x.r.gcPts)],
    statRow('clr'), statRow('tkl'), statRow('blk'),
    ['Goals / assists', `${sc.goal} × ${sum(x => x.r.goals).toFixed(3)} + ${sc.assist} × ${sum(x => x.r.assists).toFixed(3)}`, sum(x => x.r.attPts)],
    ['Cards and other', 'his card rate', sum(x => x.r.cardPts + state.plan.other)],
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
  tip.innerHTML = `<div class="cp-tip-title">${esc(title)} <span>${esc(d.name)}: expected points if he plays the whole game</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">If he plays 90</td><td class="cp-tip-num">${fmt2(total)}</td></tr>
    ${el.dataset.tip === 'fx' ? '' : `<tr class="cp-tip-total"><td colspan="2">× expected minutes ${mins}/90</td><td class="cp-tip-num">${fmt2(total * mins / 90)}</td></tr>`}</table>${style}`;
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
    return c && (state.showAll || minsOf(d) > 0) && (state.league === 'All' || c.league === state.league)
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
      <td>${esc(c.short)}<div class="cp-muted cp-small">${esc(c.league)}</div></td>
      <td class="cp-center"><input type="number" class="kp-mins ${edited(d) ? 'kp-mins-edited' : ''}" data-player="${d.id}" min="0" max="90" step="5" value="${minsOf(d)}" title="Expected minutes: 90 = plays the whole game; 45 = a 50% chance. Default ${d.xMins}." /></td>
      <td>${wk.fx.length ? wk.fx.map((x, j) => `<span class="cp-fx" data-tip="fx" data-player="${d.id}" data-gw="${gw}" data-i="${j}"><span class="cp-dot ${x.f.src === 'market' ? 'cp-dot-market' : 'cp-dot-model'}"></span>${esc(shortOf(x.f.opp))} (${x.f.ha}) <span class="cp-fx-xp">${fmt(x.r.xp)}</span></span>`).join('') : '<span class="cp-muted">No fixture</span>'}</td>
      <td class="cp-center">${wk.fx.length ? wk.fx.map(x => pct(x.r.cs)).join(' / ') : '–'}</td>
      <td class="cp-center">${wk.fx.length ? sumStat('clr').toFixed(1) : '–'}</td>
      <td class="cp-center">${wk.fx.length ? sumStat('tkl').toFixed(1) : '–'}</td>
      <td class="cp-xp" data-tip="total" data-player="${d.id}" data-gw="${gw}">${fmt(xp)}</td>
      <td class="cp-right cp-muted">${fmt(n5)}</td></tr>`;
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
      <thead><tr><th>#</th><th>Defender</th><th>Club</th><th class="cp-center" title="Editable. 90 = plays the whole game; 45 = a 50% chance he plays. Saved in this browser.">Exp. mins</th>
        <th>Fixtures (xP if he plays 90)</th><th class="cp-center">Clean sheet</th><th class="cp-center">Exp. clearances</th><th class="cp-center">Exp. tackles</th>
        <th class="cp-right">xP</th><th class="cp-right" title="Expected points over this and the next 4 gameweeks">Next 5 GWs</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="10" class="cp-muted">No defenders match.</td></tr>'}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      <li><strong>Expected minutes</strong> default to 90 for defenders who played 60+ minutes in their club's latest game, else 0 (and 0 if injured or suspended).
        Edit the box: xP scales with minutes/90, so 45 means a 50% chance he plays. Edits are saved in this browser; "Reset minutes" clears them.</li>
      <li><strong>Clearances, tackles, blocks</strong> = his role (how many he makes compared with team-mates, measured at whichever club he was at, so it moves with him)
        × his club's current defensive style × how much this week's opponent makes defenders work (long-ball sides force more clearances) × the odds.
        Each club's and opponent's style is a rolling average of recent gameweeks. Points come from the full chances of reaching each step: e.g. clearance points = P(4+) + P(8+) + P(12+) + …</li>
      <li>Tested on this season's games without having seen them: clearance points error ${t('clr').model.mae_pts.toFixed(3)} vs ${t('clr').baseline.mae_pts.toFixed(3)} for the player's own recent rate
        (the style features help most here); tackles and blocks are about level with that baseline. Whole-game xP beats the player's recent average points
        (error 2.69 vs 2.81) and ranks fixtures sensibly, but single games are mostly luck.</li>
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
