const statusEl = document.getElementById('status');
const root = document.getElementById('kp-root');
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];
const MINS_KEY = 'efl_keeper_mins_v1';

const state = { plan: null, clubs: {}, clubWeek: {}, shortByName: {}, gw: null, league: 'All', query: '', showBackups: false, mins: {} };

/* expected minutes: 90 for the expected starter, 0 otherwise; edits are saved in this browser */
function loadMins() { try { state.mins = JSON.parse(localStorage.getItem(MINS_KEY) || '{}') || {}; } catch (e) { state.mins = {}; } }
function saveMins() { try { localStorage.setItem(MINS_KEY, JSON.stringify(state.mins)); } catch (e) { /* ignore */ } }
const minsOf = k => (state.mins[k.id] !== undefined ? state.mins[k.id] : k.xMins);
const edited = k => state.mins[k.id] !== undefined && state.mins[k.id] !== k.xMins;

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
const gwMeta = gw => state.plan.gameweeks.find(g => g.gw === gw);
const shortOf = name => state.shortByName[name] || name;
const kxp = (k, gw) => { const w = weekOf(k.club, gw); return w ? (minsOf(k) / 90) * w.xp : 0; };

/* ---------- tooltip ---------- */
const tip = document.createElement('div');
tip.className = 'cp-tip';
tip.style.display = 'none';
document.body.appendChild(tip);

function breakdown(fixtures, title, mins) {
  const c = state.plan.constants;
  const n = fixtures.length;
  const sum = k => fixtures.reduce((a, f) => a + f[k], 0);
  const rows = [
    ['Appearance', `${c.appearance}${n > 1 ? ' × ' + n : ''}`, c.appearance * n],
    ['Clean sheet', `${c.cleanSheet} × ${n > 1 ? sum('cs').toFixed(2) + ' expected' : pct(sum('cs'))}`, c.cleanSheet * sum('cs')],
    ['Goals conceded', n > 1 ? `−1 per 2 (exp. ${sum('xgc').toFixed(2)} conceded)` : `−[P(2+) ${pct(fixtures[0].pg2)} + P(4+) ${pct(fixtures[0].pg4)} + …] · exp. ${fixtures[0].xgc.toFixed(2)} conceded`, sum('gcPts')],
    ['Saves', n > 1 ? `+2 per 3 (exp. ${sum('saves').toFixed(1)} saves)` : `2 × [P(3+) ${pct(fixtures[0].ps3)} + P(6+) ${pct(fixtures[0].ps6)} + P(9+) ${pct(fixtures[0].ps9)}] · exp. ${fixtures[0].saves.toFixed(1)} saves`, sum('savePts')],
    ['Penalty saves', 'keeper average', c.pen * n],
    ['Cards', 'keeper average', c.cards * n],
  ];
  const total = sum('xp');
  const scaled = mins !== undefined
    ? `<tr class="cp-tip-total"><td colspan="2">× expected minutes ${mins}/90</td><td class="cp-tip-num">${fmt2(total * mins / 90)}</td></tr>` : '';
  return `<div class="cp-tip-title">${esc(title)} <span>expected points if he plays the whole game</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">If he plays 90</td><td class="cp-tip-num">${fmt2(total)}</td></tr>${scaled}</table>
    <div class="cp-tip-foot">${fixtures.map(f => `${esc(shortOf(f.opp))} (${f.ha}) ${fmt2(f.xp)} · ${f.src === 'market' ? 'odds' : 'model'}`).join(' + ')}</div>`;
}
function showTip(el, x, y) {
  const k = state.plan.keepers.find(q => q.id === +el.dataset.keeper);
  const w = k && weekOf(k.club, +el.dataset.gw);
  if (!w || !w.fx.length) return;
  if (el.dataset.tip === 'fx') {
    const f = w.fx[+el.dataset.i];
    tip.innerHTML = breakdown([f], `${state.clubs[k.club].short} v ${shortOf(f.opp)} (${f.ha}) · ${fmtDate(f.date, true)}`);
  } else {
    tip.innerHTML = breakdown(w.fx, `${k.name} · GW ${w.gw}`, minsOf(k));
  }
  tip.style.display = 'block';
  moveTip(x, y);
}
function moveTip(x, y) {
  if (window.innerWidth > 0) tip.style.maxWidth = Math.min(520, window.innerWidth - 16) + 'px'; // small screens
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

/* ---------- rendering ---------- */
function statusTags(k) {
  const t = [];
  if (k.injury || k.status === 'injured') t.push(`<span class="cp-tag kp-tag-out" title="${esc(k.injuryStatus || '')}">${esc(k.injury || 'injured')}</span>`);
  if (k.suspended) t.push('<span class="cp-tag kp-tag-out">suspended</span>');
  if (k.startedLast) t.push('<span class="cp-tag cp-tag-picked" title="Started his club\'s most recent game">started last</span>');
  return t.join(' ');
}
function render() {
  const { plan } = state;
  const scrollY = window.scrollY;
  const gw = state.gw;
  const meta = gwMeta(gw);
  const q = state.query.trim().toLowerCase();
  const next5 = plan.gameweeks.filter(g => g.gw >= gw).slice(0, 5).map(g => g.gw);
  let list = plan.keepers.filter(k => {
    const c = state.clubs[k.club];
    return c && (state.showBackups || minsOf(k) > 0) && (state.league === 'All' || c.league === state.league)
      && (!q || k.name.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q));
  }).map(k => ({ k, c: state.clubs[k.club], w: weekOf(k.club, gw), xp: kxp(k, gw), n5: next5.reduce((a, g) => a + kxp(k, g), 0) }))
    .sort((a, b) => b.xp - a.xp);

  const options = plan.gameweeks.map(g => `<option value="${g.gw}" ${g.gw === gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  const stale = plan.latestCompletedGw > plan.startersFromGw
    ? `<div class="cp-warn">⚠ Expected starters use lineups up to GW ${plan.startersFromGw}, but GW ${plan.latestCompletedGw} has finished. Refresh after exporting the latest stats.</div>` : '';
  const rows = list.map(({ k, c, w, xp, n5 }, i) => `
    <tr class="${minsOf(k) <= 0 ? 'cp-dim' : ''}">
      <td><span class="cp-rank ${i < 3 ? 'cp-rank-' + (i + 1) : 'cp-rank-n'}">${i + 1}</span></td>
      <td><strong>${esc(k.name)}</strong> ${statusTags(k)}<div class="cp-muted cp-small">${k.starts} starts this season</div></td>
      <td>${esc(c.short)}<div class="cp-muted cp-small">${esc(c.league)}</div></td>
      <td class="cp-center"><input type="number" class="kp-mins ${edited(k) ? 'kp-mins-edited' : ''}" data-keeper="${k.id}" min="0" max="90" step="5" value="${minsOf(k)}" title="Expected minutes: 90 = plays the whole game; 45 = a 50% chance. Default ${k.xMins}." /></td>
      <td>${w && w.games ? w.fx.map((f, j) => `<span class="cp-fx" data-tip="fx" data-keeper="${k.id}" data-gw="${gw}" data-i="${j}"><span class="cp-dot ${f.src === 'market' ? 'cp-dot-market' : 'cp-dot-model'}"></span>${esc(shortOf(f.opp))} (${f.ha}) <span class="cp-fx-xp">${fmt(f.xp)}</span></span>`).join('') : '<span class="cp-muted">No fixture</span>'}</td>
      <td class="cp-center">${w && w.games ? w.fx.map(f => pct(f.cs)).join(' / ') : '–'}</td>
      <td class="cp-center">${w && w.games ? w.fx.reduce((a, f) => a + f.saves, 0).toFixed(1) : '–'}</td>
      <td class="cp-xp" data-tip="total" data-keeper="${k.id}" data-gw="${gw}">${fmt(xp)}</td>
      <td class="cp-right cp-muted">${fmt(n5)}</td>
    </tr>`).join('');
  const m = plan.savesModel.test;
  root.innerHTML = `
    <p class="cp-intro">Expected points (xP) for goalkeepers, built from the parts that score: appearance, clean sheet (+5), goals conceded (−1 per 2),
      saves (+2 per 3), penalty saves and cards. Clean sheets and goals conceded come from the match odds (or our team-strength model when there
      are no odds yet); saves come from a model trained on three seasons of shots data. xP is multiplied by the chance he starts.
      Hover a fixture or an xP for the breakdown.</p>
    <div class="cp-controls">
      <button id="kp-prev" class="cp-nav-btn">&#8592;</button><select id="kp-gw">${options}</select><button id="kp-next" class="cp-nav-btn">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="kp-search" type="text" placeholder="Search keeper or club..." value="${esc(state.query)}" autocomplete="off" />
      <label class="kp-toggle"><input type="checkbox" id="kp-backups" ${state.showBackups ? 'checked' : ''}/> Show backups and injured</label>
      ${Object.keys(state.mins).length ? '<button class="cp-link-btn" id="kp-reset">Reset minutes</button>' : ''}
    </div>
    ${stale}
    <h2 class="cp-h2">Keepers for GW ${gw} <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.marketGames} of ${meta.games} games priced from odds</span></h2>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th>#</th><th>Keeper</th><th>Club</th><th class="cp-center" title="Editable. 90 = plays the whole game; 45 = a 50% chance he plays. Saved in this browser.">Exp. mins</th><th>Fixtures (xP if he starts)</th>
        <th class="cp-center">Clean sheet</th><th class="cp-center">Exp. saves</th><th class="cp-right">xP</th><th class="cp-right" title="Expected points over this and the next 4 gameweeks">Next 5 GWs</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="9" class="cp-muted">No keepers match.</td></tr>'}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      <li><strong>Expected minutes</strong> default to 90 for the keeper who started his club's last game and 0 for everyone else (and 0 if injured or
        suspended). Edit the box to change a prediction: xP scales with minutes/90, so 45 means a 50% chance he plays. Historically the last game's starter keeps
        his place ${pct(plan.constants.pKeep)} of the time. Edits are saved in this browser; "Reset minutes" clears them.</li>
      <li>Saves and goals conceded score in steps, so xP uses the full chances: saves points = 2 × [P(3+ saves) + P(6+) + P(9+) …], not expected saves ÷ 3 × 2.</li>
      <li>Single games are mostly luck: a clean sheet is a weighted coin flip. xP is right on average and ranks fixtures; it does not predict one match.</li>
      <li>The saves model uses the odds-implied expected goals; recent shots-on-target and save-rate features were tested and added nothing.
        On this season's ${m.n} games (not used to fit it): average saves-points error ${m.model.mae_save_pts.toFixed(2)} vs ${m.baseline.mae_save_pts.toFixed(2)} for a league-average guess.</li>
      <li>Later gameweeks use the team-strength model and today's expected starters, so they are less certain.</li>
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
  document.getElementById('kp-backups').addEventListener('change', e => { state.showBackups = e.target.checked; render(); });
  document.querySelectorAll('.kp-mins').forEach(inp => inp.addEventListener('change', () => {
    const v = Math.max(0, Math.min(90, Math.round(+inp.value || 0)));
    const k = state.plan.keepers.find(q => q.id === +inp.dataset.keeper);
    if (v === k.xMins) delete state.mins[k.id]; else state.mins[k.id] = v;
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
    const res = await fetch('data/keeper_plan.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load keeper data (${res.status})`);
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
