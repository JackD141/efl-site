const statusEl = document.getElementById('status');
const root = document.getElementById('cp-root');

const state = { plan: null, byId: {}, shortByName: {}, gw: null, league: 'All', query: '', showAll: false };
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmt = n => n.toFixed(1);
function fmtDate(iso) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
}
function rankBadge(rank, title) {
  const cls = rank <= 3 ? `cp-rank-${rank}` : 'cp-rank-n';
  const t = title || (rank === 1 ? 'Best club this gameweek' : `Ranked ${rank} of 72 clubs this gameweek`);
  return `<span class="cp-rank ${cls}" title="${esc(t)}">${rank}</span>`;
}
function srcDot(src) {
  return src === 'market'
    ? '<span class="cp-dot cp-dot-market" title="Bookmaker odds"></span>'
    : '<span class="cp-dot cp-dot-model" title="Model estimate (no bookmaker odds yet)"></span>';
}
function fixturesHtml(week) {
  if (!week.games) return '<span class="cp-muted">No fixture</span>';
  return week.fx.map(f =>
    `<span class="cp-fx">${srcDot(f.src)}${esc(state.shortByName[f.opp] || f.opp)} (${f.ha}) <span class="cp-fx-xp">${fmt(f.xp)}</span></span>`
  ).join('');
}
function weekOf(club, gw) { return club.weeks.find(w => w.gw === gw); }
function gwMeta(gw) { return state.plan.gameweeks.find(g => g.gw === gw); }

function visibleClubs() {
  const q = state.query.trim().toLowerCase();
  return state.plan.clubs.filter(c =>
    (state.league === 'All' || c.league === state.league) &&
    (!q || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q)));
}

function renderControls() {
  const { plan } = state;
  const options = plan.gameweeks.map(g =>
    `<option value="${g.gw}" ${g.gw === state.gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  return `
    <p class="cp-intro">Pick 2 clubs a week; each club can only be picked <strong>5 times</strong> a season. Expected points (xP) come from our team-strength
    model: club scoring is win +5, draw +3, away win +2, clean sheet +2, 2+ goals +2, 4+ goals +2. A club with two fixtures scores both.
    Click any club to see its best weeks.</p>
    <div class="cp-controls">
      <button id="cp-prev" class="cp-nav-btn" title="Previous gameweek">&#8592;</button>
      <select id="cp-gw">${options}</select>
      <button id="cp-next" class="cp-nav-btn" title="Next gameweek">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="cp-search" type="text" placeholder="Search a club..." value="${esc(state.query)}" autocomplete="off" />
    </div>`;
}

function renderBest() {
  const { gw } = state;
  const meta = gwMeta(gw);
  let rows = visibleClubs()
    .map(c => ({ c, w: weekOf(c, gw) }))
    .filter(r => r.w && r.w.games > 0)
    .sort((a, b) => b.w.xp - a.w.xp);
  const total = rows.length;
  if (!state.showAll) rows = rows.slice(0, 15);
  const body = rows.map(({ c, w }) => {
    const k = c.top5.indexOf(gw);
    const star = k >= 0
      ? `<span class="cp-star" title="One of ${esc(c.short)}'s 5 best weeks this season (#${k + 1} of 5 by xP)">&#9733; #${k + 1}</span>`
      : '<span class="cp-muted">–</span>';
    const best = c.top5.length ? weekOf(c, c.top5[0]) : null;
    return `<tr>
      <td>${rankBadge(w.rank)}</td>
      <td><a href="#" class="cp-club" data-club="${c.id}">${esc(c.name)}</a>${w.locked ? ' <span class="cp-tag" title="This club has already played this gameweek, so it is locked">locked</span>' : ''}${w.games > 1 ? ' <span class="cp-tag cp-tag-double">double</span>' : ''}</td>
      <td class="cp-muted">${esc(c.league)}</td>
      <td>${fixturesHtml(w)}</td>
      <td class="cp-xp">${fmt(w.xp)}</td>
      <td class="cp-center">${star}</td>
      <td class="cp-muted cp-center cp-nowrap">${best ? 'GW ' + best.gw + ' · ' + fmt(best.xp) : ''}</td>
    </tr>`;
  }).join('');
  const quality = meta.marketGames === meta.games
    ? 'All fixtures priced from bookmaker odds.'
    : meta.marketGames === 0
      ? 'Model estimates only (no bookmaker odds yet): treat as indicative. The further ahead, the less certain.'
      : `${meta.marketGames} of ${meta.games} fixtures priced from bookmaker odds; the rest are model estimates.`;
  return `
    <h2 class="cp-h2">Best clubs for GW ${gw}
      <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.games} games${meta.doubles ? ` · ${meta.doubles} clubs play twice` : ''}${meta.blanks ? ` · ${meta.blanks} clubs blank` : ''}</span></h2>
    <p class="cp-note">${esc(quality)}</p>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th title="Rank among all 72 clubs this gameweek">#</th><th>Club</th><th>League</th><th>Fixtures (xP each)</th><th class="cp-right">xP</th>
        <th class="cp-center" title="Is this one of the club's 5 best weeks of the season?">Top-5 week</th><th class="cp-center" title="The club's best week of the season">Best week</th></tr></thead>
      <tbody>${body || '<tr><td colspan="7" class="cp-muted">No clubs match.</td></tr>'}</tbody>
    </table></div>
    ${total > 15 ? `<button id="cp-more" class="cp-link-btn">${state.showAll ? 'Show top 15' : `Show all ${total} clubs with a fixture`}</button>` : ''}`;
}

function renderGlance() {
  const rows = state.plan.gameweeks.map(g => {
    const top = visibleClubs()
      .map(c => ({ c, w: weekOf(c, g.gw) }))
      .filter(r => r.w && r.w.games > 0)
      .sort((a, b) => b.w.xp - a.w.xp)
      .slice(0, 3);
    const cells = top.map(({ c, w }) =>
      `<a href="#" class="cp-club" data-club="${c.id}">${esc(c.short)}</a> <span class="cp-muted">${fmt(w.xp)}${w.games > 1 ? ' ×2' : ''}</span>`).join('<span class="cp-sep">·</span>');
    const q = g.marketGames === g.games ? 'market' : g.marketGames === 0 ? 'model' : 'mixed';
    return `<tr class="${g.gw === state.gw ? 'cp-current' : ''}" data-gw="${g.gw}">
      <td><a href="#" class="cp-gw-link" data-gw="${g.gw}"><strong>GW ${g.gw}</strong></a></td>
      <td class="cp-muted">${fmtDate(g.start)}</td>
      <td class="cp-muted">${g.doubles ? g.doubles + ' double' : ''}${g.doubles && g.blanks ? ' · ' : ''}${g.blanks ? g.blanks + ' blank' : ''}${!g.doubles && !g.blanks ? '–' : ''}</td>
      <td>${cells}</td>
      <td class="cp-center"><span class="cp-q cp-q-${q}" title="${q === 'market' ? 'All fixtures priced from bookmaker odds' : q === 'model' ? 'Model estimates only' : 'Some fixtures priced from bookmaker odds'}">${q}</span></td>
    </tr>`;
  }).join('');
  return `
    <h2 class="cp-h2">Season at a glance <span class="cp-sub">top 3 clubs each gameweek</span></h2>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th>GW</th><th>From</th><th>Schedule</th><th>Best clubs (xP)</th><th class="cp-center">Odds</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

function render() {
  const { plan } = state;
  const asOf = new Date(plan.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  root.innerHTML = renderControls() + renderBest() + renderGlance() +
    `<p class="cp-foot">${esc(plan.notes.horizon)} Updated ${esc(asOf)}. Weeks with no fixture score 0, so do not pick a club that blanks.
    <span class="cp-legend"><span class="cp-dot cp-dot-market"></span> bookmaker odds <span class="cp-dot cp-dot-model"></span> model estimate</span></p>
    <div id="cp-modal" class="games-modal" style="display:none"><div class="games-modal-content cp-modal-content" id="cp-modal-body"></div></div>`;
  bind();
}

function bind() {
  document.getElementById('cp-gw').addEventListener('change', e => { state.gw = +e.target.value; state.showAll = false; render(); });
  const step = d => {
    const i = state.plan.gameweeks.findIndex(g => g.gw === state.gw) + d;
    if (i >= 0 && i < state.plan.gameweeks.length) { state.gw = state.plan.gameweeks[i].gw; state.showAll = false; render(); }
  };
  document.getElementById('cp-prev').addEventListener('click', () => step(-1));
  document.getElementById('cp-next').addEventListener('click', () => step(1));
  document.querySelectorAll('.cp-chip').forEach(b => b.addEventListener('click', () => { state.league = b.dataset.league; state.showAll = false; render(); }));
  const search = document.getElementById('cp-search');
  search.addEventListener('input', e => {
    state.query = e.target.value;
    const pos = e.target.selectionStart;
    render();
    const s = document.getElementById('cp-search');
    s.focus();
    s.setSelectionRange(pos, pos);
  });
  const more = document.getElementById('cp-more');
  if (more) more.addEventListener('click', () => { state.showAll = !state.showAll; render(); });
  document.querySelectorAll('.cp-club').forEach(a => a.addEventListener('click', e => { e.preventDefault(); openClub(+a.dataset.club); }));
  document.querySelectorAll('.cp-gw-link').forEach(a => a.addEventListener('click', e => {
    e.preventDefault(); state.gw = +a.dataset.gw; state.showAll = false; render(); window.scrollTo({ top: 0, behavior: 'smooth' });
  }));
  const modal = document.getElementById('cp-modal');
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
}

function openClub(id) {
  const c = state.byId[id];
  if (!c) return;
  const best = c.top5.map(gw => weekOf(c, gw)).sort((a, b) => b.xp - a.xp);
  const rows = best.map((w, i) => {
    const meta = gwMeta(w.gw);
    const firstDate = w.fx.length ? w.fx[0].date : meta.start;
    return `<tr>
      <td>${rankBadge(w.rank)}</td>
      <td><strong>GW ${w.gw}</strong><div class="cp-muted cp-small">${fmtDate(firstDate)}${w.games > 1 ? ' · double' : ''}</div></td>
      <td>${fixturesHtml(w)}</td>
      <td class="cp-xp">${fmt(w.xp)}</td>
    </tr>`;
  }).join('');
  const maxXp = Math.max(...c.weeks.map(w => w.xp), 1);
  const top5 = new Set(c.top5);
  const bars = c.weeks.map(w => {
    const h = w.games ? Math.max(4, (w.xp / maxXp) * 100) : 2;
    const cls = !w.games ? 'cp-bar-blank' : top5.has(w.gw) ? 'cp-bar-top' : '';
    const tip = w.games ? `GW ${w.gw}: ${fmt(w.xp)} xP, ranked ${w.rank} of 72${w.games > 1 ? ' (double)' : ''}` : `GW ${w.gw}: no fixture`;
    return `<div class="cp-bar-col" title="${esc(tip)}"><div class="cp-bar ${cls}" style="height:calc((100% - 16px) * ${(h / 100).toFixed(3)})"></div><div class="cp-bar-label">${w.gw}</div></div>`;
  }).join('');
  const body = document.getElementById('cp-modal-body');
  body.innerHTML = `
    <span class="games-modal-close" id="cp-close">&times;</span>
    <h3 class="cp-modal-title">${esc(c.name)} <span class="cp-sub">${esc(c.league)}</span></h3>
    <p class="cp-note">Best 5 weeks to use ${esc(c.short)} (up to 5 picks a season): <strong>${fmt(c.top5Total)} xP</strong> in total.
      The badge shows how this club ranks among all 72 clubs that week; <strong>1 = the best club that week</strong>.</p>
    <div class="cp-table-wrap"><table class="cp-table cp-table-tight">
      <thead><tr><th title="Rank among all 72 clubs that gameweek">Rank</th><th>Week</th><th>Fixtures (xP each)</th><th class="cp-right">xP</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="cp-muted">No fixtures left.</td></tr>'}</tbody>
    </table></div>
    <h4 class="cp-h4">xP by gameweek <span class="cp-sub">orange = this club's top 5</span></h4>
    <div class="cp-bars">${bars}</div>`;
  const modal = document.getElementById('cp-modal');
  modal.style.display = 'flex';
  document.getElementById('cp-close').addEventListener('click', closeModal);
}

function closeModal() {
  const m = document.getElementById('cp-modal');
  if (m) m.style.display = 'none';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

async function load() {
  statusEl.textContent = 'Loading...';
  try {
    const res = await fetch('data/club_plan.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Could not load club plan (${res.status})`);
    state.plan = await res.json();
    for (const c of state.plan.clubs) {
      state.byId[c.id] = c;
      state.shortByName[c.name] = c.short;
    }
    // default to the next gameweek that still has unlocked fixtures
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
