const statusEl = document.getElementById('status');
const root = document.getElementById('cp-root');

const MAX_USES = 5;
const PICKS_PER_WEEK = 2;
const STORE_KEY = 'efl_club_planner_v1';
const PROFILES = ['Jack', 'John'];
const LEAGUES = ['All', 'Championship', 'League 1', 'League 2'];

const state = {
  plan: null, byId: {}, shortByName: {}, clubWeek: {},
  gw: null, league: 'All', query: '', showAll: false,
  panelOpen: true, planOpen: false,
  store: { active: 'Jack', picks: { Jack: {}, John: {} } },
  result: null,
  cloud: { pass: '', msg: '', kind: '', busy: false },
};

/* ---------- helpers ---------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmt = n => n.toFixed(1);
const fmt2 = n => n.toFixed(2);
const pct = p => (p * 100).toFixed(1) + '%';
const odds = p => (p > 0 ? (1 / p).toFixed(2) : '–');
function fmtDate(iso, withDay) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-GB', withDay ? { weekday: 'short', day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short' });
}
function rankBadge(rank, title) {
  const cls = rank <= 3 ? `cp-rank-${rank}` : 'cp-rank-n';
  const t = title || (rank === 1 ? 'Best club this gameweek' : `Ranked ${rank} of 72 clubs this gameweek`);
  return `<span class="cp-rank ${cls}" title="${esc(t)}">${rank}</span>`;
}
function srcDot(src) {
  return src === 'market'
    ? '<span class="cp-dot cp-dot-market" title="Priced from bookmaker odds"></span>'
    : '<span class="cp-dot cp-dot-model" title="Model estimate (no bookmaker odds yet)"></span>';
}
const weekOf = (club, gw) => state.clubWeek[club.id] && state.clubWeek[club.id][gw];
const gwMeta = gw => state.plan.gameweeks.find(g => g.gw === gw);
const short = c => c.short || c.name;

/* ---------- saved picks (this browser only) ---------- */
// Keep only picks whose club exists in the data, so stale or hand-edited saves can never break the page.
function sanitizePicks(p) {
  const picks = {};
  let dropped = 0;
  for (const [gw, ids] of Object.entries(p && typeof p === 'object' ? p : {})) {
    if (!Array.isArray(ids)) { dropped++; continue; }
    const clean = [0, 1].map(i => (ids[i] && state.byId[ids[i]] ? ids[i] : null));
    dropped += [0, 1].filter(i => ids[i] && !clean[i]).length;
    if (clean[0] || clean[1]) picks[gw] = clean;
  }
  return { picks, dropped };
}
function loadStore() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s && s.picks) {
      for (const p of PROFILES) state.store.picks[p] = sanitizePicks(s.picks[p]).picks;
      if (PROFILES.includes(s.active)) state.store.active = s.active;
    }
  } catch (e) { /* storage unavailable: carry on without saving */ }
}
function saveStore() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state.store)); } catch (e) { /* ignore */ }
}
const activePicks = () => state.store.picks[state.store.active];
function usedCounts() {
  const used = {};
  for (const gw of Object.keys(activePicks())) {
    for (const id of activePicks()[gw] || []) if (id) used[id] = (used[id] || 0) + 1;
  }
  return used;
}
const picksLeft = id => MAX_USES - (usedCounts()[id] || 0);
function pickWarnings() {
  const out = [];
  const used = usedCounts();
  for (const [id, n] of Object.entries(used)) if (n > MAX_USES) out.push(`${state.byId[id].name} is entered ${n} times (max ${MAX_USES}).`);
  for (const [gw, ids] of Object.entries(activePicks())) if (ids[0] && ids[0] === ids[1]) out.push(`GW ${gw}: the same club is entered twice.`);
  return out;
}

function addPick(gw, id) {
  const picks = activePicks();
  const cur = (picks[gw] || [null, null]).slice();
  if (cur.includes(id) || (cur[0] && cur[1])) return;
  cur[cur[0] ? 1 : 0] = id;
  picks[gw] = cur;
  saveStore();
  render();
}
function removePick(gw, id) {
  const picks = activePicks();
  const cur = (picks[gw] || []).map(x => (x === id ? null : x));
  if (!cur[0] && !cur[1]) delete picks[gw]; else picks[gw] = cur;
  saveStore();
  render();
}

/* ---------- cloud backup (api/club-picks.js) ---------- */
async function cloudCall(action) {
  const c = state.cloud;
  const name = state.store.active;
  if (c.pass.length < 4) { c.msg = 'Enter a passphrase of at least 4 characters.'; c.kind = 'err'; render(); return; }
  c.busy = true; c.msg = action === 'save' ? 'Saving...' : 'Loading...'; c.kind = '';
  render();
  try {
    const res = await fetch('/api/club-picks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, name, passphrase: c.pass, picks: action === 'save' ? activePicks() : undefined }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    const when = new Date(data.savedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
    if (action === 'save') {
      c.msg = `Saved ${name}'s picks to the cloud (${when})${data.created ? '. Passphrase set: use it to load them elsewhere.' : '.'}`; c.kind = 'ok';
    } else if (Object.keys(activePicks()).length && !window.confirm(`Replace the picks in this browser for ${name} with the ones saved in the cloud (${when})?`)) {
      c.msg = 'Load cancelled.'; c.kind = '';
    } else {
      const clean = sanitizePicks(data.picks);
      state.store.picks[name] = clean.picks;
      saveStore();
      c.msg = `Loaded ${name}'s picks saved ${when}.${clean.dropped ? ` ${clean.dropped} pick(s) ignored: club not found.` : ''}`; c.kind = 'ok';
    }
  } catch (e) {
    c.msg = e.message; c.kind = 'err';
  } finally {
    c.busy = false;
    render();
  }
}

/* ---------- optimiser: exact min-cost flow ----------
   Source -> club (capacity = picks left) -> gameweek (1 pick per club per week, profit = xP) -> sink (open slots that week). */
class MinCostFlow {
  constructor(n) { this.n = n; this.g = Array.from({ length: n }, () => []); this.to = []; this.cap = []; this.cost = []; }
  add(u, v, cap, cost) {
    this.g[u].push(this.to.length); this.to.push(v); this.cap.push(cap); this.cost.push(cost);
    this.g[v].push(this.to.length); this.to.push(u); this.cap.push(0); this.cost.push(-cost);
    return this.to.length - 2;
  }
  run(s, t) {
    const n = this.n;
    for (;;) {
      const dist = new Array(n).fill(Infinity), inq = new Array(n).fill(false), prev = new Array(n).fill(-1);
      dist[s] = 0;
      const q = [s];
      inq[s] = true;
      while (q.length) {
        const u = q.shift();
        inq[u] = false;
        for (const e of this.g[u]) {
          const v = this.to[e];
          if (this.cap[e] > 0 && dist[u] + this.cost[e] < dist[v] - 1e-9) {
            dist[v] = dist[u] + this.cost[e];
            prev[v] = e;
            if (!inq[v]) { inq[v] = true; q.push(v); }
          }
        }
      }
      if (dist[t] >= -1e-9) break; // no profitable pick left
      for (let v = t; v !== s;) { const e = prev[v]; this.cap[e] -= 1; this.cap[e ^ 1] += 1; v = this.to[e ^ 1]; }
    }
  }
}

/* weeks: [{gw, slots, taken:Set}], capacity: {clubId: n}. Returns {total, byWeek:{gw:[clubIds]}} */
function solve(weeks, capacity) {
  const clubs = state.plan.clubs.filter(c => capacity[c.id] > 0);
  const S = 0, T = 1, C0 = 2, W0 = C0 + clubs.length, N = W0 + weeks.length;
  const f = new MinCostFlow(N);
  clubs.forEach((c, i) => f.add(S, C0 + i, capacity[c.id], 0));
  const edges = [];
  weeks.forEach((w, j) => {
    if (w.slots > 0) f.add(W0 + j, T, w.slots, 0);
    clubs.forEach((c, i) => {
      const wk = weekOf(c, w.gw);
      if (w.slots > 0 && wk && wk.games > 0 && !wk.locked && !w.taken.has(c.id)) {
        edges.push({ id: f.add(C0 + i, W0 + j, 1, -wk.xp), club: c.id, j, xp: wk.xp });
      }
    });
  });
  f.run(S, T);
  const byWeek = {};
  let total = 0;
  for (const e of edges) {
    if (f.cap[e.id] === 0) {
      (byWeek[weeks[e.j].gw] = byWeek[weeks[e.j].gw] || []).push(e.club);
      total += e.xp;
    }
  }
  return { total, byWeek };
}

function planWeeks() {
  const picks = activePicks();
  return state.plan.gameweeks.map(g => {
    const have = (picks[g.gw] || []).filter(Boolean);
    return { gw: g.gw, slots: Math.max(0, PICKS_PER_WEEK - have.length), taken: new Set(have) };
  });
}
function capacities() {
  const used = usedCounts();
  const cap = {};
  for (const c of state.plan.clubs) cap[c.id] = Math.max(0, MAX_USES - (used[c.id] || 0));
  return cap;
}
function greedy(weeks, capacity) {
  const cap = { ...capacity };
  let total = 0;
  for (const w of weeks) {
    const cand = state.plan.clubs
      .map(c => ({ c, wk: weekOf(c, w.gw) }))
      .filter(x => x.wk && x.wk.games > 0 && !x.wk.locked && cap[x.c.id] > 0 && !w.taken.has(x.c.id))
      .sort((a, b) => b.wk.xp - a.wk.xp).slice(0, w.slots);
    for (const x of cand) { cap[x.c.id] -= 1; total += x.wk.xp; }
  }
  return total;
}
function computeResult() {
  const weeks = planWeeks();
  const capacity = capacities();
  const best = solve(weeks, capacity);
  const target = weeks.find(w => w.gw === state.plan.firstGw);
  const next = target && target.slots > 0 ? target : null; // suggestions only for the gameweek you can still pick for
  const options = [];
  if (next) {
    const cand = state.plan.clubs
      .map(c => ({ c, wk: weekOf(c, next.gw) }))
      .filter(x => x.wk && x.wk.games > 0 && !x.wk.locked && capacity[x.c.id] > 0 && !next.taken.has(x.c.id))
      .sort((a, b) => b.wk.xp - a.wk.xp).slice(0, 14);
    for (const x of cand) {
      const w2 = weeks.map(w => (w.gw === next.gw ? { ...w, slots: w.slots - 1, taken: new Set([...w.taken, x.c.id]) } : w));
      const cap2 = { ...capacity, [x.c.id]: capacity[x.c.id] - 1 };
      const r = solve(w2, cap2);
      options.push({ club: x.c, xp: x.wk.xp, planTotal: x.wk.xp + r.total, left: capacity[x.c.id] - 1 });
    }
    options.sort((a, b) => b.planTotal - a.planTotal);
  }
  state.result = { weeks, capacity, best, next, target, options, greedyTotal: greedy(weeks, capacity) };
}

/* ---------- tooltips ---------- */
const tip = document.createElement('div');
tip.className = 'cp-tip';
tip.style.display = 'none';
document.body.appendChild(tip);

function fixtureOf(el) {
  const club = state.byId[+el.dataset.club];
  const wk = club && weekOf(club, +el.dataset.gw);
  return { club, wk, f: wk && wk.fx[+el.dataset.i] };
}
function oddsTip(club, wk, f) {
  const lose = Math.max(0, 1 - f.win - f.draw);
  const rows = [['Win', f.win, f.bk && f.bk[0]], ['Draw', f.draw, f.bk && f.bk[1]], ['Lose', lose, f.bk && f.bk[2]]];
  const hasMk = !!f.bk;
  const opp = esc(state.shortByName[f.opp] || f.opp);
  return `<div class="cp-tip-title">${esc(short(club))} v ${opp} (${f.ha}) <span>GW ${wk.gw} · ${fmtDate(f.date, true)}</span></div>
    <table class="cp-tip-table"><tr><th>${esc(short(club))}</th><th>Model</th>${hasMk ? '<th>Market</th>' : ''}</tr>
    ${rows.map(r => `<tr><td>${r[0]}</td><td><strong>${odds(r[1])}</strong> <span>${pct(r[1])}</span></td>${hasMk ? `<td><strong>${odds(r[2])}</strong> <span>${pct(r[2])}</span></td>` : ''}</tr>`).join('')}</table>
    <div class="cp-tip-foot">Decimal odds, no bookmaker margin. ${hasMk ? 'Market = bookmaker average with the margin removed.' : 'Model estimate: no bookmaker odds listed yet.'}</div>`;
}
function components(fixtures) {
  const sc = state.plan.scoring;
  const multi = fixtures.length > 1;
  const q = n => (multi ? n.toFixed(2) + ' expected' : pct(n)); // two games: expected number of times, not a probability
  const sum = k => fixtures.reduce((a, f) => a + f[k], 0);
  const away = fixtures.filter(f => f.ha === 'A').reduce((a, f) => a + f.win, 0);
  return [
    ['Win', `${sc.win} × ${q(sum('win'))}`, sc.win * sum('win')],
    ['Away win bonus', `${sc.awayWin} × ${q(away)}`, sc.awayWin * away],
    ['Draw', `${sc.draw} × ${q(sum('draw'))}`, sc.draw * sum('draw')],
    ['Clean sheet', `${sc.cleanSheet} × ${q(sum('cs'))}`, sc.cleanSheet * sum('cs')],
    ['2+ goals scored', `${sc.twoGoals} × ${q(sum('g2'))}`, sc.twoGoals * sum('g2')],
    ['4+ goals scored', `${sc.fourGoals} × ${q(sum('g4'))}`, sc.fourGoals * sum('g4')],
  ].filter(r => r[0] !== 'Away win bonus' || fixtures.some(f => f.ha === 'A'));
}
function xpTip(fixtures, title) {
  const rows = components(fixtures);
  const total = fixtures.reduce((a, f) => a + f.xp, 0);
  const per = fixtures.length > 1
    ? `<div class="cp-tip-foot">${fixtures.map(f => `${esc(state.shortByName[f.opp] || f.opp)} (${f.ha}) ${fmt2(f.xp)}`).join(' + ')}</div>` : '';
  return `<div class="cp-tip-title">${esc(title)} <span>expected points</span></div>
    <table class="cp-tip-table">${rows.map(r => `<tr><td>${r[0]}</td><td><span>${r[1]}</span></td><td class="cp-tip-num">${fmt2(r[2])}</td></tr>`).join('')}
    <tr class="cp-tip-total"><td colspan="2">Total</td><td class="cp-tip-num">${fmt2(total)}</td></tr></table>
    ${per}<div class="cp-tip-foot">Each probability × its points, added up. A loss scores 0.</div>`;
}
function showTip(el, x, y) {
  const type = el.dataset.tip;
  const { club, wk, f } = fixtureOf(el);
  if (!club || !wk) return;
  if (type === 'odds' && f) tip.innerHTML = oddsTip(club, wk, f);
  else if (type === 'xp' && f) tip.innerHTML = xpTip([f], `${short(club)} v ${state.shortByName[f.opp] || f.opp} (${f.ha})`);
  else if (type === 'xptotal' && wk.fx.length) tip.innerHTML = xpTip(wk.fx, `${short(club)} · GW ${wk.gw}`);
  else return;
  tip.style.display = 'block';
  moveTip(x, y);
}
function moveTip(x, y) {
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
function fixturesHtml(club, week) {
  if (!week.games) return '<span class="cp-muted">No fixture</span>';
  return week.fx.map((f, i) =>
    `<span class="cp-fx" data-tip="odds" data-club="${club.id}" data-gw="${week.gw}" data-i="${i}">${srcDot(f.src)}${esc(state.shortByName[f.opp] || f.opp)} (${f.ha}) <span class="cp-fx-xp" data-tip="xp" data-club="${club.id}" data-gw="${week.gw}" data-i="${i}">${fmt(f.xp)}</span></span>`
  ).join('');
}
function visibleClubs() {
  const q = state.query.trim().toLowerCase();
  return state.plan.clubs.filter(c =>
    (state.league === 'All' || c.league === state.league) &&
    (!q || c.name.toLowerCase().includes(q) || c.short.toLowerCase().includes(q)));
}
function leftBadge(id) {
  const n = picksLeft(id);
  const cls = n <= 0 ? 'cp-left-0' : n <= 2 ? 'cp-left-low' : '';
  return `<span class="cp-left ${cls}" title="${n} of ${MAX_USES} picks left for ${esc(state.store.active)}">${Math.max(0, n)}/${MAX_USES}</span>`;
}
function pickState(gw) {
  const r = state.result;
  const w = r.weeks.find(x => x.gw === gw);
  return { planned: new Set(r.best.byWeek[gw] || []), picked: new Set(w ? [...w.taken] : []) };
}

function renderControls() {
  const options = state.plan.gameweeks.map(g =>
    `<option value="${g.gw}" ${g.gw === state.gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');
  const chips = LEAGUES.map(l => `<button class="cp-chip ${state.league === l ? 'active' : ''}" data-league="${l}">${l}</button>`).join('');
  return `
    <p class="cp-intro">Pick 2 clubs a week; each club can only be picked <strong>${MAX_USES} times</strong> a season. Expected points (xP) come from our team-strength
    model: club scoring is win +5, draw +3, away win +2, clean sheet +2, 2+ goals +2, 4+ goals +2. A club with two fixtures scores both.
    Hover a fixture for its win/draw/lose odds, hover an xP number to see where it comes from, and click a club for its best weeks.</p>
    <div class="cp-controls">
      <button id="cp-prev" class="cp-nav-btn" title="Previous gameweek">&#8592;</button>
      <select id="cp-gw">${options}</select>
      <button id="cp-next" class="cp-nav-btn" title="Next gameweek">&#8594;</button>
      <div class="cp-chips">${chips}</div>
      <input id="cp-search" type="text" placeholder="Search a club..." value="${esc(state.query)}" autocomplete="off" />
    </div>`;
}

function clubOptions(selected) {
  const clubs = [...state.plan.clubs].sort((a, b) => a.name.localeCompare(b.name));
  return '<option value="">–</option>' + clubs.map(c => `<option value="${c.id}" ${c.id === selected ? 'selected' : ''}>${esc(c.name)}</option>`).join('');
}

function pairFaceEachOther(gw, ids) {
  if (ids.length < 2) return false;
  const [a, b] = ids.map(i => state.byId[i]);
  const wa = weekOf(a, gw);
  return !!(wa && wa.fx.some(f => f.opp === b.name));
}

function renderPlan() {
  const r = state.result;
  const profile = state.store.active;
  const gain = r.best.total - r.greedyTotal;
  let thisWeek = '';
  if (r.next) {
    const ids = r.best.byWeek[r.next.gw] || [];
    const already = [...r.next.taken].map(i => state.byId[i]);
    const chips = ids.map(i => {
      const c = state.byId[i];
      const wk = weekOf(c, r.next.gw);
      return `<div class="cp-suggest"><a href="#" class="cp-club" data-club="${c.id}">${esc(c.name)}</a>
        <div class="cp-suggest-fx">${fixturesHtml(c, wk)}</div><div class="cp-suggest-xp" data-tip="xptotal" data-club="${c.id}" data-gw="${r.next.gw}">${fmt(wk.xp)} xP</div>
        <div class="cp-muted cp-small">${picksLeft(c.id)} picks left before this week</div>
        <button class="cp-add-btn" data-add="${c.id}" data-gw="${r.next.gw}">Add to GW ${r.next.gw} picks</button></div>`;
    }).join('');
    const both = ids.length === 2 && r.next.slots === 2
      ? `<button class="cp-link-btn" data-add-both="${ids.join(',')}" data-gw="${r.next.gw}">Add both to GW ${r.next.gw} picks</button>` : '';
    const clash = pairFaceEachOther(r.next.gw, ids) ? '<div class="cp-warn">⚠ These two clubs play each other this week, so one of them will lose.</div>' : '';
    thisWeek = `<h4 class="cp-h4">Suggested picks for GW ${r.next.gw}${already.length ? ` <span class="cp-sub">(already picked: ${already.map(c => esc(short(c))).join(', ')})</span>` : ''}</h4>
      <div class="cp-suggest-row">${chips || '<span class="cp-muted">No eligible clubs.</span>'}</div>${both}${clash}`;
  } else if (r.target) {
    const cards = [...r.target.taken].map(i => {
      const c = state.byId[i];
      const wk = weekOf(c, r.target.gw);
      return `<div class="cp-suggest cp-suggest-set"><a href="#" class="cp-club" data-club="${c.id}">${esc(c.name)}</a> <span class="cp-set-tick">&#10003; picked</span>
        <div class="cp-suggest-fx">${wk ? fixturesHtml(c, wk) : ''}</div><div class="cp-suggest-xp" data-tip="xptotal" data-club="${c.id}" data-gw="${r.target.gw}">${wk ? fmt(wk.xp) : '0.0'} xP</div>
        <button class="cp-link-btn" data-remove="${c.id}" data-gw="${r.target.gw}">Remove</button></div>`;
    }).join('');
    const nextGw = state.plan.gameweeks[1] ? state.plan.gameweeks[1].gw : null;
    thisWeek = `<h4 class="cp-h4">GW ${r.target.gw} picks are set</h4><div class="cp-suggest-row">${cards}</div>
      <p class="cp-note">${nextGw ? `Suggestions for GW ${nextGw} appear after the next refresh, once GW ${r.target.gw}'s deadline has passed.` : 'No later gameweeks to pick for.'}</p>`;
  } else {
    thisWeek = '<p class="cp-note">No gameweek is open for picks.</p>';
  }
  const options = r.options.length ? `
    <h4 class="cp-h4">Other options this week <span class="cp-sub">season plan total if you pick this club now (best other picks chosen for the rest)</span></h4>
    <div class="cp-table-wrap"><table class="cp-table cp-table-tight"><thead><tr><th>Club</th><th class="cp-right">xP this week</th><th class="cp-right">Plan total</th><th class="cp-right">vs best</th><th class="cp-center">Left after</th></tr></thead><tbody>
    ${r.options.slice(0, 8).map((o, i) => `<tr><td><a href="#" class="cp-club" data-club="${o.club.id}">${esc(o.club.name)}</a></td><td class="cp-xp">${fmt(o.xp)}</td><td class="cp-xp">${fmt(o.planTotal)}</td>
      <td class="cp-right ${i === 0 || r.options[0].planTotal - o.planTotal < 0.05 ? 'cp-muted' : 'cp-neg'}">${i === 0 ? 'best' : (r.options[0].planTotal - o.planTotal < 0.05 ? '≈ same' : '−' + fmt(r.options[0].planTotal - o.planTotal))}</td><td class="cp-center">${o.left}</td></tr>`).join('')}
    </tbody></table></div>` : '';
  const weeksRows = state.plan.gameweeks.map(g => {
    const w = r.weeks.find(x => x.gw === g.gw);
    const ids = r.best.byWeek[g.gw] || [];
    const cells = [...[...w.taken].map(i => ({ id: i, fixed: true })), ...ids.map(i => ({ id: i, fixed: false }))];
    const xp = cells.reduce((a, x) => a + ((weekOf(state.byId[x.id], g.gw) || { xp: 0 }).xp), 0);
    const clash = pairFaceEachOther(g.gw, cells.map(x => x.id));
    return `<tr class="${r.next && g.gw === r.next.gw ? 'cp-current' : ''}"><td><a href="#" class="cp-gw-link" data-gw="${g.gw}"><strong>GW ${g.gw}</strong></a>${g.doubles ? ` <span class="cp-muted cp-small">${g.doubles} dbl</span>` : ''}</td>
      <td>${cells.map(x => `<a href="#" class="cp-club" data-club="${x.id}">${esc(short(state.byId[x.id]))}</a>${x.fixed ? '<span class="cp-muted cp-small"> (entered)</span>' : ''}`).join('<span class="cp-sep">·</span>') || '<span class="cp-muted">–</span>'}${clash ? ' <span title="These two clubs play each other">⚠</span>' : ''}</td>
      <td class="cp-xp">${fmt(xp)}</td></tr>`;
  }).join('');
  const usage = {};
  for (const [gw, ids] of Object.entries(r.best.byWeek)) for (const id of ids) (usage[id] = usage[id] || []).push(+gw);
  const usageRows = Object.entries(usage).sort((a, b) => b[1].length - a[1].length || state.byId[a[0]].name.localeCompare(state.byId[b[0]].name))
    .map(([id, gws]) => `<tr><td><a href="#" class="cp-club" data-club="${id}">${esc(state.byId[id].name)}</a></td><td>${gws.sort((a, b) => a - b).join(', ')}</td><td class="cp-center">${picksLeft(+id) - gws.length}</td></tr>`).join('');
  return `
    <hr class="cp-hr" />
    ${thisWeek}
    <p class="cp-note"><strong>Plan for the rest of the season (${esc(profile)}):</strong> ${fmt(r.best.total)} xP over the weeks still to pick, the best possible given picks left. Choosing the two best clubs each week one after another (greedy) would give ${fmt(r.greedyTotal)}, so planning adds ${fmt(gain)} xP.</p>
    <div class="cp-caveat"><strong>Before you rely on this</strong>
      <ul>
        <li><strong>Fixtures can change</strong>: postponements, cup ties and TV moves can add, remove or move games, including doubles and blanks. Check each week.</li>
        <li><strong>Our view of team strength can change</strong>: injuries, form and managers move quickly. Estimates are model-only beyond the current week, about 2 points less reliable on win probability next week than the bookmakers, and about 4 points at three months out. The plan is re-run each week, so only this week's picks are a real decision.</li>
        <li><strong>Variance</strong>: these are averages. One club game has a standard deviation of about 3.8 points (a double about 5.4), so a high-xP pick often scores 0 to 2 and a low one can score 11.</li>
        <li><strong>Covariance</strong>: the plan maximises total expected points and ignores risk. Two clubs playing each other in the same week cannot both win (marked ⚠), a club's two games in a double share form and injury risk, and strength errors persist across weeks.</li>
        <li>It assumes you keep to ${PICKS_PER_WEEK} picks a week, ${MAX_USES} per club, and that the picks you entered are correct.</li>
      </ul></div>
    ${options}
    <details class="cp-sub-details" id="cp-plan-details" ${state.planOpen ? 'open' : ''}><summary>Full season plan <span class="cp-sub">provisional: later weeks change as the season goes on</span></summary>
      <div class="cp-plan-grid">
        <div class="cp-table-wrap"><table class="cp-table cp-table-tight"><thead><tr><th>Gameweek</th><th>Picks</th><th class="cp-right">xP</th></tr></thead><tbody>${weeksRows}</tbody></table></div>
        <div class="cp-table-wrap"><table class="cp-table cp-table-tight"><thead><tr><th>Club</th><th>Planned weeks</th><th class="cp-center">Left after</th></tr></thead><tbody>${usageRows}</tbody></table></div>
      </div></details>`;
}

function renderPanel() {
  const profile = state.store.active;
  const picks = activePicks();
  const tabs = PROFILES.map(p => `<button class="cp-chip ${p === profile ? 'active' : ''}" data-profile="${p}">${p}</button>`).join('');
  const rows = [];
  for (let gw = 1; gw <= state.plan.firstGw; gw++) {
    const cur = picks[gw] || [];
    rows.push(`<tr><td><strong>GW ${gw}</strong>${gw === state.plan.firstGw ? ' <span class="cp-tag">this week</span>' : ''}</td>
      <td><select class="cp-pick" data-gw="${gw}" data-slot="0">${clubOptions(cur[0])}</select></td>
      <td><select class="cp-pick" data-gw="${gw}" data-slot="1">${clubOptions(cur[1])}</select></td></tr>`);
  }
  const used = usedCounts();
  const usedChips = Object.entries(used).sort((a, b) => b[1] - a[1] || state.byId[a[0]].name.localeCompare(state.byId[b[0]].name))
    .map(([id, n]) => `<span class="cp-used ${n >= MAX_USES ? 'cp-used-full' : ''}" title="${n} of ${MAX_USES} used">${esc(short(state.byId[id]))} ${n}/${MAX_USES}</span>`).join('');
  const warns = pickWarnings().map(w => `<div class="cp-warn">⚠ ${esc(w)}</div>`).join('');
  const nPicks = Object.values(used).reduce((a, b) => a + b, 0);
  return `
    <details class="cp-panel" id="cp-panel" ${state.panelOpen ? 'open' : ''}>
      <summary>My picks &amp; season plan <span class="cp-sub">${esc(profile)} · ${nPicks} of ${PICKS_PER_WEEK * state.plan.firstGw} picks entered</span></summary>
      <div class="cp-panel-body">
        <div class="cp-profiles">${tabs}</div>
        <p class="cp-note">Enter the two clubs ${esc(profile)} has picked in each gameweek so far (include this week if already set). Saved in this browser only, separately for each profile.</p>
        ${warns}
        <div class="cp-picks-grid">
          <div class="cp-table-wrap cp-picks-table"><table class="cp-table cp-table-tight">
            <thead><tr><th>Gameweek</th><th>Pick 1</th><th>Pick 2</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>
          <div class="cp-used-box"><h4 class="cp-h4">Picks used</h4>
            <div>${usedChips || '<span class="cp-muted">Nothing entered yet: every club has all 5 picks left.</span>'}</div>
            <div class="cp-btn-row"><button class="cp-link-btn" id="cp-export">Export</button><button class="cp-link-btn" id="cp-import">Import</button><button class="cp-link-btn" id="cp-clear">Clear ${esc(profile)}</button></div>
            <div class="cp-cloud"><h4 class="cp-h4">Cloud backup <span class="cp-sub">save here, load from any browser</span></h4>
              <input type="password" id="cp-cloud-pass" placeholder="Passphrase for ${esc(profile)}" value="${esc(state.cloud.pass)}" autocomplete="off" />
              <div class="cp-btn-row">
                <button class="cp-add-btn cp-cloud-btn" id="cp-cloud-save" ${state.cloud.busy ? 'disabled' : ''}>Save to cloud</button>
                <button class="cp-add-btn cp-cloud-btn cp-cloud-load" id="cp-cloud-load" ${state.cloud.busy ? 'disabled' : ''}>Load from cloud</button></div>
              <div class="cp-cloud-msg ${state.cloud.kind ? 'cp-cloud-' + state.cloud.kind : ''}">${esc(state.cloud.msg)}</div>
              <p class="cp-note cp-small">The first save sets the passphrase for ${esc(profile)}. Picks and a hash of the passphrase are stored in the public GitHub repo, so use a passphrase you do not use anywhere else.</p></div></div>
        </div>
        ${renderPlan()}
      </div>
    </details>`;
}

function renderBest() {
  const { gw } = state;
  const meta = gwMeta(gw);
  const ps = pickState(gw);
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
    const left = picksLeft(c.id);
    return `<tr class="${left <= 0 ? 'cp-dim' : ''}">
      <td>${rankBadge(w.rank)}</td>
      <td><a href="#" class="cp-club" data-club="${c.id}">${esc(c.name)}</a>${ps.picked.has(c.id) ? ' <span class="cp-tag cp-tag-picked" title="You have picked this club for this gameweek">picked</span>' : ps.planned.has(c.id) ? ' <span class="cp-tag cp-tag-plan" title="In your season plan for this gameweek">plan pick</span>' : ''}${w.locked ? ' <span class="cp-tag" title="This club has already played this gameweek, so it is locked">locked</span>' : ''}${w.games > 1 ? ' <span class="cp-tag cp-tag-double">double</span>' : ''}</td>
      <td class="cp-muted">${esc(c.league)}</td>
      <td>${fixturesHtml(c, w)}</td>
      <td class="cp-xp" data-tip="xptotal" data-club="${c.id}" data-gw="${gw}">${fmt(w.xp)}</td>
      <td class="cp-center">${leftBadge(c.id)}</td>
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
    <p class="cp-note">${esc(quality)} Picks left are for ${esc(state.store.active)}.</p>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th title="Rank among all 72 clubs this gameweek">#</th><th>Club</th><th>League</th><th>Fixtures (xP each)</th><th class="cp-right">xP</th>
        <th class="cp-center" title="How many of the club's 5 picks ${esc(state.store.active)} still has">Picks left</th>
        <th class="cp-center" title="Is this one of the club's 5 best weeks of the season?">Top-5 week</th><th class="cp-center" title="The club's best week of the season">Best week</th></tr></thead>
      <tbody>${body || '<tr><td colspan="8" class="cp-muted">No clubs match.</td></tr>'}</tbody>
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
  computeResult();
  const { plan } = state;
  const scrollY = window.scrollY;
  const asOf = new Date(plan.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  root.innerHTML = renderControls() + renderPanel() + renderBest() + renderGlance() +
    `<p class="cp-foot">${esc(plan.notes.horizon)} Updated ${esc(asOf)}. Weeks with no fixture score 0, so do not pick a club that blanks.
    <span class="cp-legend"><span class="cp-dot cp-dot-market"></span> bookmaker odds <span class="cp-dot cp-dot-model"></span> model estimate</span></p>
    <div id="cp-modal" class="games-modal" style="display:none"><div class="games-modal-content cp-modal-content" id="cp-modal-body"></div></div>`;
  tip.style.display = 'none';
  bind();
  window.scrollTo(0, scrollY);
}

function bind() {
  document.getElementById('cp-gw').addEventListener('change', e => { state.gw = +e.target.value; state.showAll = false; render(); });
  const step = d => {
    const i = state.plan.gameweeks.findIndex(g => g.gw === state.gw) + d;
    if (i >= 0 && i < state.plan.gameweeks.length) { state.gw = state.plan.gameweeks[i].gw; state.showAll = false; render(); }
  };
  document.getElementById('cp-prev').addEventListener('click', () => step(-1));
  document.getElementById('cp-next').addEventListener('click', () => step(1));
  document.querySelectorAll('.cp-chip[data-league]').forEach(b => b.addEventListener('click', () => { state.league = b.dataset.league; state.showAll = false; render(); }));
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
    e.preventDefault(); state.gw = +a.dataset.gw; state.showAll = false; render();
    document.querySelector('.cp-h2').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }));
  const modal = document.getElementById('cp-modal');
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

  document.getElementById('cp-panel').addEventListener('toggle', e => { state.panelOpen = e.target.open; });
  const pd = document.getElementById('cp-plan-details');
  if (pd) pd.addEventListener('toggle', e => { state.planOpen = e.target.open; });
  document.querySelectorAll('.cp-chip[data-profile]').forEach(b => b.addEventListener('click', () => {
    state.store.active = b.dataset.profile; saveStore(); render();
  }));
  document.querySelectorAll('.cp-pick').forEach(sel => sel.addEventListener('change', () => {
    const gw = sel.dataset.gw, slot = +sel.dataset.slot;
    const picks = activePicks();
    const cur = (picks[gw] || [null, null]).slice();
    cur[slot] = sel.value ? +sel.value : null;
    if (!cur[0] && !cur[1]) delete picks[gw]; else picks[gw] = cur;
    saveStore(); render();
  }));
  document.querySelectorAll('[data-add]').forEach(b => b.addEventListener('click', () => addPick(b.dataset.gw, +b.dataset.add)));
  document.querySelectorAll('[data-add-both]').forEach(b => b.addEventListener('click', () => {
    b.dataset.addBoth.split(',').forEach(id => addPick(b.dataset.gw, +id));
  }));
  document.querySelectorAll('[data-remove]').forEach(b => b.addEventListener('click', () => removePick(b.dataset.gw, +b.dataset.remove)));
  document.getElementById('cp-cloud-pass').addEventListener('input', e => { state.cloud.pass = e.target.value; });
  document.getElementById('cp-cloud-save').addEventListener('click', () => cloudCall('save'));
  document.getElementById('cp-cloud-load').addEventListener('click', () => cloudCall('load'));
  document.getElementById('cp-clear').addEventListener('click', () => {
    if (confirm(`Clear all picks entered for ${state.store.active}?`)) { state.store.picks[state.store.active] = {}; saveStore(); render(); }
  });
  document.getElementById('cp-export').addEventListener('click', () => {
    window.prompt('Copy this to back up or move your picks:', JSON.stringify(state.store));
  });
  document.getElementById('cp-import').addEventListener('click', () => {
    const raw = window.prompt('Paste exported picks here (this replaces what is saved for both profiles):');
    if (!raw) return;
    try {
      const s = JSON.parse(raw);
      if (!s || !s.picks) throw new Error('bad format');
      for (const p of PROFILES) state.store.picks[p] = sanitizePicks(s.picks[p]).picks;
      if (PROFILES.includes(s.active)) state.store.active = s.active;
      saveStore(); render();
    } catch (e) { alert('Could not read that. Paste the text exactly as exported.'); }
  });
}

function openClub(id) {
  const c = state.byId[id];
  if (!c) return;
  const best = c.top5.map(gw => weekOf(c, gw)).sort((a, b) => b.xp - a.xp);
  const rows = best.map(w => {
    const meta = gwMeta(w.gw);
    const firstDate = w.fx.length ? w.fx[0].date : meta.start;
    return `<tr>
      <td>${rankBadge(w.rank)}</td>
      <td><strong>GW ${w.gw}</strong><div class="cp-muted cp-small">${fmtDate(firstDate)}${w.games > 1 ? ' · double' : ''}</div></td>
      <td>${fixturesHtml(c, w)}</td>
      <td class="cp-xp" data-tip="xptotal" data-club="${c.id}" data-gw="${w.gw}">${fmt(w.xp)}</td>
    </tr>`;
  }).join('');
  const maxXp = Math.max(...c.weeks.map(w => w.xp), 1);
  const top5 = new Set(c.top5);
  const bars = c.weeks.map(w => {
    const h = w.games ? Math.max(4, (w.xp / maxXp) * 100) : 2;
    const cls = !w.games ? 'cp-bar-blank' : top5.has(w.gw) ? 'cp-bar-top' : '';
    const tipText = w.games ? `GW ${w.gw}: ${fmt(w.xp)} xP, ranked ${w.rank} of 72${w.games > 1 ? ' (double)' : ''}` : `GW ${w.gw}: no fixture`;
    return `<div class="cp-bar-col" title="${esc(tipText)}"><div class="cp-bar ${cls}" style="height:calc((100% - 16px) * ${(h / 100).toFixed(3)})"></div><div class="cp-bar-label">${w.gw}</div></div>`;
  }).join('');
  const body = document.getElementById('cp-modal-body');
  body.innerHTML = `
    <span class="games-modal-close" id="cp-close">&times;</span>
    <h3 class="cp-modal-title">${esc(c.name)} <span class="cp-sub">${esc(c.league)}</span></h3>
    <p class="cp-note">Best 5 weeks to use ${esc(c.short)} (up to ${MAX_USES} picks a season): <strong>${fmt(c.top5Total)} xP</strong> in total.
      ${esc(state.store.active)} has <strong>${Math.max(0, picksLeft(c.id))} of ${MAX_USES}</strong> picks left. The badge shows how this club ranks among all 72 clubs that week; <strong>1 = the best club that week</strong>.</p>
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
      state.clubWeek[c.id] = Object.fromEntries(c.weeks.map(w => [w.gw, w]));
    }
    loadStore();
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
