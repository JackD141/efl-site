// League Rivals page: the top managers of a Fantasy EFL league, rated by what their picks were expected to score
// (ml/team_strengths/league_rivals.py -> data/league_rivals.json).
const statusEl = document.getElementById('status');
const root = document.getElementById('rv-root');
const state = { data: null, clubs: {}, league: 0, sort: 'ev', open: new Set(), openGw: new Set() };
const POS_ORDER = { GK: 0, DEF: 1, MID: 2, FWD: 3 };

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const f1 = n => n.toFixed(1);
const f0 = n => Math.round(n).toLocaleString('en-GB');
const signed = (n, d = 0) => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(d)}`;
const ordinal = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');
const luckCls = n => (n > 0.5 ? 'rv-up' : n < -0.5 ? 'rv-down' : 'cp-muted');

const SORTS = {
  ev: { label: 'Expected', get: m => m.ev },
  act: { label: 'Actual', get: m => m.act },
  luck: { label: 'Luck', get: m => m.act - m.ev },
  evP: { label: 'Players', get: m => m.evP },
  evC: { label: 'Captain', get: m => m.evC },
  evS: { label: 'Clubs', get: m => m.evS },
};

// one gameweek of one manager: every pick with expected and real points
function weekHtml(w) {
  const d = state.data, pg = d.pg[w.gw] || {}, cg = d.cg[w.gw] || {};
  if (w.none) return '<p class="cp-muted cp-small">No team this gameweek.</p>';
  const ids = [...w.p].sort((a, b) => (POS_ORDER[(d.players[a] || [])[1]] ?? 9) - (POS_ORDER[(d.players[b] || [])[1]] ?? 9));
  const rows = ids.map(i => {
    const [name, pos, club] = d.players[i] || [String(i), '', null];
    const [ev, pts, mins] = pg[i] || [0, 0, 0];
    const doubled = w.dbl === i;
    const badge = w.c === i ? ' <span class="op-badge op-cap op-inline" title="Captain">C</span>' : w.v === i ? ' <span class="op-badge op-vice op-inline" title="Vice-captain">V</span>' : '';
    return `<tr><td><span class="position-badge pos-${pos}">${pos}</span></td>
      <td class="pp-club">${shirtIcon(state.clubs[club], pos === 'GK')}<div><strong>${esc(name)}</strong>${badge}<div class="cp-muted cp-small">${esc((state.clubs[club] || {}).name || '')}</div></div></td>
      <td class="cp-right">${mins > 0 ? f1(ev) : '<span class="cp-muted">0</span>'}${doubled ? ' <span class="cp-muted cp-small">(×2)</span>' : ''}</td>
      <td class="cp-right"><strong>${pts}</strong>${doubled ? ' <span class="cp-muted cp-small">(×2)</span>' : ''}</td>
      <td class="cp-right cp-muted">${mins > 0 ? mins : 'did not play'}</td></tr>`;
  }).join('');
  const clubs = w.s.map(c => {
    const [ev, pts] = cg[c] || [0, 0];
    return `<tr><td><span class="position-badge">CLUB</span></td><td class="pp-club">${shirtIcon(state.clubs[c], false)}<div><strong>${esc(d.clubs[c] || c)}</strong></div></td>
      <td class="cp-right">${f1(ev)}</td><td class="cp-right"><strong>${pts}</strong></td><td></td></tr>`;
  }).join('');
  const mc = w.mc ? `<tr><td></td><td><strong>Max Captain</strong> <span class="cp-muted cp-small">top scorer of the seven counts twice</span></td>
      <td class="cp-right">${f1(w.evC)}</td><td class="cp-right"><strong>${w.actC}</strong></td><td></td></tr>` : '';
  return `<table class="cp-table bt-team"><thead><tr><th></th><th>Pick</th><th class="cp-right">Expected</th><th class="cp-right">Points</th><th class="cp-right">Mins</th></tr></thead>
    <tbody>${rows}${mc}${clubs}</tbody></table>`;
}

function managerDetail(m) {
  const rows = m.weeks.map(w => {
    const key = `${m.userId}-${w.gw}`, open = state.openGw.has(key), luck = w.act - w.ev;
    const chips = [w.mc ? '<span class="cp-tag cp-tag-double">Max Captain</span>' : '', w.oc ? '<span class="cp-tag">One Club</span>' : ''].join('');
    return `<tr><td><button class="cp-link-btn rv-gw" data-key="${key}">${open ? '▾' : '▸'} GW ${w.gw}</button>${chips}</td>
      <td class="cp-right"><strong>${w.act}</strong></td><td class="cp-right">${f1(w.ev)}</td><td class="cp-right ${luckCls(luck)}">${signed(luck)}</td>
      <td class="cp-right cp-muted">${w.none ? '' : `${f1(w.evP)} / ${w.actP}`}</td><td class="cp-right cp-muted">${w.none ? '' : `${f1(w.evC)} / ${w.actC}`}</td>
      <td class="cp-right cp-muted">${w.none ? '' : `${f1(w.evS)} / ${w.actS}`}</td></tr>
      ${open ? `<tr><td colspan="7">${weekHtml(w)}</td></tr>` : ''}`;
  }).join('');
  return `<div class="rv-detail"><table class="cp-table cp-table-tight">
    <thead><tr><th>Gameweek</th><th class="cp-right">Actual</th><th class="cp-right">Expected</th><th class="cp-right">Luck</th>
      <th class="cp-right">Players exp / act</th><th class="cp-right">Captain exp / act</th><th class="cp-right">Clubs exp / act</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function render() {
  const d = state.data, lg = d.leagues[state.league];
  const get = SORTS[state.sort].get;
  const byEv = [...lg.managers].sort((a, b) => b.ev - a.ev);
  const evRank = new Map(byEv.map((m, i) => [m.userId, i + 1]));
  const list = [...lg.managers].sort((a, b) => get(b) - get(a));
  const best = byEv[0], you = lg.managers.find(m => m.userId === lg.you);
  const lucks = lg.managers.map(m => m.act - m.ev);
  const meanLuck = lucks.reduce((a, b) => a + b, 0) / lucks.length;
  const th = (key, extra = '') => `<th class="cp-right rv-sort${state.sort === key ? ' rv-sorted' : ''}" data-sort="${key}" title="Sort by this column">${SORTS[key].label}${extra}${state.sort === key ? ' ▾' : ''}</th>`;
  const rows = list.map(m => {
    const open = state.open.has(m.userId), luck = m.act - m.ev, isYou = m.userId === lg.you;
    const miss = m.eflDone != null && Math.abs(m.eflDone - m.act) > 0.5 ? Math.round(m.eflDone - m.act) : 0;
    return `<tr class="${isYou ? 'cp-current' : ''}">
      <td class="cp-center"><span class="cp-rank${evRank.get(m.userId) <= 3 ? ' cp-rank-' + evRank.get(m.userId) : ''}">${evRank.get(m.userId)}</span></td>
      <td><button class="cp-link-btn rv-toggle" data-id="${m.userId}">${open ? '▾' : '▸'} ${esc(m.team)}</button>${isYou ? '<span class="cp-tag cp-tag-plan">You</span>' : ''}
        <div class="cp-muted cp-small rv-name">${esc(m.name)}</div></td>
      <td class="cp-right cp-muted">${m.rank}</td>
      <td class="cp-right">${f0(m.eflDone ?? m.act)}${miss ? `<span class="rv-miss" title="${miss} of these points are from a pick whose games are not in our stats (he has since moved club), so they are left out of Expected and Luck">†</span>` : ''}</td>
      <td class="cp-xp">${f0(m.ev)}</td>
      <td class="cp-right ${luckCls(luck)}"><strong>${signed(luck)}</strong></td>
      <td class="cp-right">${f0(m.evP)}</td><td class="cp-right">${f0(m.evC)}</td><td class="cp-right">${f0(m.evS)}</td></tr>
      ${open ? `<tr class="rv-detail-row"><td colspan="9">${managerDetail(m)}</td></tr>` : ''}`;
  }).join('');
  const gap = you && best.userId !== you.userId ? ` You are ${ordinal(evRank.get(you.userId))} on expected points, ${f0(best.ev - you.ev)} behind.` : you ? ' That is you.' : '';
  root.innerHTML = `
    <p class="cp-intro">Who in the league is picking well, and who has been lucky. For every manager in the top 20 and every finished gameweek, each pick is
      scored with what our model <strong>expected</strong> it to score given the minutes the player really played, next to what it did score.
      Expected points reward good picks; the difference to the real points is mostly luck (goals going in or not).</p>
    <div class="cp-controls"><div class="cp-chips">${d.leagues.map((l, i) => `<button class="cp-chip${i === state.league ? ' active' : ''}" data-league="${i}">${esc(l.name)}</button>`).join('')}</div>
      <span class="cp-muted cp-small">GW ${d.gws[0]}–${d.gws[d.gws.length - 1]}${lg.entrants ? `, top ${lg.managers.length} of ${lg.entrants.toLocaleString('en-GB')} managers` : ''}</span></div>
    <p class="cp-note"><strong>${esc(best.team)}</strong> (${esc(best.name)}) has the best picks on expected points: ${f0(best.ev)}.${gap}
      On average these ${lg.managers.length} managers scored ${signed(meanLuck)} against expected: being in the top 20 takes some luck.</p>
    <div class="cp-table-wrap"><table class="cp-table rv-table">
      <thead><tr><th class="cp-center" title="Rank by expected points">#</th><th>Team <span class="rv-th-sub">click for gameweeks</span></th>
        <th class="cp-right" title="Rank in the league on real points">League rank</th>${th('act')}${th('ev')}${th('luck')}
        ${th('evP', ' <span class="rv-th-sub">exp</span>')}${th('evC', ' <span class="rv-th-sub">exp</span>')}${th('evS', ' <span class="rv-th-sub">exp</span>')}</tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      <li><strong>Expected</strong> is the model's expected points for the picks made, as it stood before each gameweek (models refitted on earlier games only,
        closing odds for the fixtures). Players are scored on the minutes they really played: a pick who did not play is 0, so choosing players who start
        counts as skill here.</li>
      <li><strong>Luck</strong> = actual − expected. Over ${d.gws.length} gameweeks it is large: a few goals from a captain move it by 20 or more, so small gaps in
        expected points between two managers are not meaningful.</li>
      <li><strong>Captain</strong> is the captain's expected points counted again (the vice-captain's if the captain did not play). In a Max Captain week it is the
        expected top score of the seven (simulated: ${d.maxCaptain.sim.toFixed(1)} on average across all teams here against ${d.maxCaptain.real.toFixed(1)} real).</li>
      <li><strong>Clubs</strong> are the two club picks, expected points from the closing odds.</li>
      <li>Only the current top 20 on real points are listed, so a manager with good picks and bad luck further down is not shown.</li>
      <li>Names and teams are as the EFL site shows them in the league table.</li>
    </ul></div>
    <p class="cp-foot">League table and teams fetched ${esc(new Date(lg.fetchedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }))}.
      Scored ${esc(new Date(d.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }))} with <code>league_rivals.py</code>.
      The real totals are rebuilt from the picks and checked against the EFL's totals${lg.managers.some(m => m.eflDone != null && Math.abs(m.eflDone - m.act) > 0.5)
        ? ' (†: a few points are from a pick whose games are not in our stats because he has since moved club; they are left out of Expected and Luck)' : ''}.</p>`;
  root.querySelectorAll('[data-league]').forEach(b => b.addEventListener('click', () => { state.league = +b.dataset.league; state.open.clear(); state.openGw.clear(); render(); }));
  root.querySelectorAll('.rv-sort').forEach(h => h.addEventListener('click', () => { state.sort = h.dataset.sort; render(); }));
  root.querySelectorAll('.rv-toggle').forEach(b => b.addEventListener('click', () => {
    const id = +b.dataset.id;
    if (state.open.has(id)) state.open.delete(id); else state.open.add(id);
    render();
  }));
  root.querySelectorAll('.rv-gw').forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.key;
    if (state.openGw.has(k)) state.openGw.delete(k); else state.openGw.add(k);
    render();
  }));
}

async function load() {
  statusEl.textContent = 'Loading...';
  try {
    const [rv, cl] = await Promise.all(['league_rivals.json', 'club_plan.json'].map(async f => {
      const r = await fetch('data/' + f, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`Could not load ${f} (${r.status})`);
      return r.json();
    }));
    state.data = rv;
    for (const c of cl.clubs) state.clubs[c.id] = c;
    statusEl.textContent = '';
    render();
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

load();
