// Backtest page: how the model's team would have scored each gameweek of 2026/27 (ml/team_strengths/backtest.py).
const statusEl = document.getElementById('status');
const root = document.getElementById('bt-root');
const COLORS = { model: '#f05a28', form: '#1f3d7a' };
const state = { data: null, clubs: {}, open: new Set() };

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmt = n => (Math.round(n * 10) / 10).toString();

function chart(weeks) {
  const W = 760, H = 260, L = 44, R = 70, T = 16, B = 34;
  let cm = 0, cf = 0;
  const pts = weeks.map(w => ({ gw: w.gw, model: (cm += w.model.total), form: (cf += w.form.total) }));
  const ymax = Math.max(10, ...pts.map(p => Math.max(p.model, p.form))) * 1.05;
  const x = i => L + (weeks.length === 1 ? 0 : i * (W - L - R) / (weeks.length - 1));
  const y = v => T + (H - T - B) * (1 - v / ymax);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => Math.round(ymax * f / 50) * 50).filter((v, i, a) => a.indexOf(v) === i && v <= ymax);
  const line = key => pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
  const last = pts[pts.length - 1];
  return `<div class="bt-chart-wrap"><svg class="bt-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Cumulative points by gameweek: model ${last.model}, form picker ${last.form}">
    ${ticks.map(v => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="bt-grid"/><text x="${L - 6}" y="${y(v) + 4}" class="bt-axis" text-anchor="end">${v}</text>`).join('')}
    ${pts.map((p, i) => `<text x="${x(i)}" y="${H - B + 18}" class="bt-axis" text-anchor="middle">GW${p.gw}</text>`).join('')}
    <path d="${line('form')}" fill="none" stroke="${COLORS.form}" stroke-width="2"/>
    <path d="${line('model')}" fill="none" stroke="${COLORS.model}" stroke-width="2"/>
    ${pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.form)}" r="4" fill="${COLORS.form}" stroke="#fff" stroke-width="2"/><circle cx="${x(i)}" cy="${y(p.model)}" r="4" fill="${COLORS.model}" stroke="#fff" stroke-width="2"/>`).join('')}
    <text x="${x(pts.length - 1) + 8}" y="${y(last.model) + 4}" class="bt-label">Model ${last.model}</text>
    <text x="${x(pts.length - 1) + 8}" y="${y(last.form) + 4}" class="bt-label">Form ${last.form}</text>
    ${pts.map((p, i) => `<rect x="${x(i) - 20}" y="${T}" width="40" height="${H - T - B}" fill="transparent" class="bt-hit" data-i="${i}"/>`).join('')}
    <line class="bt-cross" x1="0" x2="0" y1="${T}" y2="${H - B}" style="display:none"/>
  </svg><div class="bt-tip" style="display:none"></div></div>
  <div class="bt-legend"><span><i style="background:${COLORS.model}"></i>Model's team</span><span><i style="background:${COLORS.form}"></i>Form picker (last-5 average points)</span></div>`;
}

function teamHtml(w) {
  const rows = w.team.map(p => `<tr><td><span class="position-badge pos-${p.pos}">${p.pos}</span></td>
      <td class="pp-club">${shirtIcon(state.clubs[p.club], p.pos === 'GK')}<div><strong>${esc(p.name)}</strong>${p.captain ? ' <span class="op-badge op-cap op-inline" title="Captain">C</span>' : ''}<div class="cp-muted cp-small">${esc(p.clubName)}</div></div></td>
      <td class="cp-right">${p.xp.toFixed(1)}</td><td class="cp-right"><strong>${p.points}</strong>${p.captain && p.mins > 0 ? ` <span class="cp-muted cp-small">(×2)</span>` : ''}</td><td class="cp-right cp-muted">${p.mins}</td></tr>`).join('');
  const clubs = w.clubs.map(c => `<tr><td><span class="position-badge">CLUB</span></td><td class="pp-club">${shirtIcon(state.clubs[c.id], false)}<div><strong>${esc(c.name)}</strong></div></td>
      <td class="cp-right">${c.xp.toFixed(1)}</td><td class="cp-right"><strong>${c.points}</strong></td><td></td></tr>`).join('');
  return `<table class="cp-table bt-team"><thead><tr><th></th><th>Pick</th><th class="cp-right">xP</th><th class="cp-right">Points</th><th class="cp-right">Mins</th></tr></thead>
    <tbody>${rows}${clubs}</tbody></table>`;
}

function render() {
  const d = state.data, t = d.totals, weeks = d.weeks;
  const n = weeks.length;
  const rows = weeks.map(w => `<tr class="bt-row" data-gw="${w.gw}">
      <td><button class="cp-link-btn bt-toggle" data-gw="${w.gw}">${state.open.has(w.gw) ? '▾' : '▸'} GW ${w.gw}</button></td>
      <td class="cp-right"><strong>${w.model.total}</strong> <span class="cp-muted cp-small">(${w.model.players} + clubs ${w.model.clubs})</span></td>
      <td class="cp-right cp-muted">${fmt(w.model.xp)}</td>
      <td class="cp-right">${w.form.total}</td>
      <td class="cp-right cp-muted">${w.hindsight.total}</td></tr>
      ${state.open.has(w.gw) ? `<tr><td colspan="5">${teamHtml(w)}</td></tr>` : ''}`).join('');
  root.innerHTML = `
    <p class="cp-intro">How the model would have done this season if it had picked every week. For each gameweek the models were retrained
      on games <strong>before</strong> it only, then the site's rules picked the best team (formation, at most 2 per club, captain) and the two best clubs
      (each club used at most 5 times), scored with the real points.</p>
    <div class="bt-tiles">
      <div class="bt-tile bt-tile-main"><div class="bt-num">${t.model}</div><div class="bt-lab">Model's team, ${n} gameweeks</div><div class="bt-sub">${fmt(t.model / n)} a week</div></div>
      <div class="bt-tile"><div class="bt-num">${t.form}</div><div class="bt-lab">Form picker</div><div class="bt-sub">same rules and clubs, picks by last-5 average points</div></div>
      <div class="bt-tile"><div class="bt-num">${fmt(t.xp)}</div><div class="bt-lab">Model's expected points</div><div class="bt-sub">${t.model < t.xp ? `${Math.round((1 - t.model / t.xp) * 100)}% below: picking the top projections favours ones that came out too high` : 'at or below the real total'}</div></div>
      <div class="bt-tile"><div class="bt-num">${t.hindsight}</div><div class="bt-lab">Best possible in hindsight</div><div class="bt-sub">the ceiling nobody reaches</div></div>
    </div>
    <h2 class="cp-h2">Cumulative points</h2>
    ${chart(weeks)}
    <h2 class="cp-h2">Week by week <span class="cp-sub">click a gameweek to see the team</span></h2>
    <div class="cp-table-wrap"><table class="cp-table">
      <thead><tr><th>Gameweek</th><th class="cp-right">Model's team</th><th class="cp-right">Model expected</th><th class="cp-right">Form picker</th><th class="cp-right">Hindsight best</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <div class="cp-caveat"><strong>How to read this</strong><ul>
      <li>Everything the model used was available before each gameweek: closing odds (the market's pre-match view; the few games without odds in our
        data use the team-strength model fitted on earlier games), player data up to the previous gameweek, and FotMob history before the first kick-off.</li>
      <li>Harder than live: no injury or team news, so expected minutes come only from the minutes model.</li>
      <li>Small leaks: the keeper saves model and the league step-up multipliers were fitted with some of this season's data; the choice of model features was made on last season only.</li>
      <li>Eight gameweeks is a small sample: single weeks are mostly luck, so judge the totals, not one week.</li>
    </ul></div>
    <p class="cp-foot">Run ${esc(new Date(d.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }))} with <code>backtest.py</code>.</p>`;
  bind(weeks);
}

function bind(weeks) {
  document.querySelectorAll('.bt-toggle').forEach(b => b.addEventListener('click', () => {
    const gw = +b.dataset.gw;
    if (state.open.has(gw)) state.open.delete(gw); else state.open.add(gw);
    render();
  }));
  const svg = document.querySelector('.bt-chart'), tip = document.querySelector('.bt-tip'), cross = svg.querySelector('.bt-cross');
  let cm = 0, cf = 0;
  const cum = weeks.map(w => ({ gw: w.gw, w, model: (cm += w.model.total), form: (cf += w.form.total) }));
  svg.querySelectorAll('.bt-hit').forEach(r => {
    r.addEventListener('mouseenter', () => {
      const c = cum[+r.dataset.i], cx = +r.getAttribute('x') + 20;
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.display = '';
      tip.innerHTML = `<strong>GW ${c.gw}</strong><div><i style="background:${COLORS.model}"></i>Model ${c.w.model.total} (total ${c.model})</div>
        <div><i style="background:${COLORS.form}"></i>Form ${c.w.form.total} (total ${c.form})</div><div class="cp-muted">Hindsight best ${c.w.hindsight.total}</div>`;
      tip.style.display = 'block';
      const box = svg.getBoundingClientRect();
      tip.style.left = Math.min(box.width - 190, Math.max(0, cx / 760 * box.width + 10)) + 'px';
    });
    r.addEventListener('mouseleave', () => { tip.style.display = 'none'; cross.style.display = 'none'; });
  });
}

async function load() {
  statusEl.textContent = 'Loading...';
  try {
    const [bt, cl] = await Promise.all(['backtest.json', 'club_plan.json'].map(async f => {
      const r = await fetch('data/' + f, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`Could not load ${f} (${r.status})`);
      return r.json();
    }));
    state.data = bt;
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
