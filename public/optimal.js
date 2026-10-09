// Optimal Picks (homepage): best 7 players + 2 clubs + captain for a gameweek, under the Fantasy EFL rules.
// xP comes from the same data and maths as the Keeper / Defender / Midfielder / Forward / Club pages; minutes edited on
// those pages (saved in this browser) are used here too. Games that have already kicked off are not excluded.
const statusEl = document.getElementById('status');
const root = document.getElementById('op-root');

const FORMATIONS = [
  { name: '1-2-2-2', GK: 1, DEF: 2, MID: 2, FWD: 2 },
  { name: '1-2-3-1', GK: 1, DEF: 2, MID: 3, FWD: 1 },
  { name: '1-3-2-1', GK: 1, DEF: 3, MID: 2, FWD: 1 },
];
const POSITIONS = ['GK', 'DEF', 'MID', 'FWD'];
const POS_NAME = { GK: 'Goalkeeper', DEF: 'Defender', MID: 'Midfielder', FWD: 'Forward' };
const POS_PAGE = { GK: 'picks.html?pos=GK', DEF: 'picks.html?pos=DEF', MID: 'picks.html?pos=MID', FWD: 'picks.html?pos=FWD' };
const MAX_PER_CLUB = 2;
const MAX_USES = 5;
const PICKS_PER_WEEK = 2;
const OWN_STORE = 'efl_optimal_v1';
const MINS_KEYS = { GK: 'efl_keeper_mins_v1', DEF: 'efl_defender_mins_v1', MID: 'efl_mid_mins_v1', FWD: 'efl_fwd_mins_v1' };
const POOL_PER_POS = 40;

const state = {
  plans: {}, clubs: {}, shortByName: {}, week: { GK: {}, DEF: {}, MID: {}, FWD: {}, CLUB: {} },
  mins: {}, players: [], gw: null, abbrByName: {}, kitById: {},
  opts: { pinned: [], excluded: [], oneClub: false }, result: null,
};

/* ---------- helpers ---------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmt = n => n.toFixed(1);
const fmt2 = n => n.toFixed(2);
function fmtDate(iso, withDay) {
  const d = new Date(iso + 'T12:00:00');
  return d.toLocaleDateString('en-GB', withDay ? { weekday: 'short', day: 'numeric', month: 'short' } : { day: 'numeric', month: 'short' });
}
const shortOf = name => state.shortByName[name] || name;
function abbr(name) {
  if (state.abbrByName[name]) return state.abbrByName[name];
  const w = shortOf(name).replace(/[^A-Za-z ]/g, '').split(' ').filter(Boolean);
  return (w.length === 1 ? w[0].slice(0, 3) : w[0].slice(0, 2) + w[1][0]).toUpperCase();
}

/* ---------- kits: club colours from the EFL squad data, plus patterns / fixes for well-known kits (approximate) ---------- */
const KITS = {
  LIN: ['#e10613', '#ffffff', 'stripes'], SHU: ['#ed1c24', '#ffffff', 'stripes'], STO: ['#d7172f', '#ffffff', 'stripes'],
  SOU: ['#e3051b', '#ffffff', 'stripes'], EXE: ['#e1211c', '#ffffff', 'stripes'], SHW: ['#0971ce', '#ffffff', 'stripes'],
  HUD: ['#0971ce', '#ffffff', 'stripes'], WIG: ['#00539e', '#ffffff', 'stripes'], COL: ['#005eb8', '#ffffff', 'stripes'],
  GRI: ['#111111', '#ffffff', 'stripes'], NOT: ['#111111', '#ffffff', 'stripes'], BRA: ['#72253d', '#f2b51c', 'stripes'],
  WBA: ['#122f67', '#ffffff', 'stripes'], CLT: ['#e1231b', '#ffffff', 'stripes'],
  BLA: ['#014898', '#ffffff', 'halves'], BRR: ['#1a51a0', '#ffffff', 'quarters'], WYC: ['#55b1e2', '#0b1f4b', 'quarters'],
  QPR: ['#0054a2', '#ffffff', 'hoops'], REA: ['#0133a0', '#ffffff', 'hoops'], DON: ['#e2211c', '#ffffff', 'hoops'],
  BUR: ['#6c1d45', '#99d6ea', 'sleeves'], WHU: ['#7a263a', '#1bb1e7', 'sleeves'], FLE: ['#e1211c', '#ffffff', 'sleeves'],
  ROT: ['#e1211c', '#ffffff', 'sleeves'], WAT: ['#fbee23', '#111111', 'sleeves'], NOR: ['#fff200', '#00a650', 'sleeves'],
  WOL: ['#fdb913', '#231f20', 'plain'], PNE: ['#ffffff', '#0e1d49', 'plain'], BOL: ['#ffffff', '#06205c', 'plain'],
  TRA: ['#ffffff', '#001489', 'plain'], PVL: ['#ffffff', '#111111', 'plain'], MKD: ['#ffffff', '#e30613', 'plain'],
  DER: ['#ffffff', '#111111', 'plain'], SWA: ['#ffffff', '#111111', 'plain'], BRO: ['#ffffff', '#111111', 'plain'],
};
function kitOf(clubId) {
  const c = state.kitById[clubId] || {};
  const k = KITS[c.abbr];
  return k ? { a: k[0], b: k[1], pattern: k[2] } : { a: c.color || '#1f3d7a', b: c.textColor || '#ffffff', pattern: 'plain' };
}
const SHIRT_PATH = 'M21 3 L7 11 L2 26 L13 29 L13 57 L51 57 L51 29 L62 26 L57 11 L43 3 Q32 12 21 3 Z';
let shirtSeq = 0;
function shirtSvg(clubId, keeper) {
  let { a, b, pattern } = kitOf(clubId);
  if (keeper) { b = a; a = '#c6e33a'; pattern = 'sleeves'; } // keepers: a generic keeper kit with club-colour sleeves
  const id = 'sh' + (++shirtSeq);
  const over = {
    plain: '',
    stripes: [16, 28, 40].map(x => `<rect x="${x}" y="0" width="7" height="60" fill="${b}"/>`).join(''),
    hoops: [14, 28, 42].map(y => `<rect x="0" y="${y}" width="64" height="7" fill="${b}"/>`).join(''),
    halves: `<rect x="32" y="0" width="32" height="60" fill="${b}"/>`,
    quarters: `<rect x="32" y="0" width="32" height="30" fill="${b}"/><rect x="0" y="30" width="32" height="30" fill="${b}"/>`,
    sleeves: `<path d="M7 11 L2 26 L13 29 L16 14 Z M57 11 L62 26 L51 29 L48 14 Z" fill="${b}"/>`,
  }[pattern] || '';
  return `<svg viewBox="0 0 64 60" aria-hidden="true"><defs><clipPath id="${id}"><path d="${SHIRT_PATH}"/></clipPath></defs>
    <g clip-path="url(#${id})"><rect x="0" y="0" width="64" height="60" fill="${a}"/>${over}</g>
    <path d="${SHIRT_PATH}" fill="none" stroke="rgba(0,0,0,0.35)" stroke-width="1.5"/></svg>`;
}
function readJson(key, fallback) {
  try { const v = JSON.parse(localStorage.getItem(key) || 'null'); return v && typeof v === 'object' ? v : fallback; } catch (e) { return fallback; }
}
function saveOwn() {
  try { localStorage.setItem(OWN_STORE, JSON.stringify(state.opts)); } catch (e) { /* ignore */ }
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
function attackerFixtureXp(pos, d, f, mins) {
  const p = state.plans[pos], sc = p.scoring, club = state.clubs[pos][d.club], opp = state.clubs[pos][f.oppId], t = mins / 90;
  if (mins <= 0) return 0;
  const mu = {};
  for (const stat of Object.keys(p.model)) {
    const vals = { role: d.role[stat], team: club.style[stat].team, opp: opp ? opp.style[stat].opp : club.style[stat].opp, lam_own: f.lamOwn, lam_opp: f.lamOpp, home: f.home };
    mu[stat] = glm(p.model[stat], vals, 1e-4) * t;
  }
  const g = nb(mu.goals, p.model.goals.r, 1, [3]);
  const r = d.rates;
  return appPoints(p, mins) + sc.goal * mu.goals + sc.hatTrick * g.atLeast[3] + sc.assist * mu.assists
    + sc.sot * mu.sot + nb(mu.kp, p.model.kp.r, sc.keyPassPer, [2]).pts + (p.model.int ? sc.interception * mu.int : 0)
    + t * (sc.yellow * r.y90 + sc.red * r.r90 + sc.penMiss * r.pm90 + sc.ownGoal * r.og90);
}
// [{f, xp}] for a player's fixtures in a gameweek, at his expected minutes
function playerWeek(pl, gw) {
  const wk = state.week[pl.pos][pl.club] && state.week[pl.pos][pl.club][gw];
  if (!wk || !wk.games || pl.mins <= 0) return { wk, fx: (wk && wk.fx || []).map(f => ({ f, xp: 0 })) };
  const s = pl.mins / 90;
  let fx;
  if (pl.pos === 'GK') fx = wk.fx.map(f => ({ f, xp: s * f.xp }));
  else if (pl.pos === 'DEF') fx = wk.fx.map(f => ({ f, xp: s * defenderFixtureXp(pl.raw, f) }));
  else fx = wk.fx.map(f => ({ f, xp: attackerFixtureXp(pl.pos, pl.raw, f, pl.mins) }));
  return { wk, fx };
}

/* ---------- clubs: the two best for this gameweek ---------- */
const clubWeek = (id, gw) => state.week.CLUB[id] && state.week.CLUB[id][gw];
function chooseClubs(gw) {
  return state.plans.CLUB.clubs.map(c => ({ id: c.id, club: c, wk: clubWeek(c.id, gw) }))
    .filter(x => x.wk && x.wk.games > 0).sort((a, b) => b.wk.xp - a.wk.xp).slice(0, PICKS_PER_WEEK);
}

/* ---------- player optimiser: exact branch and bound, captain = highest xP (counts double) ---------- */
function optimise(gw) {
  const pinnedSet = new Set(state.opts.pinned), excluded = new Set(state.opts.excluded);
  const all = state.players.map(pl => {
    const w = playerWeek(pl, gw);
    return { ...pl, fx: w.fx, xp: w.fx.reduce((a, x) => a + x.xp, 0) };
  });
  const byId = Object.fromEntries(all.map(p => [p.id, p]));
  const pinned = state.opts.pinned.map(id => byId[id]).filter(Boolean);
  const limit = state.opts.oneClub ? 7 : MAX_PER_CLUB;
  const pool = [];
  for (const pos of POSITIONS) {
    pool.push(...all.filter(p => p.pos === pos && !pinnedSet.has(p.id) && !excluded.has(p.id) && p.mins > 0 && p.xp > 0)
      .sort((a, b) => b.xp - a.xp).slice(0, POOL_PER_POS));
  }
  pool.sort((a, b) => b.xp - a.xp);
  const n = pool.length;
  // cnt[pos][i] = candidates of that position before global index i; prefix[pos] = running xP sums in order
  const cnt = {}, prefix = {};
  for (const pos of POSITIONS) {
    cnt[pos] = new Array(n + 1).fill(0);
    prefix[pos] = [0];
    for (let i = 0; i < n; i++) {
      cnt[pos][i + 1] = cnt[pos][i] + (pool[i].pos === pos ? 1 : 0);
      if (pool[i].pos === pos) prefix[pos].push(prefix[pos][prefix[pos].length - 1] + pool[i].xp);
    }
  }
  const results = [];
  for (const F of FORMATIONS) {
    const need = { GK: F.GK, DEF: F.DEF, MID: F.MID, FWD: F.FWD };
    const clubCnt = {};
    let sum = 0, mx = 0, problem = null;
    for (const p of pinned) {
      need[p.pos] -= 1;
      clubCnt[p.club] = (clubCnt[p.club] || 0) + 1;
      sum += p.xp; mx = Math.max(mx, p.xp);
      if (need[p.pos] < 0) problem = `too many pinned ${POS_NAME[p.pos].toLowerCase()}s`;
      if (clubCnt[p.club] > limit) problem = `more than ${limit} pinned players from one club`;
    }
    if (problem) { results.push({ F, problem }); continue; }
    let best = { total: -Infinity, team: null };
    const chosen = [];
    let nodes = 0;
    const dfs = (i, s, m) => {
      if (++nodes > 3e6) return;
      if (need.GK + need.DEF + need.MID + need.FWD === 0) {
        if (s + m > best.total) best = { total: s + m, team: chosen.slice() };
        return;
      }
      if (i >= n) return;
      let bound = s;
      for (const pos of POSITIONS) {
        if (!need[pos]) continue;
        const k = cnt[pos][i];
        if (k + need[pos] > prefix[pos].length - 1) return;
        bound += prefix[pos][k + need[pos]] - prefix[pos][k];
      }
      if (bound + Math.max(m, pool[i].xp) <= best.total + 1e-9) return;
      const p = pool[i];
      if (need[p.pos] > 0 && (clubCnt[p.club] || 0) < limit) {
        need[p.pos] -= 1; clubCnt[p.club] = (clubCnt[p.club] || 0) + 1; chosen.push(p);
        dfs(i + 1, s + p.xp, Math.max(m, p.xp));
        chosen.pop(); clubCnt[p.club] -= 1; need[p.pos] += 1;
      }
      dfs(i + 1, s, m);
    };
    dfs(0, sum, mx);
    if (!best.team) { results.push({ F, problem: 'not enough available players' }); continue; }
    const team = [...pinned, ...best.team];
    const order = team.slice().sort((a, b) => b.xp - a.xp);
    results.push({ F, team, total: best.total, players: team.reduce((a, p) => a + p.xp, 0), captain: order[0], vice: order[1] });
  }
  const ok = results.filter(r => !r.problem).sort((a, b) => b.total - a.total);
  return { all, byId, results, best: ok[0] || null, pinned };
}

/* ---------- tooltip ---------- */
const tip = document.createElement('div');
tip.className = 'cp-tip';
tip.style.display = 'none';
document.body.appendChild(tip);
function showTip(el, x, y) {
  const r = state.result;
  let html = '';
  if (el.dataset.player) {
    const p = r.players.byId[+el.dataset.player];
    if (!p) return;
    const cap = r.best && r.best.captain && r.best.captain.id === p.id;
    html = `<div class="cp-tip-title">${esc(p.name)} <span>${POS_NAME[p.pos]} · ${esc(p.clubShort)} · expected minutes ${p.mins}${p.minsEdited ? ' (your edit)' : ''}</span></div>
      <table class="cp-tip-table">${p.fx.map(x => `<tr><td>v ${esc(shortOf(x.f.opp))} (${x.f.ha})</td><td><span>${fmtDate(x.f.date, true)}${x.f.src === 'market' ? ' · odds' : ' · model'}</span></td><td class="cp-tip-num">${fmt2(x.xp)}</td></tr>`).join('') || '<tr><td colspan="3">No fixture</td></tr>'}
      <tr class="cp-tip-total"><td colspan="2">Gameweek xP${cap ? ' (× 2 as captain)' : ''}</td><td class="cp-tip-num">${fmt2(p.xp * (cap ? 2 : 1))}</td></tr></table>
      <div class="cp-tip-foot">Breakdown and minutes on <a href="${POS_PAGE[p.pos]}">Player Picks</a>. Click × on his shirt to exclude him.</div>`;
  } else if (el.dataset.club) {
    const wk = clubWeek(+el.dataset.club, state.gw);
    const c = state.plans.CLUB.clubs.find(q => q.id === +el.dataset.club);
    if (!wk || !c) return;
    html = `<div class="cp-tip-title">${esc(c.name)} <span>club pick · ${esc(c.league)}</span></div>
      <table class="cp-tip-table">${wk.fx.map(f => `<tr><td>v ${esc(shortOf(f.opp))} (${f.ha})</td><td><span>win ${(f.win * 100).toFixed(0)}% · draw ${(f.draw * 100).toFixed(0)}% · clean sheet ${(f.cs * 100).toFixed(0)}%</span></td><td class="cp-tip-num">${fmt2(f.xp)}</td></tr>`).join('')}
      <tr class="cp-tip-total"><td colspan="2">Gameweek xP</td><td class="cp-tip-num">${fmt2(wk.xp)}</td></tr></table>`;
  }
  if (!html) return;
  tip.innerHTML = html;
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

/* ---------- render ---------- */
function playerCard(p, best) {
  const cap = best.captain && best.captain.id === p.id, vice = best.vice && best.vice.id === p.id;
  const pinned = state.opts.pinned.includes(p.id);
  const fx = p.fx.length ? p.fx.map(x => `${abbr(x.f.opp)} (${x.f.ha})`).join(', ') : 'No fixture';
  return `<div class="op-player" data-tip="p" data-player="${p.id}">
    <div class="op-shirt">${shirtSvg(p.club, p.pos === 'GK')}
      ${cap ? '<span class="op-badge op-cap" title="Captain: double points">C</span>' : vice ? '<span class="op-badge op-vice" title="Vice-captain: double points if the captain does not play">V</span>' : ''}
      <button class="op-act op-x" data-act="exclude" data-id="${p.id}" title="Exclude ${esc(p.name)} and re-optimise">×</button>
      ${pinned ? `<button class="op-act op-pinned" data-act="unpin" data-id="${p.id}" title="Pinned: click to unpin">⇧</button>` : ''}
    </div>
    <div class="op-plate"><div class="op-name">${esc(p.name)}</div><div class="op-club">${esc(p.clubShort)} · ${esc(fx)}</div></div>
    <div class="op-xp">${fmt(p.xp * (cap ? 2 : 1))}</div>
  </div>`;
}
function clubCard(x) {
  const c = x.club, k = kitOf(c.id);
  return `<div class="op-clubcard" data-tip="c" data-club="${x.id}">
    <div class="op-crest" style="background:${k.a};color:${k.b}">${esc(c.abbr || abbr(c.name))}</div>
    <div class="op-clubinfo"><strong>${esc(c.short || c.name)}</strong>
      <div class="op-club">${x.wk && x.wk.fx.length ? x.wk.fx.map(f => `${abbr(f.opp)} (${f.ha})`).join(', ') : 'No fixture'}</div>
      <div class="op-club">${esc(c.league)}</div></div>
    <div class="op-xp">${x.wk ? fmt(x.wk.xp) : '0.0'}</div></div>`;
}
function render() {
  const scrollY = window.scrollY;
  const gw = state.gw;
  const meta = state.plans.CLUB.gameweeks.find(g => g.gw === gw);
  shirtSeq = 0;
  const r = { players: optimise(gw), clubs: chooseClubs(gw) };
  state.result = r;
  const best = r.players.best;
  const clubRows = r.clubs;
  const clubXp = clubRows.reduce((a, x) => a + x.wk.xp, 0);
  const options = state.plans.CLUB.gameweeks.map(g => `<option value="${g.gw}" ${g.gw === gw ? 'selected' : ''}>GW ${g.gw} · ${fmtDate(g.start)}${g.end !== g.start ? '–' + fmtDate(g.end) : ''}</option>`).join('');

  let pitch = '<div class="cp-warn">No valid team: check pinned players.</div>';
  if (best) {
    const rows = POSITIONS.map(pos => `<div class="op-row">${best.team.filter(p => p.pos === pos).sort((a, b) => b.xp - a.xp).map(p => playerCard(p, best)).join('')}</div>`).join('');
    pitch = `<div class="op-pitch"><div class="op-lines"><span class="op-box"></span><span class="op-circle"></span></div>${rows}</div>`;
  }
  const formations = r.players.results.map(x => `<tr class="${best && x === best ? 'op-best' : ''}"><td>${x.F.name}</td><td class="cp-right">${x.problem ? `<span class="cp-muted">${esc(x.problem)}</span>` : fmt(x.total)}</td></tr>`).join('');
  // next best options not in the team, per position
  const inTeam = new Set(best ? best.team.map(p => p.id) : []);
  const excluded = new Set(state.opts.excluded);
  const bench = POSITIONS.map(pos => {
    const alts = r.players.all.filter(p => p.pos === pos && !inTeam.has(p.id) && !excluded.has(p.id) && p.mins > 0)
      .sort((a, b) => b.xp - a.xp).slice(0, 5);
    return `<div class="op-alt"><h3>${POS_NAME[pos]}s</h3>${alts.map(p => `<div class="op-alt-row" data-tip="p" data-player="${p.id}">
      <span>${esc(p.name)} <span class="cp-muted cp-small">${esc(p.clubShort)}</span></span><span class="op-alt-xp">${fmt(p.xp)}</span>
      <button class="op-act-inline" data-act="pin" data-id="${p.id}" title="Pin him into the team">⇧ pin</button></div>`).join('') || '<div class="cp-muted cp-small">None</div>'}</div>`;
  }).join('');
  const pinnedList = state.opts.pinned.map(id => r.players.byId[id]).filter(Boolean);
  const excludedList = state.opts.excluded.map(id => r.players.byId[id]).filter(Boolean);
  const chip = (p, act) => `<span class="op-chip">${esc(p.name)} <button class="op-act-inline" data-act="${act}" data-id="${p.id}" title="Undo">×</button></span>`;
  const names = state.players.map(p => `<option value="${esc(`${p.name} (${p.clubShort}, ${p.pos})`)}"></option>`).join('');
  const total = (best ? best.total : 0) + clubXp;

  root.innerHTML = `
    <div class="cp-controls">
      <button id="op-prev" class="cp-nav-btn">&#8592;</button><select id="op-gw">${options}</select><button id="op-next" class="cp-nav-btn">&#8594;</button>
      <label class="kp-toggle" title="One Club chip: no limit on players from one club this gameweek (once a season)"><input type="checkbox" id="op-oneclub" ${state.opts.oneClub ? 'checked' : ''}/> One Club chip</label>
    </div>
    <h2 class="cp-h2">Best team for GW ${gw} <span class="cp-sub">${fmtDate(meta.start)}${meta.end !== meta.start ? ' – ' + fmtDate(meta.end) : ''} · ${meta.games} games${meta.doubles ? ` · ${meta.doubles} clubs play twice` : ''}</span></h2>
    <div class="op-board">
      <div class="op-left">${pitch}
        <div class="op-clubs-row">${clubRows.map(clubCard).join('') || '<div class="cp-muted">No club picks available.</div>'}</div>
      </div>
      <div class="op-side">
        <div class="op-total"><div class="op-total-num">${fmt(total)}</div><div class="op-total-lab">expected points</div>
          <div class="op-total-split">players ${best ? fmt(best.players) : '0.0'} + captain ${best ? fmt(best.captain.xp) : '0.0'} + clubs ${fmt(clubXp)}</div></div>
        ${best ? `<div class="op-panel"><div><span class="op-badge op-cap op-inline">C</span> <strong>${esc(best.captain.name)}</strong> ${fmt(best.captain.xp)} → ${fmt(2 * best.captain.xp)}</div>
          <div><span class="op-badge op-vice op-inline">V</span> ${esc(best.vice.name)} ${fmt(best.vice.xp)}</div></div>` : ''}
        <div class="op-panel"><strong>Formations</strong> <span class="cp-muted cp-small">(players incl. captain)</span><table class="op-ftable">${formations}</table></div>
        <div class="op-panel"><strong>Clubs</strong>
          <p class="cp-small">The two highest-xP clubs this gameweek. This ignores picks left (each club can be picked ${MAX_USES} times a season):
          the <a href="clubs.html">Club Planner</a> tracks those and plans the whole season.</p></div>
      </div>
    </div>
    <div class="op-tools">
      <div><label class="cp-small" for="op-pin">Pin a player into the team:</label>
        <input id="op-pin" list="op-names" type="text" placeholder="Type a name..." autocomplete="off" /><datalist id="op-names">${names}</datalist></div>
      ${pinnedList.length ? `<div class="cp-small">Pinned: ${pinnedList.map(p => chip(p, 'unpin')).join(' ')}</div>` : ''}
      ${excludedList.length ? `<div class="cp-small">Excluded: ${excludedList.map(p => chip(p, 'unexclude')).join(' ')} <button class="cp-link-btn" id="op-reset">Clear all</button></div>` : ''}
    </div>
    <h2 class="cp-h2">Next best options <span class="cp-sub">not in the team above</span></h2>
    <div class="op-alts">${bench}</div>
    <div class="cp-caveat"><strong>How this works</strong><ul>
      <li>Rules (Fantasy EFL help centre): 7 players in a 1-2-2-2, 1-2-3-1 or 1-3-2-1 formation, at most ${MAX_PER_CLUB} players from one club (unless the One Club chip is on),
        captain scores double, plus 2 clubs, each club usable ${MAX_USES} times a season. Doubles score twice; blanks score nothing.
        Games that have already kicked off still count, so this is the best team for the whole gameweek.</li>
      <li>The team is the exact best for total xP with the captain counted twice (every formation is checked). Captain = highest xP, vice = second highest.</li>
      <li>Player xP and expected minutes come from the <a href="picks.html">Player Picks</a> page (keepers, defenders, midfielders and
        forwards), including any minutes you have edited there (saved in this browser). Set a player to 0 minutes there, or click × here, to leave him out.</li>
      <li>Max Captain chip: the captain becomes whoever actually scores most, which is worth more than picking the highest xP when several players have similar xP (e.g. a double gameweek). Not modelled here.</li>
      <li>Single gameweeks are mostly luck: a 1-point xP gap is small.</li>
    </ul></div>
    <p class="cp-foot">Data updated ${esc(new Date(state.plans.CLUB.generatedAt).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }))}.</p>`;
  tip.style.display = 'none';
  bind();
  window.scrollTo(0, scrollY);
}

function bind() {
  document.getElementById('op-gw').addEventListener('change', e => { state.gw = +e.target.value; render(); });
  const step = d => {
    const gws = state.plans.CLUB.gameweeks;
    const i = gws.findIndex(g => g.gw === state.gw) + d;
    if (i >= 0 && i < gws.length) { state.gw = gws[i].gw; render(); }
  };
  document.getElementById('op-prev').addEventListener('click', () => step(-1));
  document.getElementById('op-next').addEventListener('click', () => step(1));
  document.getElementById('op-oneclub').addEventListener('change', e => { state.opts.oneClub = e.target.checked; saveOwn(); render(); });
  document.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', e => {
    e.stopPropagation();
    const id = +b.dataset.id, o = state.opts;
    const drop = (arr, v) => arr.filter(x => x !== v);
    if (b.dataset.act === 'exclude') { o.excluded = [...drop(o.excluded, id), id]; o.pinned = drop(o.pinned, id); }
    if (b.dataset.act === 'unexclude') o.excluded = drop(o.excluded, id);
    if (b.dataset.act === 'pin') { o.pinned = [...drop(o.pinned, id), id]; o.excluded = drop(o.excluded, id); }
    if (b.dataset.act === 'unpin') o.pinned = drop(o.pinned, id);
    saveOwn(); render();
  }));
  const reset = document.getElementById('op-reset');
  if (reset) reset.addEventListener('click', () => { state.opts.excluded = []; saveOwn(); render(); });
  document.getElementById('op-pin').addEventListener('change', e => {
    const p = state.players.find(q => `${q.name} (${q.clubShort}, ${q.pos})` === e.target.value);
    if (p) { state.opts.pinned = [...state.opts.pinned.filter(x => x !== p.id), p.id]; state.opts.excluded = state.opts.excluded.filter(x => x !== p.id); saveOwn(); render(); }
  });
}

async function load() {
  statusEl.textContent = 'Loading...';
  try {
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
    const own = readJson(OWN_STORE, {});
    state.opts = { pinned: Array.isArray(own.pinned) ? own.pinned : [], excluded: Array.isArray(own.excluded) ? own.excluded : [], oneClub: !!own.oneClub };
    state.gw = state.plans.CLUB.firstGw;
    statusEl.textContent = '';
    render();
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

load();
