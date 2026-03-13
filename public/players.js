const PLAYERS_URL = '/api/players';
const ROUNDS_URL = '/api/rounds';
const PROFILE_URL = id => `/api/player?id=${id}`;

const searchInput    = document.getElementById('search-input');
const searchResults  = document.getElementById('search-results');
const leaderboardDiv = document.getElementById('leaderboard');
const playerStats    = document.getElementById('player-stats');

let allPlayers = [];
let allRounds = [];
const profileCache = {};

// ─── Column definitions by position ──────────────────────────────────────────
const BASE_COLS = [
  { label: 'GW',       key: 'roundId',        title: 'Game Week' },
  { label: 'Mins',     key: 'minutesPlayed',   title: 'Minutes Played (+1 if <60, +2 if 60+)' },
  { label: 'Goals',    key: 'goalsScored',     title: 'Goals Scored (GK +10, DEF +7, MID +6, FWD +5)' },
  { label: 'Hat-T',    key: 'hatTricks',       title: 'Hat-Tricks (+5 bonus)' },
  { label: 'Assists',  key: 'assists',         title: 'Assists (+3)' },
  { label: 'Pen Miss', key: 'penaltyMisses',   title: 'Penalty Misses (-3)' },
  { label: 'OG',       key: 'ownGoals',        title: 'Own Goals (-3)' },
  { label: 'YC',       key: 'yellowCards',     title: 'Yellow Cards (-1)' },
  { label: 'RC',       key: 'redCards',        title: 'Red Cards (-3)' },
];

const COLS_BY_POS = {
  GK: [
    ...BASE_COLS,
    { label: 'Saves',    key: 'saves',          title: 'Saves (every 3 = +2)' },
    { label: 'Pen Save', key: 'penaltySaves',   title: 'Penalty Saves (+5)' },
    { label: 'CS',       key: 'cleanSheet',     title: 'Clean Sheet — 60+ mins (+5)' },
    { label: 'GC',       key: 'goalsConceded',  title: 'Goals Conceded (every 2 = -1)' },
    { label: 'Pts',      key: 'points',         title: 'Total Points' },
  ],
  DEF: [
    ...BASE_COLS,
    { label: 'CS',         key: 'cleanSheet',    title: 'Clean Sheet — 60+ mins (+5)' },
    { label: 'GC',         key: 'goalsConceded', title: 'Goals Conceded (every 2 = -1)' },
    { label: 'Clearances', key: 'clearances',    title: 'Clearances (every 4 = +1)' },
    { label: 'Blocks',     key: 'blocks',        title: 'Blocks (every 2 = +1)' },
    { label: 'Tackles',    key: 'tackles',       title: 'Tackles (every 2 = +1)' },
    { label: 'Pts',        key: 'points',        title: 'Total Points' },
  ],
  MID: [
    ...BASE_COLS,
    { label: 'Intercepts', key: 'interceptions', title: 'Interceptions (+2 each)' },
    { label: 'KP',         key: 'keyPasses',     title: 'Key Passes (every 2 = +1)' },
    { label: 'SOT',        key: 'shotsOnTarget', title: 'Shots on Target (+1 each)' },
    { label: 'Pts',        key: 'points',        title: 'Total Points' },
  ],
  FWD: [
    ...BASE_COLS,
    { label: 'KP',  key: 'keyPasses',     title: 'Key Passes (every 2 = +1)' },
    { label: 'SOT', key: 'shotsOnTarget', title: 'Shots on Target (+1 each)' },
    { label: 'Pts', key: 'points',        title: 'Total Points' },
  ],
};

// ─── Points calculation ───────────────────────────────────────────────────────
// Returns the points this stat contributed in a single game, or null if
// this column should not show a points bracket (GW, Pts).
function calcStatPoints(key, val, pos, minsPlayed) {
  val = val || 0;
  switch (key) {
    case 'minutesPlayed':  return val >= 60 ? 2 : val > 0 ? 1 : 0;
    case 'goalsScored':    return val * ({ GK: 10, DEF: 7, MID: 6, FWD: 5 }[pos] || 0);
    case 'hatTricks':      return val * 5;
    case 'assists':        return val * 3;
    case 'penaltyMisses':  return val * -3;
    case 'ownGoals':       return val * -3;
    case 'yellowCards':    return val * -1;
    case 'redCards':       return val * -3;
    case 'saves':          return Math.floor(val / 3) * 2;
    case 'penaltySaves':   return val * 5;
    case 'cleanSheet':     return (val && minsPlayed >= 60) ? 5 : 0;
    case 'goalsConceded':  return -Math.floor(val / 2);
    case 'clearances':     return Math.floor(val / 4);
    case 'blocks':         return Math.floor(val / 2);
    case 'tackles':        return Math.floor(val / 2);
    case 'interceptions':  return val * 2;
    case 'keyPasses':      return Math.floor(val / 2);
    case 'shotsOnTarget':  return val;
    default:               return null; // roundId, points — no bracket
  }
}

function formatCell(key, val, pos, minsPlayed) {
  if (key === 'roundId' || key === 'points') return val;
  const pts = calcStatPoints(key, val, pos, minsPlayed);
  if (pts === null || val === 0) return val;
  if (pts === 0) return val; // stat happened but worth 0pts (e.g. CS with <60 mins)
  const sign = pts > 0 ? '+' : '';
  const cls  = pts > 0 ? 'stat-pts-pos' : 'stat-pts-neg';
  return `${val} <span class="${cls}">(${sign}${pts})</span>`;
}

// ─── Build game lookup by round and squad ─────────────────────────────────────
function buildGamesByRound(rounds) {
  const gamesByRound = {};
  for (const round of rounds) {
    if (round.status !== 'completed') continue;
    gamesByRound[round.roundNumber] = {};
    for (const game of round.games) {
      gamesByRound[round.roundNumber][game.homeId] = { ...game, isHome: true };
      gamesByRound[round.roundNumber][game.awayId] = { ...game, isHome: false };
    }
  }
  return gamesByRound;
}

// ─── Points per 90 calculation ────────────────────────────────────────────────
function calcPer90(games, cols, pos, squadId) {
  const totalMins = games.reduce((s, g) => s + (g.minutesPlayed || 0), 0);
  if (totalMins === 0) return null;

  const overallPts = games.reduce((s, g) => s + (g.points || 0), 0);
  const overallPer90 = (overallPts / totalMins) * 90;

  // Calculate home/away per-90
  const gamesByRound = buildGamesByRound(allRounds);
  let homeMins = 0, homePts = 0, awayMins = 0, awayPts = 0;

  for (const game of games) {
    const roundNum = game.round || game.roundId || game.roundNumber;
    const gameInfo = gamesByRound[roundNum]?.[squadId];
    const mins = game.minutesPlayed || game.minutes || 0;
    const pts = game.points || 0;

    if (gameInfo && gameInfo.isHome) {
      homeMins += mins;
      homePts += pts;
    } else if (gameInfo && !gameInfo.isHome) {
      awayMins += mins;
      awayPts += pts;
    }
  }

  let homePer90 = homeMins > 0 ? (homePts / homeMins) * 90 : 0;
  let awayPer90 = awayMins > 0 ? (awayPts / awayMins) * 90 : 0;

  // Fallback: if no games matched, use overall per-90 for both
  if (homePer90 === 0 && awayPer90 === 0 && totalMins > 0) {
    homePer90 = overallPer90;
    awayPer90 = overallPer90;
  }

  const perNinety = {};
  for (const col of cols) {
    if (col.key === 'roundId' || col.key === 'points') continue;
    const total = games.reduce((s, g) =>
      s + calcStatPoints(col.key, g[col.key] || 0, pos, g.minutesPlayed || 0), 0);
    const per90 = (total / totalMins) * 90;
    if (Math.abs(per90) >= 0.01) perNinety[col.key] = per90;
  }

  return { totalMins, overallPer90, homePer90, awayPer90, perNinety };
}

function buildPer90Section(per90Data, cols) {
  if (!per90Data) return '';

  const { overallPer90, homePer90, awayPer90, totalMins } = per90Data;
  const totalMinsLabel = `${Math.round(totalMins)} mins played`;

  const pills = cols
    .filter(c => c.key !== 'roundId' && c.key !== 'points' && per90Data.perNinety[c.key] !== undefined)
    .map(c => {
      const val = per90Data.perNinety[c.key];
      const cls = val > 0 ? 'per90-pos' : 'per90-neg';
      const sign = val > 0 ? '+' : '';
      return `<div class="per90-pill ${cls}">${c.label} <strong>${sign}${val.toFixed(2)}</strong></div>`;
    });

  const overallSign = overallPer90 > 0 ? '+' : '';
  const homeSign = homePer90 > 0 ? '+' : '';
  const awaySign = awayPer90 > 0 ? '+' : '';

  const totalPill = `<div class="per90-pill per90-total">Overall <strong>${overallSign}${overallPer90.toFixed(1)}</strong></div>`;
  const homePill = `<div class="per90-pill">Home <strong>${homeSign}${homePer90.toFixed(1)}</strong></div>`;
  const awayPill = `<div class="per90-pill">Away <strong>${awaySign}${awayPer90.toFixed(1)}</strong></div>`;

  return `
    <div class="per90-section">
      <h3 class="per90-label">Points per 90 min <span class="per90-mins">(${totalMinsLabel})</span></h3>
      <div class="per90-pills">${totalPill}${homePill}${awayPill}${pills.join('')}</div>
    </div>
  `;
}

// ─── Sort state ───────────────────────────────────────────────────────────────
let sortKey = 'roundId';
let sortDir = 'desc';

// ─── Module-level player/profile refs for sort re-render ─────────────────────
let currentPlayer  = null;
let currentProfile = null;

// ─── Data loading ─────────────────────────────────────────────────────────────
async function loadPlayers() {
  try {
    const [playersRes, roundsRes] = await Promise.all([
      fetch(PLAYERS_URL),
      fetch(ROUNDS_URL),
    ]);
    allPlayers = await playersRes.json();
    allRounds = await roundsRes.json();
    showLeaderboard();
  } catch (e) {
    leaderboardDiv.innerHTML = '<p class="error-msg">Failed to load player list.</p>';
  }
}

// ─── Leaderboard ─────────────────────────────────────────────────────────────
function showLeaderboard() {
  const MIN_APPS = 12;
  const top = allPlayers
    .filter(p => p.appearances >= MIN_APPS)
    .sort((a, b) => (b.averagePoints || 0) - (a.averagePoints || 0))
    .slice(0, 30);

  if (top.length === 0) {
    leaderboardDiv.innerHTML = '<p class="loading-msg">No player data yet.</p>';
    return;
  }

  const rows = top.map((p, i) => {
    const name = p.displayName || `${p.firstName} ${p.lastName}`;
    return `
      <tr class="lb-row" data-id="${p.id}">
        <td class="pos">${i + 1}</td>
        <td>${name}</td>
        <td><span class="player-pos pos-${p.position}">${p.position}</span></td>
        <td>${p.appearances}</td>
        <td>${p.totalPoints}</td>
        <td class="pts-col">${(p.averagePoints || 0).toFixed(2)}</td>
      </tr>`;
  }).join('');

  leaderboardDiv.innerHTML = `
    <h2 class="section-heading">Top Players — pts/game (min. 12 apps) • Click for actual Home/Away per-90</h2>
    <div class="stats-table-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Player</th><th>Pos</th><th>Apps</th><th>Pts</th><th>Pts/Game</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;

  leaderboardDiv.querySelectorAll('.lb-row').forEach(row => {
    row.addEventListener('click', () => {
      const player = allPlayers.find(p => p.id === Number(row.dataset.id));
      if (player) showPlayer(player);
    });
  });

  leaderboardDiv.style.display = '';
}

// ─── Search ───────────────────────────────────────────────────────────────────
function positionOrder(pos) {
  return { GK: 0, DEF: 1, MID: 2, FWD: 3 }[pos] ?? 4;
}

searchInput.addEventListener('input', () => {
  const query = searchInput.value.trim().toLowerCase();
  searchResults.innerHTML = '';
  playerStats.innerHTML = '';

  if (query.length < 2) {
    leaderboardDiv.style.display = '';
    return;
  }

  leaderboardDiv.style.display = 'none';

  const matches = allPlayers
    .filter(p => (p.displayName || `${p.firstName} ${p.lastName}`).toLowerCase().includes(query))
    .sort((a, b) => positionOrder(a.position) - positionOrder(b.position) || b.totalPoints - a.totalPoints)
    .slice(0, 20);

  if (matches.length === 0) {
    searchResults.innerHTML = '<div class="result-item">No players found.</div>';
    return;
  }

  matches.forEach(player => {
    const name = player.displayName || `${player.firstName} ${player.lastName}`;
    const div  = document.createElement('div');
    div.className   = 'result-item';
    div.textContent = `${name} — ${player.position} — ${player.totalPoints} pts`;
    div.addEventListener('click', () => showPlayer(player));
    searchResults.appendChild(div);
  });
});

// ─── Show player ──────────────────────────────────────────────────────────────
async function showPlayer(player) {
  const name = player.displayName || `${player.firstName} ${player.lastName}`;
  leaderboardDiv.style.display = 'none';
  searchResults.innerHTML = '';
  searchInput.value = name;
  playerStats.innerHTML = `<p class="loading-msg">Loading stats for ${name}...</p>`;

  sortKey = 'roundId';
  sortDir = 'desc';

  try {
    let profile = profileCache[player.id];
    if (!profile) {
      const res = await fetch(PROFILE_URL(player.id));
      profile   = await res.json();
      profileCache[player.id] = profile;
    }
    currentPlayer  = player;
    currentProfile = profile;
    renderPlayerStats(player, profile);
  } catch (e) {
    playerStats.innerHTML = `<p class="error-msg">Failed to load stats for ${name}.</p>`;
  }
}

// ─── Render player stats ──────────────────────────────────────────────────────
function renderPlayerStats(player, profile) {
  const name  = player.displayName || `${player.firstName} ${player.lastName}`;
  const games = profile.results || [];
  const pos   = player.position;

  if (games.length === 0) {
    playerStats.innerHTML = `<p class="loading-msg">No game data found for ${name}.</p>`;
    return;
  }

  const cols = COLS_BY_POS[pos] || [...BASE_COLS, { label: 'Pts', key: 'points', title: 'Total Points' }];

  // Add back button
  const backBtn = `<button style="margin-bottom: 16px; padding: 8px 16px; background: #f05a28; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;" onclick="searchInput.value = ''; searchResults.innerHTML = ''; playerStats.innerHTML = ''; leaderboardDiv.style.display = '';">← Back to Leaderboard</button>`;

  // Calculate per-90 stats
  const per90Data = calcPer90(games, cols, pos, player.squadId);
  const per90Overall = per90Data ? per90Data.overallPer90.toFixed(1) : '—';

  // Calculate home/away minutes
  const gamesByRound = buildGamesByRound(allRounds);
  let homeMins = 0, awayMins = 0;
  for (const game of games) {
    const roundNum = game.round || game.roundId || game.roundNumber;
    const gameInfo = gamesByRound[roundNum]?.[player.squadId];
    const mins = game.minutesPlayed || game.minutes || 0;
    if (gameInfo && gameInfo.isHome) {
      homeMins += mins;
    } else if (gameInfo && !gameInfo.isHome) {
      awayMins += mins;
    }
  }
  const totalMins = per90Data?.totalMins || 0;

  const summaryHtml = `
    <div class="player-header">
      <h2>${name}</h2>
      <span class="player-pos pos-${pos}">${pos}</span>
    </div>
    <div class="season-summary">
      <div class="stat-pill">Total Pts <strong>${player.totalPoints}</strong></div>
      <div class="stat-pill">Pts/90 <strong>${per90Overall}</strong></div>
      <div class="stat-pill">Total Mins <strong>${Math.round(totalMins)}</strong></div>
      <div class="stat-pill">Home Mins <strong>${homeMins}</strong></div>
      <div class="stat-pill">Away Mins <strong>${awayMins}</strong></div>
      <div class="stat-pill">Apps <strong>${player.appearances}</strong></div>
      <div class="stat-pill">Goals <strong>${player.goalsScored}</strong></div>
      <div class="stat-pill">Assists <strong>${player.assists}</strong></div>
      ${pos === 'GK' || pos === 'DEF' ? `<div class="stat-pill">Clean Sheets <strong>${player.cleanSheets}</strong></div>` : ''}
      ${pos === 'GK'  ? `<div class="stat-pill">Saves <strong>${player.saves}</strong></div>` : ''}
      ${pos === 'MID' ? `<div class="stat-pill">Intercepts <strong>${player.interceptions}</strong></div>` : ''}
      ${pos === 'MID' || pos === 'FWD' ? `<div class="stat-pill">SOT <strong>${player.shotsOnTarget}</strong></div>` : ''}
      ${pos === 'DEF' ? `<div class="stat-pill">Tackles <strong>${player.tackles}</strong></div>` : ''}
    </div>
  `;

  const per90Html = buildPer90Section(per90Data, cols);

  playerStats.innerHTML = backBtn + summaryHtml + per90Html + buildTable(games, cols, pos);
  attachSortHandlers(cols, pos);
}

// ─── Table ────────────────────────────────────────────────────────────────────
function buildTable(games, cols, pos) {
  const sorted = [...games].sort((a, b) => {
    const av = a[sortKey] ?? 0;
    const bv = b[sortKey] ?? 0;
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  const headers = cols.map(c => {
    const isActive = c.key === sortKey;
    const dirClass = isActive ? ` sort-${sortDir}` : '';
    return `<th class="sortable${dirClass}" data-key="${c.key}" title="${c.title}">${c.label}</th>`;
  }).join('');

  const rows = sorted.map(g => {
    const cells = cols.map(c => {
      const raw = g[c.key] ?? 0;
      const content = formatCell(c.key, raw, pos, g.minutesPlayed || 0);
      const tdCls = c.key === 'points' ? ' class="pts-col"' : '';
      return `<td${tdCls}>${content}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');

  return `
    <div class="stats-table-wrap">
      <table id="stats-table">
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function attachSortHandlers(cols, pos) {
  document.querySelectorAll('#stats-table thead th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (sortKey === key) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = key;
        sortDir = 'desc';
      }
      const wrap = document.querySelector('.stats-table-wrap');
      if (wrap) {
        const games = currentProfile.results || [];
        wrap.outerHTML = buildTable(games, cols, pos);
        attachSortHandlers(cols, pos);
      }
    });
  });
}

loadPlayers();
