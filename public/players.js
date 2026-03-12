const PLAYERS_URL = '/api/players';
const PROFILE_URL = id => `/api/player?id=${id}`;

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const playerStats = document.getElementById('player-stats');

let allPlayers = [];
const profileCache = {};

// ─── Column definitions by position ──────────────────────────────────────────
// Based on official EFL Fantasy scoring rules.
// Keys map directly to fields in the player_profiles results array.

// Stats that score for ALL positions
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
  // GK: base + saves (every 3 = +2), pen save (+5), clean sheet (+5), goals conceded (every 2 = -1)
  GK: [
    ...BASE_COLS,
    { label: 'Saves',    key: 'saves',          title: 'Saves (every 3 = +2)' },
    { label: 'Pen Save', key: 'penaltySaves',   title: 'Penalty Saves (+5)' },
    { label: 'CS',       key: 'cleanSheet',     title: 'Clean Sheet — 60+ mins (+5)' },
    { label: 'GC',       key: 'goalsConceded',  title: 'Goals Conceded (every 2 = -1)' },
    { label: 'Pts',      key: 'points',         title: 'Total Points' },
  ],
  // DEF: base + clean sheet (+5), goals conceded (every 2 = -1), clearances (every 4 = +1), blocks (every 2 = +1), tackles (every 2 = +1)
  DEF: [
    ...BASE_COLS,
    { label: 'CS',         key: 'cleanSheet',    title: 'Clean Sheet — 60+ mins (+5)' },
    { label: 'GC',         key: 'goalsConceded', title: 'Goals Conceded (every 2 = -1)' },
    { label: 'Clearances', key: 'clearances',    title: 'Clearances (every 4 = +1)' },
    { label: 'Blocks',     key: 'blocks',        title: 'Blocks (every 2 = +1)' },
    { label: 'Tackles',    key: 'tackles',       title: 'Tackles (every 2 = +1)' },
    { label: 'Pts',        key: 'points',        title: 'Total Points' },
  ],
  // MID: base + interceptions (+2 each), key passes (every 2 = +1), shots on target (+1 each)
  // No clean sheet or goals conceded for MID
  MID: [
    ...BASE_COLS,
    { label: 'Intercepts', key: 'interceptions', title: 'Interceptions (+2 each)' },
    { label: 'KP',         key: 'keyPasses',     title: 'Key Passes (every 2 = +1)' },
    { label: 'SOT',        key: 'shotsOnTarget', title: 'Shots on Target (+1 each)' },
    { label: 'Pts',        key: 'points',        title: 'Total Points' },
  ],
  // FWD: base + key passes (every 2 = +1), shots on target (+1 each)
  FWD: [
    ...BASE_COLS,
    { label: 'KP',  key: 'keyPasses',     title: 'Key Passes (every 2 = +1)' },
    { label: 'SOT', key: 'shotsOnTarget', title: 'Shots on Target (+1 each)' },
    { label: 'Pts', key: 'points',        title: 'Total Points' },
  ],
};

// ─── Sort state ───────────────────────────────────────────────────────────────
let sortKey = 'roundId';
let sortDir = 'desc'; // default: latest GW first

// ─── Data loading ─────────────────────────────────────────────────────────────
async function loadPlayers() {
  try {
    const res = await fetch(PLAYERS_URL);
    allPlayers = await res.json();
  } catch (e) {
    searchResults.textContent = 'Failed to load player list.';
  }
}

function positionOrder(pos) {
  return { GK: 0, DEF: 1, MID: 2, FWD: 3 }[pos] ?? 4;
}

// ─── Search ───────────────────────────────────────────────────────────────────
searchInput.addEventListener('input', () => {
  const query = searchInput.value.trim().toLowerCase();
  searchResults.innerHTML = '';
  playerStats.innerHTML = '';

  if (query.length < 2) return;

  const matches = allPlayers
    .filter(p => (p.displayName || `${p.firstName} ${p.lastName}`).toLowerCase().includes(query))
    .sort((a, b) => positionOrder(a.position) - positionOrder(b.position) || b.totalPoints - a.totalPoints)
    .slice(0, 20);

  if (matches.length === 0) {
    searchResults.textContent = 'No players found.';
    return;
  }

  matches.forEach(player => {
    const name = player.displayName || `${player.firstName} ${player.lastName}`;
    const div = document.createElement('div');
    div.className = 'result-item';
    div.textContent = `${name} — ${player.position} — ${player.totalPoints} pts`;
    div.addEventListener('click', () => showPlayer(player));
    searchResults.appendChild(div);
  });
});

// ─── Show player ──────────────────────────────────────────────────────────────
async function showPlayer(player) {
  const name = player.displayName || `${player.firstName} ${player.lastName}`;
  playerStats.innerHTML = `<p class="loading-msg">Loading stats for ${name}...</p>`;
  searchResults.innerHTML = '';
  searchInput.value = name;

  // Reset sort to default when switching player
  sortKey = 'roundId';
  sortDir = 'desc';

  try {
    let profile = profileCache[player.id];
    if (!profile) {
      const res = await fetch(PROFILE_URL(player.id));
      profile = await res.json();
      profileCache[player.id] = profile;
    }
    renderPlayerStats(player, profile);
  } catch (e) {
    playerStats.innerHTML = `<p class="error-msg">Failed to load stats for ${name}.</p>`;
  }
}

// ─── Render ───────────────────────────────────────────────────────────────────
let currentPlayer = null;
let currentProfile = null;

function renderPlayerStats(player, profile) {
  currentPlayer = player;
  currentProfile = profile;

  const name = player.displayName || `${player.firstName} ${player.lastName}`;
  const games = profile.results || [];

  if (games.length === 0) {
    playerStats.innerHTML = `<p class="loading-msg">No game data found for ${name}.</p>`;
    return;
  }

  const pos = player.position;
  const cols = COLS_BY_POS[pos] || [...BASE_COLS, { label: 'Pts', key: 'points', title: 'Total Points' }];

  const summaryHtml = `
    <div class="player-header">
      <h2>${name}</h2>
      <span class="player-pos pos-${pos}">${pos}</span>
    </div>
    <div class="season-summary">
      <div class="stat-pill">Total Pts <strong>${player.totalPoints}</strong></div>
      <div class="stat-pill">Avg Pts <strong>${player.averagePoints?.toFixed(1) ?? '—'}</strong></div>
      <div class="stat-pill">Apps <strong>${player.appearances}</strong></div>
      <div class="stat-pill">Goals <strong>${player.goalsScored}</strong></div>
      <div class="stat-pill">Assists <strong>${player.assists}</strong></div>
      ${pos === 'GK' || pos === 'DEF' ? `<div class="stat-pill">Clean Sheets <strong>${player.cleanSheets}</strong></div>` : ''}
      ${pos === 'GK' ? `<div class="stat-pill">Saves <strong>${player.saves}</strong></div>` : ''}
      ${pos === 'MID' ? `<div class="stat-pill">Intercepts <strong>${player.interceptions}</strong></div>` : ''}
      ${pos === 'MID' || pos === 'FWD' ? `<div class="stat-pill">SOT <strong>${player.shotsOnTarget}</strong></div>` : ''}
      ${pos === 'DEF' ? `<div class="stat-pill">Tackles <strong>${player.tackles}</strong></div>` : ''}
    </div>
  `;

  playerStats.innerHTML = summaryHtml + buildTable(games, cols);
  attachSortHandlers(cols);
}

function buildTable(games, cols) {
  const sorted = sortGames([...games], sortKey, sortDir);

  const headers = cols.map(c => {
    const isActive = c.key === sortKey;
    const dirClass = isActive ? ` sort-${sortDir}` : '';
    return `<th class="sortable${dirClass}" data-key="${c.key}" title="${c.title}">${c.label}</th>`;
  }).join('');

  const rows = sorted.map(g => cols.map(c => {
    const val = g[c.key] ?? 0;
    const cls = c.key === 'points' ? ' class="pts-col"' : '';
    return `<td${cls}>${val}</td>`;
  }).join('')).map(cells => `<tr>${cells}</tr>`).join('');

  return `
    <div class="stats-table-wrap">
      <table id="stats-table">
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function sortGames(games, key, dir) {
  return games.sort((a, b) => {
    const av = a[key] ?? 0;
    const bv = b[key] ?? 0;
    return dir === 'asc' ? av - bv : bv - av;
  });
}

function attachSortHandlers(cols) {
  document.querySelectorAll('#stats-table thead th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (sortKey === key) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = key;
        sortDir = key === 'roundId' ? 'desc' : 'desc';
      }
      // Re-render just the table portion
      const games = currentProfile.results || [];
      const pos = currentPlayer.position;
      const colsToUse = COLS_BY_POS[pos] || [...BASE_COLS, { label: 'Pts', key: 'points', title: 'Total Points' }];
      const wrap = document.querySelector('.stats-table-wrap');
      wrap.outerHTML = buildTable(games, colsToUse);
      attachSortHandlers(colsToUse);
    });
  });
}

loadPlayers();
