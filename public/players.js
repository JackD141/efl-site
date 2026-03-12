const PLAYERS_URL = 'https://fantasy.efl.com/json/fantasy/players.json';
const PROFILE_URL = id => `https://fantasy.efl.com/json/fantasy/player_profiles/${id}.json`;

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const playerStats = document.getElementById('player-stats');

let allPlayers = [];
const profileCache = {};

// Load player list on page open
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

async function showPlayer(player) {
  const name = player.displayName || `${player.firstName} ${player.lastName}`;
  playerStats.innerHTML = `<p class="loading-msg">Loading stats for ${name}...</p>`;
  searchResults.innerHTML = '';
  searchInput.value = name;

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

function renderPlayerStats(player, profile) {
  const name = player.displayName || `${player.firstName} ${player.lastName}`;
  const games = profile.results || [];

  if (games.length === 0) {
    playerStats.innerHTML = `<p class="loading-msg">No game data found for ${name}.</p>`;
    return;
  }

  // Season summary from the player list entry
  const summaryHtml = `
    <div class="player-header">
      <h2>${name}</h2>
      <span class="player-pos">${player.position}</span>
    </div>
    <div class="season-summary">
      <div class="stat-pill">Total Pts <strong>${player.totalPoints}</strong></div>
      <div class="stat-pill">Avg Pts <strong>${player.averagePoints?.toFixed(1) ?? '—'}</strong></div>
      <div class="stat-pill">Apps <strong>${player.appearances}</strong></div>
      <div class="stat-pill">Goals <strong>${player.goalsScored}</strong></div>
      <div class="stat-pill">Assists <strong>${player.assists}</strong></div>
      <div class="stat-pill">Clean Sheets <strong>${player.cleanSheets}</strong></div>
    </div>
  `;

  const cols = [
    { label: 'GW',      key: 'roundId' },
    { label: 'Mins',    key: 'minutesPlayed' },
    { label: 'Goals',   key: 'goalsScored' },
    { label: 'Assists', key: 'assists' },
    { label: 'CS',      key: 'cleanSheet' },
    { label: 'GC',      key: 'goalsConceded' },
    { label: 'Saves',   key: 'saves' },
    { label: 'YC',      key: 'yellowCards' },
    { label: 'RC',      key: 'redCards' },
    { label: 'KP',      key: 'keyPasses' },
    { label: 'SOT',     key: 'shotsOnTarget' },
    { label: 'Pts',     key: 'points' },
  ];

  const rows = games.map(g => `
    <tr>
      ${cols.map(c => {
        const val = g[c.key] ?? 0;
        const cls = c.key === 'points' ? ' class="pts-col"' : '';
        return `<td${cls}>${val}</td>`;
      }).join('')}
    </tr>
  `).join('');

  playerStats.innerHTML = summaryHtml + `
    <div class="stats-table-wrap">
      <table>
        <thead>
          <tr>${cols.map(c => `<th>${c.label}</th>`).join('')}</tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

loadPlayers();
