const statusEl = document.getElementById('status');
const picksContainer = document.getElementById('picks-container');

const VALID_FORMATIONS = [
  { gk: 1, def: 2, mid: 2, fwd: 2, name: '1-2-2-2' },
  { gk: 1, def: 2, mid: 3, fwd: 1, name: '1-2-3-1' },
  { gk: 1, def: 3, mid: 2, fwd: 1, name: '1-3-2-1' },
];

// Module-level state for re-solving without re-fetching
let allPlayers = [];
let nextRound = null;
let squadsMap = {};
let fixtureCount = {};

let filters = {
  excludeInjured: true,
  min1000mins: true,
};

async function loadPicks() {
  statusEl.textContent = 'Loading...';
  picksContainer.innerHTML = '';

  try {
    const [playersRes, roundsRes, squadsRes] = await Promise.all([
      fetch('/api/players'),
      fetch('/api/rounds'),
      fetch('/api/squads'),
    ]);

    if (!playersRes.ok || !roundsRes.ok || !squadsRes.ok) {
      throw new Error('Failed to fetch required data');
    }

    allPlayers = await playersRes.json();
    const rounds = await roundsRes.json();
    const squads = await squadsRes.json();

    squadsMap = Object.fromEntries(squads.map(s => [s.id, s]));

    // Find next gameweek
    const now = new Date();
    nextRound = null;
    for (const round of rounds) {
      const lockoutDate = new Date(round.lockoutDate);
      if (lockoutDate > now) {
        nextRound = round;
        break;
      }
    }
    if (!nextRound) {
      nextRound = rounds[rounds.length - 1];
    }

    // Build fixture count map
    fixtureCount = {};
    for (const squad of squads) {
      fixtureCount[squad.id] = 0;
    }
    for (const game of nextRound.games) {
      fixtureCount[game.homeId] = (fixtureCount[game.homeId] || 0) + 1;
      fixtureCount[game.awayId] = (fixtureCount[game.awayId] || 0) + 1;
    }

    // Enrich players with projection
    enrichPlayers(allPlayers);

    // Solve and render
    renderWithFilters();
    statusEl.textContent = '';
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

function enrichPlayers(players) {
  for (const p of players) {
    const fixtures = fixtureCount[p.squadId] || 0;
    const multiplier = fixtures >= 2 ? 2 : fixtures === 1 ? 1 : 0;
    p.fixtures = fixtures;
    p.projectedPts = (p.averagePoints || 0) * multiplier;
  }
}

function renderWithFilters() {
  const optimalTeam = solveOptimalTeam(allPlayers, filters);

  if (!optimalTeam) {
    picksContainer.innerHTML =
      '<p class="error-msg">Could not find valid team with current filters.</p>';
    return;
  }

  renderPicks(nextRound, optimalTeam, squadsMap);
}

function solveOptimalTeam(players, filters) {
  // Try each formation, pick the best result
  let bestTeam = null;
  let bestScore = -Infinity;

  for (const formation of VALID_FORMATIONS) {
    const team = solveForFormation(players, formation, filters);
    if (team && team.totalPts > bestScore) {
      bestScore = team.totalPts;
      bestTeam = team;
    }
  }

  return bestTeam;
}

function solveForFormation(players, formation, filters) {
  // Filter eligible players
  const eligible = players.filter(p => {
    if (filters.excludeInjured && p.injuryDetails) return false;
    if (filters.min1000mins && (p.appearances * 90 < 1000)) return false;
    return true;
  });

  // Sort by projectedPts descending
  const sorted = eligible.sort((a, b) => b.projectedPts - a.projectedPts);

  // Greedy pick with constraints
  const team = [];
  const squadCount = {};
  const posCount = { GK: 0, DEF: 0, MID: 0, FWD: 0 };

  for (const player of sorted) {
    const pos = player.position;

    // Check position cap
    if (posCount[pos] >= formation[pos.toLowerCase()]) continue;

    // Check squad cap (max 2 per squad)
    if ((squadCount[player.squadId] || 0) >= 2) continue;

    team.push(player);
    squadCount[player.squadId] = (squadCount[player.squadId] || 0) + 1;
    posCount[pos]++;

    if (team.length === 7) break;
  }

  if (team.length < 7) return null;

  // Assign captain: highest projectedPts in the team
  const captain = team.reduce((max, p) => p.projectedPts > max.projectedPts ? p : max);
  captain.isCaptain = true;
  captain.projectedPtsDisplay = captain.projectedPts * 2;

  const totalPts = team.reduce((s, p) => s + (p.isCaptain ? p.projectedPtsDisplay : p.projectedPts), 0);

  return { team, formation, totalPts, captain };
}

function renderPicks(round, optimalTeam, squads) {
  const { team, formation, totalPts } = optimalTeam;

  const formationStr = `${formation.gk}-${formation.def}-${formation.mid}-${formation.fwd}`;

  let html = `
    <div class="picks-filters">
      <label>
        <input type="checkbox" id="exclude-injured" ${filters.excludeInjured ? 'checked' : ''} />
        Exclude Injured
      </label>
      <label>
        <input type="checkbox" id="min-mins" ${filters.min1000mins ? 'checked' : ''} />
        Min 1000 mins
      </label>
    </div>

    <div class="picks-header">
      <h2>Dexter's Optimal Picks</h2>
      <p class="gw-label">Gameweek ${round.roundNumber}</p>
      <p class="formation-label">Formation ${formationStr} • Projected: <strong>${totalPts.toFixed(1)} pts</strong></p>
    </div>

    <div class="picks-formation">
  `;

  // Group by position for formation display
  const byPos = {
    GK: team.filter(p => p.position === 'GK'),
    DEF: team.filter(p => p.position === 'DEF'),
    MID: team.filter(p => p.position === 'MID'),
    FWD: team.filter(p => p.position === 'FWD'),
  };

  // Render in formation order
  for (const pos of ['GK', 'DEF', 'MID', 'FWD']) {
    if (byPos[pos].length === 0) continue;

    html += `<div class="picks-position-group ${pos.toLowerCase()}-group">`;

    for (const player of byPos[pos]) {
      const name = player.displayName || `${player.firstName} ${player.lastName}`;
      const squad = squads[player.squadId];
      const squadName = squad?.shortName || squad?.name || '?';
      const fixtures = player.fixtures || 0;
      const fixtureLabel = fixtures >= 2 ? `${fixtures}x` : fixtures === 1 ? 'H' : '—';
      const captainClass = player.isCaptain ? 'is-captain' : '';
      const projDisplay = player.isCaptain ? `${player.projectedPtsDisplay.toFixed(1)}*` : player.projectedPts.toFixed(1);

      html += `
        <div class="pick-card pick-${pos} ${captainClass}">
          <div class="pick-header">
            <div class="pick-name">${name}</div>
            <div class="pick-squad">${squadName}</div>
          </div>
          <div class="pick-stats">
            <div class="pick-stat">
              <span class="pick-label">Avg</span>
              <span class="pick-value">${(player.averagePoints || 0).toFixed(2)}</span>
            </div>
            <div class="pick-stat">
              <span class="pick-label">Fix</span>
              <span class="pick-value">${fixtureLabel}</span>
            </div>
            <div class="pick-stat">
              <span class="pick-label">Proj</span>
              <span class="pick-value">${projDisplay}</span>
            </div>
          </div>
        </div>
      `;
    }

    html += '</div>';
  }

  html += '</div>';

  picksContainer.innerHTML = html;

  // Attach filter listeners
  document.getElementById('exclude-injured').addEventListener('change', (e) => {
    filters.excludeInjured = e.target.checked;
    renderWithFilters();
  });

  document.getElementById('min-mins').addEventListener('change', (e) => {
    filters.min1000mins = e.target.checked;
    renderWithFilters();
  });
}

loadPicks();
