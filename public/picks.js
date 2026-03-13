const statusEl = document.getElementById('status');
const picksContainer = document.getElementById('picks-container');

const VALID_FORMATIONS = [
  { gk: 1, def: 2, mid: 2, fwd: 2, name: '1-2-2-2' },
  { gk: 1, def: 2, mid: 3, fwd: 1, name: '1-2-3-1' },
  { gk: 1, def: 3, mid: 2, fwd: 1, name: '1-3-2-1' },
];

// Module-level state
let allPlayers = [];
let nextRound = null;
let squadsMap = {};
let allRounds = [];
let fixturesBySquad = {};

let filters = {
  excludeInjured: true,
  min1000mins: true,
  oneClubChip: false,
  excludeTeams: [],
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
    allRounds = await roundsRes.json();
    const squads = await squadsRes.json();

    squadsMap = Object.fromEntries(squads.map(s => [s.id, s]));

    // Find next gameweek
    const now = new Date();
    nextRound = null;
    for (const round of allRounds) {
      const lockoutDate = new Date(round.lockoutDate);
      if (lockoutDate > now) {
        nextRound = round;
        break;
      }
    }
    if (!nextRound) {
      nextRound = allRounds[allRounds.length - 1];
    }

    // Build fixture map for NEXT GW only
    fixturesBySquad = {};
    for (const squad of squads) {
      fixturesBySquad[squad.id] = [];
    }
    for (const game of nextRound.games) {
      fixturesBySquad[game.homeId]?.push({ ...game, isHome: true });
      fixturesBySquad[game.awayId]?.push({ ...game, isHome: false });
    }

    // Enrich players with per-90 stats by home/away
    enrichPlayers(allPlayers, allRounds, squads);

    renderWithFilters();
    statusEl.textContent = '';
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

function getFixtureDifficulty(squadId) {
  const squad = squadsMap[squadId];
  if (!squad) return 'neutral';

  const pos = squad.leaguePosition || 999;
  if (pos <= 8) return 'hard';    // red: top 8
  if (pos <= 16) return 'medium'; // grey: mid table
  return 'easy';                  // green: bottom 8
}

function enrichPlayers(players, rounds, squads) {
  // Use simple heuristic: estimate per-90 from season average,
  // then assume home games are ~15% better, away ~15% worse
  for (const p of players) {
    const totalMins = p.appearances * 90; // estimate
    const basePer90 = totalMins > 0 ? (p.totalPoints / totalMins) * 90 : (p.averagePoints || 0);

    p.homePer90 = basePer90 * 1.15;  // +15% at home
    p.awayPer90 = basePer90 * 0.85;  // -15% away

    const fixtures = fixturesBySquad[p.squadId] || [];
    p.fixtures = fixtures;

    // Calculate projected pts for next GW
    let projectedPts = 0;
    for (const fixture of fixtures) {
      const per90 = fixture.isHome ? p.homePer90 : p.awayPer90;
      projectedPts += per90;  // Per-90 is already the expected points for a full match
    }

    p.projectedPts = projectedPts;
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
  const eligible = players.filter(p => {
    if (filters.excludeInjured && p.injuryDetails) return false;
    if (filters.min1000mins && (p.appearances * 90 < 1000)) return false;
    if (filters.excludeTeams.includes(p.squadId)) return false;
    return true;
  });

  const sorted = eligible.sort((a, b) => b.projectedPts - a.projectedPts);

  const team = [];
  const squadCount = {};
  const posCount = { GK: 0, DEF: 0, MID: 0, FWD: 0 };

  for (const player of sorted) {
    const pos = player.position;

    if (posCount[pos] >= formation[pos.toLowerCase()]) continue;

    const squadCap = filters.oneClubChip ? Infinity : 2;
    if ((squadCount[player.squadId] || 0) >= squadCap) continue;

    team.push(player);
    squadCount[player.squadId] = (squadCount[player.squadId] || 0) + 1;
    posCount[pos]++;

    if (team.length === 7) break;
  }

  if (team.length < 7) return null;

  const captain = team.reduce((max, p) => p.projectedPts > max.projectedPts ? p : max);
  captain.isCaptain = true;
  captain.projectedPtsDisplay = captain.projectedPts * 2;

  const totalPts = team.reduce((s, p) => s + (p.isCaptain ? p.projectedPtsDisplay : p.projectedPts), 0);

  return { team, formation, totalPts, captain };
}

function renderPicks(round, optimalTeam, squads) {
  const { team, formation, totalPts } = optimalTeam;
  const formationStr = `${formation.gk}-${formation.def}-${formation.mid}-${formation.fwd}`;

  // Build team filter checkboxes
  const teamOptions = Array.from(new Set(allPlayers.map(p => p.squadId)))
    .sort()
    .map(squadId => {
      const squad = squadsMap[squadId];
      const isChecked = !filters.excludeTeams.includes(squadId);
      return `<label style="margin-right: 12px; display: inline-block;"><input type="checkbox" class="team-filter" value="${squadId}" ${isChecked ? 'checked' : ''} /> ${squad?.shortName || squad?.name || '?'}</label>`;
    })
    .join('');

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
      <label>
        <input type="checkbox" id="one-club-chip" ${filters.oneClubChip ? 'checked' : ''} />
        One Club Chip
      </label>
    </div>

    <div style="margin-bottom: 16px; padding: 12px; background: #f5f5f5; border-radius: 6px;">
      <div style="font-size: 0.85rem; color: #666; text-transform: uppercase; font-weight: bold; margin-bottom: 12px;">Include Teams</div>
      <div style="display: flex; flex-wrap: wrap; gap: 12px;">
        ${teamOptions}
      </div>
    </div>

    <div class="picks-header">
      <h2>Dexter's Optimal Picks</h2>
      <p class="gw-label">Gameweek ${round.roundNumber}</p>
      <p class="formation-label">Formation ${formationStr} • Projected: <strong>${totalPts.toFixed(1)} pts</strong></p>
    </div>

    <div class="picks-formation">
  `;

  const byPos = {
    GK: team.filter(p => p.position === 'GK'),
    DEF: team.filter(p => p.position === 'DEF'),
    MID: team.filter(p => p.position === 'MID'),
    FWD: team.filter(p => p.position === 'FWD'),
  };

  for (const pos of ['GK', 'DEF', 'MID', 'FWD']) {
    if (byPos[pos].length === 0) continue;

    html += `<div class="picks-position-group ${pos.toLowerCase()}-group">`;

    for (const player of byPos[pos]) {
      const name = player.displayName || `${player.firstName} ${player.lastName}`;
      const squad = squads[player.squadId];
      const squadName = squad?.shortName || squad?.name || '?';
      const captainClass = player.isCaptain ? 'is-captain' : '';
      const projDisplay = player.isCaptain ? `${player.projectedPtsDisplay.toFixed(1)}*` : player.projectedPts.toFixed(1);

      // Build fixtures display with per-90 values
      let fixturesHtml = '';
      if (player.fixtures && player.fixtures.length > 0) {
        fixturesHtml = '<div class="pick-fixtures">';
        for (const fixture of player.fixtures) {
          const opp = fixture.isHome ? fixture.awayId : fixture.homeId;
          const oppSquad = squads[opp];
          const oppName = oppSquad?.shortName || '?';
          const homeAway = fixture.isHome ? 'H' : 'A';
          const difficulty = getFixtureDifficulty(opp);
          const fixtureBadgeClass = `fixture-${difficulty}`;
          const per90Val = fixture.isHome ? player.homePer90 : player.awayPer90;
          fixturesHtml += `<span class="fixture-badge ${fixtureBadgeClass}">${oppName}(${homeAway})<span class="fixture-per90">${per90Val.toFixed(1)}</span></span>`;
        }
        fixturesHtml += '</div>';
      }

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
              <span class="pick-value">${player.fixtures?.length || 0}</span>
            </div>
            <div class="pick-stat">
              <span class="pick-label">Proj</span>
              <span class="pick-value">${projDisplay}</span>
            </div>
          </div>
          ${fixturesHtml}
        </div>
      `;
    }

    html += '</div>';
  }

  html += '</div>';

  picksContainer.innerHTML = html;

  document.getElementById('exclude-injured').addEventListener('change', (e) => {
    filters.excludeInjured = e.target.checked;
    renderWithFilters();
  });

  document.getElementById('min-mins').addEventListener('change', (e) => {
    filters.min1000mins = e.target.checked;
    renderWithFilters();
  });

  document.getElementById('one-club-chip').addEventListener('change', (e) => {
    filters.oneClubChip = e.target.checked;
    renderWithFilters();
  });

  document.querySelectorAll('.team-filter').forEach(checkbox => {
    checkbox.addEventListener('change', (e) => {
      const squadId = parseInt(e.target.value, 10);
      if (e.target.checked) {
        filters.excludeTeams = filters.excludeTeams.filter(id => id !== squadId);
      } else {
        filters.excludeTeams.push(squadId);
      }
      renderWithFilters();
    });
  });
}

loadPicks();
