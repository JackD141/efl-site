const statusEl = document.getElementById('status');
const picksContainer = document.getElementById('picks-container');

const VALID_FORMATIONS = [
  { gk: 1, def: 2, mid: 2, fwd: 2, name: '1-2-2-2' },
  { gk: 1, def: 2, mid: 3, fwd: 1, name: '1-2-3-1' },
  { gk: 1, def: 3, mid: 2, fwd: 1, name: '1-3-2-1' },
];

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

    const players = await playersRes.json();
    const rounds = await roundsRes.json();
    const squads = await squadsRes.json();

    // Find next gameweek (first with status "scheduled", or current/latest if all completed)
    const now = new Date();
    let nextRound = null;
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

    // Build a map: squadId -> fixture count in nextRound
    const fixtureCount = {};
    for (const squad of squads) {
      fixtureCount[squad.id] = 0;
    }
    for (const game of nextRound.games) {
      fixtureCount[game.homeId] = (fixtureCount[game.homeId] || 0) + 1;
      fixtureCount[game.awayId] = (fixtureCount[game.awayId] || 0) + 1;
    }

    // Calculate projected pts for next GW
    const playerPicks = players
      .filter(p => p.averagePoints && p.averagePoints > 0)
      .map(p => {
        const fixtures = fixtureCount[p.squadId] || 0;
        // If 2 fixtures, double the pts/game. If 1, use as-is. If 0, use base.
        const projectionMultiplier = fixtures >= 2 ? 2 : fixtures === 1 ? 1 : 0.5;
        const projectedPts = p.averagePoints * projectionMultiplier;
        return {
          ...p,
          fixtures,
          projectedPts,
        };
      });

    // Solve for optimal team
    const optimalTeam = solveOptimalTeam(playerPicks);

    if (!optimalTeam) {
      statusEl.textContent = 'Could not find valid team.';
      return;
    }

    renderPicks(nextRound, optimalTeam, squads);
    statusEl.textContent = '';
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

function solveOptimalTeam(players) {
  // Simple greedy approach: pick best players per position until formation is valid
  const byPos = {
    GK: players.filter(p => p.position === 'GK').sort((a, b) => b.projectedPts - a.projectedPts),
    DEF: players.filter(p => p.position === 'DEF').sort((a, b) => b.projectedPts - a.projectedPts),
    MID: players.filter(p => p.position === 'MID').sort((a, b) => b.projectedPts - a.projectedPts),
    FWD: players.filter(p => p.position === 'FWD').sort((a, b) => b.projectedPts - a.projectedPts),
  };

  // Try each formation
  let bestTeam = null;
  let bestScore = -Infinity;

  for (const formation of VALID_FORMATIONS) {
    const team = [
      ...byPos.GK.slice(0, formation.gk),
      ...byPos.DEF.slice(0, formation.def),
      ...byPos.MID.slice(0, formation.mid),
      ...byPos.FWD.slice(0, formation.fwd),
    ];

    if (team.length === 7) {
      const score = team.reduce((s, p) => s + p.projectedPts, 0);
      if (score > bestScore) {
        bestScore = score;
        bestTeam = { team, formation };
      }
    }
  }

  return bestTeam;
}

function renderPicks(round, optimalTeam, squads) {
  const { team, formation } = optimalTeam;
  const squadMap = Object.fromEntries(squads.map(s => [s.id, s]));

  const formationStr = `${formation.gk}-${formation.def}-${formation.mid}-${formation.fwd}`;
  const totalProjectedPts = team.reduce((s, p) => s + p.projectedPts, 0);

  let html = `
    <div class="picks-header">
      <h2>Dexter's Optimal Picks</h2>
      <p class="gw-label">Gameweek ${round.roundNumber}</p>
      <p class="formation-label">Formation ${formationStr} • Projected: <strong>${totalProjectedPts.toFixed(1)} pts</strong></p>
    </div>

    <div class="picks-grid">
  `;

  // Group by position for display
  const byPos = {
    GK: team.filter(p => p.position === 'GK'),
    DEF: team.filter(p => p.position === 'DEF'),
    MID: team.filter(p => p.position === 'MID'),
    FWD: team.filter(p => p.position === 'FWD'),
  };

  for (const pos of ['GK', 'DEF', 'MID', 'FWD']) {
    for (const player of byPos[pos]) {
      const name = player.displayName || `${player.firstName} ${player.lastName}`;
      const squad = squadMap[player.squadId];
      const squadName = squad?.shortName || squad?.name || '?';
      const fixtures = player.fixtures || 0;
      const fixtureLabel = fixtures >= 2 ? `${fixtures}x` : fixtures === 1 ? 'H' : '—';
      html += `
        <div class="pick-card pick-${pos}">
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
              <span class="pick-label">Fixtures</span>
              <span class="pick-value">${fixtureLabel}</span>
            </div>
            <div class="pick-stat">
              <span class="pick-label">Proj</span>
              <span class="pick-value">${player.projectedPts.toFixed(1)}</span>
            </div>
          </div>
        </div>
      `;
    }
  }

  html += '</div>';

  picksContainer.innerHTML = html;
}

loadPicks();
