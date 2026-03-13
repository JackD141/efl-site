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

    // Enrich players with actual per-90 stats by home/away
    await enrichPlayers(allPlayers, allRounds, squads);

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

// Build a map of squad to games played (to match player games to home/away)
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

async function enrichPlayers(players, rounds, squads) {
  const gamesByRound = buildGamesByRound(rounds);

  // Fetch profiles with timeout for all players in parallel batches
  const fetchProfileWithTimeout = async (playerId) => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000); // 3 second timeout per player

      const profileRes = await fetch(`/api/player?id=${playerId}`, { signal: controller.signal });
      clearTimeout(timeout);

      if (!profileRes.ok) return null;
      return await profileRes.json();
    } catch (err) {
      return null;
    }
  };

  // Fetch profiles in batches of 5 to avoid overwhelming the server
  const batchSize = 5;
  const profiles = {};

  for (let i = 0; i < players.length; i += batchSize) {
    const batch = players.slice(i, i + batchSize);
    const promises = batch.map(p => fetchProfileWithTimeout(p.id).then(profile => ({ id: p.id, profile })));
    const results = await Promise.all(promises);

    for (const result of results) {
      if (result.profile) {
        profiles[result.id] = result.profile;
      }
    }
  }

  // Process each player
  for (const player of players) {
    let homeMins = 0, homePts = 0, awayMins = 0, awayPts = 0;
    const recentGames = [];
    const profile = profiles[player.id];
    const games = profile ? (profile.results || profile.games || []) : [];

    if (games.length > 0) {
      // Match each game to determine home/away
      for (const game of games) {
        const roundNum = game.round || game.roundId || game.roundNumber;
        const gameInfo = gamesByRound[roundNum]?.[player.squadId];

        if (gameInfo) {
          const mins = game.minutes || game.minutesPlayed || 0;
          const pts = game.points || 0;

          if (gameInfo.isHome) {
            homeMins += mins;
            homePts += pts;
          } else {
            awayMins += mins;
            awayPts += pts;
          }
        }

        // Store for last 5 games popup
        recentGames.push({
          round: roundNum,
          minutes: game.minutes || game.minutesPlayed || 0,
          points: game.points || 0,
        });
      }

      // Calculate per-90 values
      player.homePer90 = homeMins > 0 ? (homePts / homeMins) * 90 : 0;
      player.awayPer90 = awayMins > 0 ? (awayPts / awayMins) * 90 : 0;

      // Fallback: if no games matched home/away, use overall per-90
      if (player.homePer90 === 0 && player.awayPer90 === 0 && games.length > 0) {
        const totalMins = games.reduce((s, g) => s + (g.minutes || g.minutesPlayed || 0), 0);
        const totalPts = games.reduce((s, g) => s + (g.points || 0), 0);
        const overallPer90 = totalMins > 0 ? (totalPts / totalMins) * 90 : 0;
        player.homePer90 = overallPer90;
        player.awayPer90 = overallPer90;
      }

      player.recentGames = recentGames.slice(-5);
    } else {
      // No profile data - use sensible defaults
      player.homePer90 = player.averagePoints || 0;
      player.awayPer90 = player.averagePoints || 0;
      player.recentGames = [];
    }

    // Get next GW fixtures and calculate projection
    const fixtures = fixturesBySquad[player.squadId] || [];
    player.fixtures = fixtures;

    let projectedPts = 0;
    for (const fixture of fixtures) {
      const per90 = fixture.isHome ? player.homePer90 : player.awayPer90;
      projectedPts += per90 / 90;
    }

    player.projectedPts = projectedPts;
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

function showLastGamesPopup(player) {
  const games = player.recentGames || [];
  if (games.length === 0) {
    alert(`No recent game data for ${player.firstName} ${player.lastName}`);
    return;
  }

  let tableHtml = '<table style="width: 100%; border-collapse: collapse;"><tr><th style="border: 1px solid #ddd; padding: 8px;">GW</th><th style="border: 1px solid #ddd; padding: 8px;">Mins</th><th style="border: 1px solid #ddd; padding: 8px;">Pts</th></tr>';
  for (const game of games) {
    tableHtml += `<tr><td style="border: 1px solid #ddd; padding: 8px;">${game.round}</td><td style="border: 1px solid #ddd; padding: 8px;">${game.minutes}</td><td style="border: 1px solid #ddd; padding: 8px;">${game.points}</td></tr>`;
  }
  tableHtml += '</table>';

  const popup = document.createElement('div');
  popup.style.position = 'fixed';
  popup.style.top = '50%';
  popup.style.left = '50%';
  popup.style.transform = 'translate(-50%, -50%)';
  popup.style.background = '#fff';
  popup.style.padding = '20px';
  popup.style.borderRadius = '8px';
  popup.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
  popup.style.zIndex = '1000';
  popup.style.maxWidth = '400px';
  popup.innerHTML = `<h3>${player.firstName} ${player.lastName} - Last 5 Games</h3>${tableHtml}<button onclick="this.parentElement.remove()" style="margin-top: 12px; padding: 8px 16px; background: #f05a28; color: #fff; border: none; border-radius: 4px; cursor: pointer;">Close</button>`;
  document.body.appendChild(popup);

  const backdrop = document.createElement('div');
  backdrop.style.position = 'fixed';
  backdrop.style.top = '0';
  backdrop.style.left = '0';
  backdrop.style.width = '100%';
  backdrop.style.height = '100%';
  backdrop.style.background = 'rgba(0,0,0,0.4)';
  backdrop.style.zIndex = '999';
  backdrop.onclick = () => backdrop.remove() || popup.remove();
  document.body.appendChild(backdrop);
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
      <label>
        <input type="checkbox" id="one-club-chip" ${filters.oneClubChip ? 'checked' : ''} />
        One Club Chip
      </label>
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
          <button class="pick-info-btn" onclick="showLastGamesPopup(${JSON.stringify(player).replace(/"/g, '&quot;')})" style="margin-top: 8px; padding: 4px 8px; background: #f05a28; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.75rem; width: 100%;">ℹ Last 5 Games</button>
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
}

loadPicks();
