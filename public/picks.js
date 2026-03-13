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
  minRecentAvgMins: 0,
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

    // Fetch game data to calculate empirical home/away and recent mins
    await enrichPlayerGameData(allPlayers);

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

// Enrich players with actual game data: recentAvgMins and empirical home/away per-90
async function enrichPlayerGameData(players) {
  // Check if we have fresh cached data (within 24 hours)
  const cacheKey = 'picks_game_data_cache';
  const cacheTimestampKey = 'picks_game_data_timestamp';
  const cacheTimestamp = localStorage.getItem(cacheTimestampKey);
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;

  if (cacheTimestamp && now - parseInt(cacheTimestamp) < oneDayMs) {
    // Use cached data
    const cachedData = JSON.parse(localStorage.getItem(cacheKey) || '{}');
    for (const player of players) {
      if (cachedData[player.id]) {
        const cached = cachedData[player.id];
        player.games = cached.games;
        player.recentAvgMins = cached.recentAvgMins;
        player.homePer90 = cached.homePer90;
        player.awayPer90 = cached.awayPer90;
      }
    }
    console.log('[CACHE] Using cached game data');
    return;
  }

  statusEl.textContent = 'Loading game data...';
  const cachedData = {};
  let loaded = 0;

  try {
    // Only fetch for players who have appeared (to avoid unnecessary API calls)
    const playersToFetch = players.filter(p => p.appearances > 0);
    const batchSize = 3;

    for (let i = 0; i < playersToFetch.length; i += batchSize) {
      const batch = playersToFetch.slice(i, i + batchSize);

      const batchPromises = batch.map(async (player) => {
        try {
          const res = await fetch(`/api/player?id=${player.id}`);
          if (!res.ok) return null;
          const profile = await res.json();

          if (!profile.games || !Array.isArray(profile.games)) return null;

          player.games = profile.games;

          // Calculate recent avg mins (last 5 games)
          const recentGames = profile.games.slice(-5);
          if (recentGames.length > 0) {
            const totalMins = recentGames.reduce((sum, g) => sum + (g.minutes || 0), 0);
            player.recentAvgMins = totalMins / recentGames.length;
          } else {
            player.recentAvgMins = 0;
          }

          // Calculate empirical home/away per-90
          const homeGames = profile.games.filter(g => g.isHome);
          const awayGames = profile.games.filter(g => !g.isHome);

          if (homeGames.length > 0) {
            const homePoints = homeGames.reduce((sum, g) => sum + (g.points || 0), 0);
            const homeMins = homeGames.reduce((sum, g) => sum + (g.minutes || 0), 0);
            player.homePer90 = homeMins > 0 ? (homePoints / homeMins) * 90 : player.homePer90;
          }

          if (awayGames.length > 0) {
            const awayPoints = awayGames.reduce((sum, g) => sum + (g.points || 0), 0);
            const awayMins = awayGames.reduce((sum, g) => sum + (g.minutes || 0), 0);
            player.awayPer90 = awayMins > 0 ? (awayPoints / awayMins) * 90 : player.awayPer90;
          }

          // Cache this player's data
          cachedData[player.id] = {
            games: player.games,
            recentAvgMins: player.recentAvgMins,
            homePer90: player.homePer90,
            awayPer90: player.awayPer90
          };

          return true;
        } catch (err) {
          console.warn(`Could not fetch game data for player ${player.id}: ${err.message}`);
          return null;
        }
      });

      await Promise.all(batchPromises);

      loaded += batch.length;
      statusEl.textContent = `Loading game data: ${loaded}/${playersToFetch.length}`;
    }

    // Log results
    const numWithData = Object.keys(cachedData).length;
    const numFiltered = playersToFetch.filter(p => p.recentAvgMins > 0).length;
    console.log(`[GAME-DATA] Fetched data for ${numWithData}/${playersToFetch.length} players. ${numFiltered} have recentAvgMins calculated.`);

    // Store cache if we got some data
    if (numWithData > 0) {
      try {
        localStorage.setItem(cacheKey, JSON.stringify(cachedData));
        localStorage.setItem(cacheTimestampKey, now.toString());
        console.log(`[CACHE] Saved game data for ${numWithData} players`);
      } catch (err) {
        console.warn('Could not save cache to localStorage:', err.message);
      }
    }
  } catch (err) {
    console.warn('Error during game data fetch:', err.message);
  }
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

    // Initialize recentAvgMins (will be overwritten if we fetch game data)
    p.recentAvgMins = (p.appearances || 0) > 0 ? 60 : 0;

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
    // Count eligible players to provide helpful feedback
    const eligible = allPlayers.filter(p => {
      if (filters.excludeInjured && p.injuryDetails) return false;
      if (filters.min1000mins && (p.appearances * 90 < 1000)) return false;
      if (filters.excludeTeams.includes(p.squadId)) return false;
      if (filters.minRecentAvgMins > 0 && p.recentAvgMins < filters.minRecentAvgMins) return false;
      return true;
    });

    const msg = eligible.length === 0
      ? '<p class="error-msg">No players available with current filters. Try adjusting your selections.</p>'
      : '<p class="error-msg">Could not form a valid team with current filters. Try adjusting your selections.</p>';
    picksContainer.innerHTML = msg;
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
    if (filters.minRecentAvgMins > 0 && p.recentAvgMins < filters.minRecentAvgMins) return false;
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
    .sort((a, b) => {
      const squadA = squadsMap[a];
      const squadB = squadsMap[b];
      const nameA = (squadA?.shortName || squadA?.name || '').toUpperCase();
      const nameB = (squadB?.shortName || squadB?.name || '').toUpperCase();
      return nameA.localeCompare(nameB);
    })
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
      <div style="margin-top: 12px;">
        <label style="display: block; margin-bottom: 6px;">
          Min Recent Avg Mins: <strong id="recent-mins-value">${filters.minRecentAvgMins}</strong>
        </label>
        <input type="range" id="recent-mins-slider" min="0" max="90" value="${filters.minRecentAvgMins}" style="width: 200px;" />
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

      const hasGames = player.games && player.games.length > 0;
      const infoButtonHtml = hasGames ? `<button class="pick-info-btn" data-player-id="${player.id}" title="View last 5 games">ℹ️</button>` : '';

      html += `
        <div class="pick-card pick-${pos} ${captainClass}">
          <div class="pick-header">
            <div class="pick-name">${name} ${infoButtonHtml}</div>
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

  html += '</div>'; // close picks-formation

  // Build Teams to Target section
  // Group squads by league - find all unique league values first
  const leagueMap = {};
  for (const squad of Object.values(squadsMap)) {
    // Try multiple possible league property names
    const league = squad.league || squad.divisionName || squad.leagueName || squad.division || 'Unknown';
    if (!leagueMap[league]) {
      leagueMap[league] = [];
    }
    leagueMap[league].push(squad);
  }

  // Sort squads by name within each league
  for (const league of Object.keys(leagueMap)) {
    leagueMap[league].sort((a, b) => (a.shortName || a.name).localeCompare(b.shortName || b.name));
  }

  let teamsHtml = '<div style="margin-top: 40px; padding-top: 24px; border-top: 2px solid #ddd;">';
  teamsHtml += '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">';
  teamsHtml += '<h3 style="margin: 0; font-size: 1.2rem;">Teams to Target</h3>';
  teamsHtml += '<div style="display: flex; gap: 8px;">';
  teamsHtml += '<button id="include-all-teams" style="padding: 6px 14px; font-size: 0.8rem; background: #f05a28; color: white; border: none; border-radius: 4px; cursor: pointer;">Include All</button>';
  teamsHtml += '<button id="exclude-all-teams" style="padding: 6px 14px; font-size: 0.8rem; background: #999; color: white; border: none; border-radius: 4px; cursor: pointer;">Exclude All</button>';
  teamsHtml += '</div></div>';
  teamsHtml += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px;">';

  // Get sorted league names
  const sortedLeagues = Object.keys(leagueMap).sort();
  for (const league of sortedLeagues) {
    const squads = leagueMap[league];

    teamsHtml += `<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px;">`;
    teamsHtml += `<h4 style="margin-bottom: 12px; color: #f05a28; font-size: 0.95rem;">${league}</h4>`;

    for (const squad of squads) {
      const fixtures = fixturesBySquad[squad.id] || [];
      const fixtureCount = fixtures.length;
      const isExcluded = filters.excludeTeams.includes(squad.id);

      let fixturesDisplay = '';
      for (const fixture of fixtures) {
        const opp = fixture.isHome ? fixture.awayId : fixture.homeId;
        const oppSquad = squadsMap[opp];
        const oppName = oppSquad?.shortName || '?';
        const difficulty = getFixtureDifficulty(opp);
        const fixtureBadgeClass = `fixture-${difficulty}`;
        fixturesDisplay += `<span class="fixture-badge ${fixtureBadgeClass}" style="margin-right: 4px; display: inline-block;">${oppName}</span>`;
      }

      const label = fixtureCount === 2 ? '2️⃣ ' : '';
      teamsHtml += `
        <div style="margin-bottom: 12px; padding: 8px; background: ${isExcluded ? '#f0f0f0' : '#fafafa'}; border-radius: 4px; cursor: pointer;" class="team-target-row" data-squad-id="${squad.id}">
          <div style="font-weight: bold; margin-bottom: 4px; ${isExcluded ? 'opacity: 0.5;' : ''}">${label}${squad.shortName || squad.name}</div>
          <div style="font-size: 0.8rem; ${isExcluded ? 'opacity: 0.5;' : ''}">${fixturesDisplay}</div>
        </div>
      `;
    }

    teamsHtml += '</div>';
  }

  teamsHtml += '</div></div>';
  html += teamsHtml;

  html += `
    <div id="games-modal" class="games-modal" style="display: none;">
      <div class="games-modal-content">
        <span class="games-modal-close">&times;</span>
        <h3 id="games-modal-title"></h3>
        <table class="games-table">
          <thead>
            <tr>
              <th>GW</th>
              <th>Minutes</th>
              <th>Points</th>
            </tr>
          </thead>
          <tbody id="games-table-body">
          </tbody>
        </table>
      </div>
    </div>
  `;

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

  document.getElementById('recent-mins-slider').addEventListener('input', (e) => {
    filters.minRecentAvgMins = parseInt(e.target.value, 10);
    document.getElementById('recent-mins-value').textContent = filters.minRecentAvgMins;
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

  // Include/Exclude All buttons
  document.getElementById('include-all-teams').addEventListener('click', () => {
    filters.excludeTeams = [];
    document.querySelectorAll('.team-filter').forEach(cb => { cb.checked = true; });
    renderWithFilters();
  });

  document.getElementById('exclude-all-teams').addEventListener('click', () => {
    const allTeamIds = Array.from(document.querySelectorAll('.team-filter')).map(cb => parseInt(cb.value, 10));
    filters.excludeTeams = allTeamIds;
    document.querySelectorAll('.team-filter').forEach(cb => { cb.checked = false; });
    renderWithFilters();
  });

  // Team target rows (click to toggle)
  document.querySelectorAll('.team-target-row').forEach(row => {
    row.addEventListener('click', () => {
      const squadId = parseInt(row.getAttribute('data-squad-id'), 10);
      if (filters.excludeTeams.includes(squadId)) {
        filters.excludeTeams = filters.excludeTeams.filter(id => id !== squadId);
      } else {
        filters.excludeTeams.push(squadId);
      }
      // Also toggle the corresponding checkbox
      const checkbox = document.querySelector(`.team-filter[value="${squadId}"]`);
      if (checkbox) {
        checkbox.checked = !checkbox.checked;
      }
      renderWithFilters();
    });
  });

  // Modal functionality
  const modal = document.getElementById('games-modal');
  const closeBtn = document.querySelector('.games-modal-close');

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.style.display = 'none';
    }
  });

  document.querySelectorAll('.pick-info-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const playerId = parseInt(btn.getAttribute('data-player-id'), 10);
      const player = allPlayers.find(p => p.id === playerId);

      if (player && player.games && player.games.length > 0) {
        const recentGames = player.games.slice(-5).reverse(); // Last 5, most recent first
        const playerName = player.displayName || `${player.firstName} ${player.lastName}`;
        document.getElementById('games-modal-title').textContent = `${playerName} - Last 5 Games`;

        const tableBody = document.getElementById('games-table-body');
        tableBody.innerHTML = recentGames.map((game, idx) => `
          <tr>
            <td>${game.round || game.roundId || game.roundNumber || 'N/A'}</td>
            <td>${game.minutes || 0}</td>
            <td>${game.points || 0}</td>
          </tr>
        `).join('');

        modal.style.display = 'block';
      }
    });
  });
}

loadPicks();
