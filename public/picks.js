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
  excludePlayers: [],
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

    // Log squad structure for debugging
    if (squads.length > 0) {
      console.log('[SQUAD-STRUCTURE] Sample squad:', squads[0]);
      console.log('[SQUAD-KEYS] Available properties:', Object.keys(squads[0]));
      // Log all teams with all relevant properties
      for (const s of squads.slice(0, 5)) {
        console.log(`${s.shortName}:`, { league: s.league, division: s.division, leagueName: s.leagueName, divisionName: s.divisionName, leaguePosition: s.leaguePosition });
      }
    }

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

    // Build fixture lookup for home/away determination (same as players.js)
    const gamesByRound = {};
    for (const round of allRounds) {
      gamesByRound[round.roundId] = {};
      for (const game of round.games) {
        gamesByRound[round.roundId][game.homeId] = { ...game, isHome: true };
        gamesByRound[round.roundId][game.awayId] = { ...game, isHome: false };
      }
    }

    // Fetch game data to calculate empirical home/away and recent mins
    await enrichPlayerGameData(allPlayers, gamesByRound, squadsMap);

    // Enrich players with fixtures and projected pts (uses empirical home/away per-90)
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

// Enrich players with actual game data: recentAvgMins and empirical home/away per-90
async function enrichPlayerGameData(players, gamesByRound, squadsMap) {
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
    const batchSize = 50;

    for (let i = 0; i < playersToFetch.length; i += batchSize) {
      const batch = playersToFetch.slice(i, i + batchSize);

      const batchPromises = batch.map(async (player) => {
        try {
          const res = await fetch(`/api/player?id=${player.id}`);
          if (!res.ok) return null;
          const profile = await res.json();

          // Games are in 'results' field, not 'games'
          const games = profile.results || [];
          if (!Array.isArray(games) || games.length === 0) return null;

          player.games = games;

          // Calculate recent avg mins (last 5 games)
          const recentGames = games.slice(-5);
          if (recentGames.length > 0) {
            const totalMins = recentGames.reduce((sum, g) => sum + (g.minutes || 0), 0);
            player.recentAvgMins = totalMins / recentGames.length;
          } else {
            player.recentAvgMins = 0;
          }

          // Calculate empirical home/away per-90 using fixture schedule
          let homeMins = 0, homePts = 0, awayMins = 0, awayPts = 0;
          let homeCount = 0, awayCount = 0;

          for (const game of games) {
            const roundNum = game.round || game.roundId || game.roundNumber;
            const gameInfo = gamesByRound[roundNum]?.[player.squadId];
            const mins = game.minutesPlayed || game.minutes || 0;
            const pts = game.points || 0;

            if (gameInfo && gameInfo.isHome) {
              homeMins += mins;
              homePts += pts;
              homeCount++;
            } else if (gameInfo && !gameInfo.isHome) {
              awayMins += mins;
              awayPts += pts;
              awayCount++;
            }
          }

          console.log(`[GAME-DATA] Player ${player.id}: ${games.length} games, ${homeCount} home, ${awayCount} away`);

          if (homeMins > 0) {
            player.homePer90 = (homePts / homeMins) * 90;
          }

          if (awayMins > 0) {
            player.awayPer90 = (awayPts / awayMins) * 90;
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
  // Use empirical home/away per-90 from actual game data if available
  // (already calculated in enrichPlayerGameData from each player's game history)
  for (const p of players) {
    const totalMins = p.appearances * 90; // estimate
    const basePer90 = totalMins > 0 ? (p.totalPoints / totalMins) * 90 : (p.averagePoints || 0);

    // Only set homePer90/awayPer90 if not already populated from game data
    if (!p.homePer90) {
      p.homePer90 = basePer90;
    }
    if (!p.awayPer90) {
      p.awayPer90 = basePer90;
    }

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
  renderPicks(nextRound, optimalTeam, squadsMap);
}

function solveOptimalTeam(players, filters) {
  // Clear any existing captain flags
  for (const p of players) {
    delete p.isCaptain;
    delete p.projectedPtsDisplay;
  }

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
    if (filters.excludePlayers.includes(p.id)) return false;
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
      <div style="margin-top: 16px; display: flex; gap: 8px;">
        <button id="save-filters-btn" style="padding: 6px 12px; background: #f05a28; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Save Filters</button>
        <button id="load-filters-btn" style="padding: 6px 12px; background: #666; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">Load Filters</button>
      </div>
    </div>

    <div class="picks-header">
      <div style="display: flex; align-items: center; gap: 12px;">
        <h2 style="margin: 0;">Certified™ Optimal Picks</h2>
        <button id="methodology-btn" style="background: none; border: none; font-size: 1.2rem; cursor: pointer; color: #999; padding: 0;" title="View methodology">ⓘ</button>
      </div>
      <p class="gw-label">Gameweek ${round.roundNumber}</p>
  `;

  if (optimalTeam) {
    const { team, formation, totalPts } = optimalTeam;
    const formationStr = `${formation.gk}-${formation.def}-${formation.mid}-${formation.fwd}`;
    html += `<p class="formation-label">Formation ${formationStr} • Projected: <strong>${totalPts.toFixed(1)} pts</strong></p>`;
    html += '</div>';

    html += `<div style="display: flex; gap: 20px; margin-top: 20px;">`;
    html += `<div class="picks-formation" style="flex: 1;">`;

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
        const excludeButtonHtml = `<button class="pick-exclude-btn" data-player-id="${player.id}" title="Exclude this player">✕</button>`;

        html += `
          <div class="pick-card pick-${pos} ${captainClass}">
            <div class="pick-header">
              <div class="pick-name">${name} ${infoButtonHtml} ${excludeButtonHtml}</div>
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

    // Build excluded players list on the right
    const excludedPlayerIds = filters.excludePlayers;
    const excludedPlayers = allPlayers.filter(p => excludedPlayerIds.includes(p.id));

    html += `<div style="flex: 0 0 250px; padding: 12px; background: #f5f5f5; border-radius: 4px;">`;
    html += `<h4 style="margin: 0 0 12px 0; font-size: 0.9rem;">Excluded Players (${excludedPlayers.length})</h4>`;
    if (excludedPlayers.length === 0) {
      html += `<p style="margin: 0; font-size: 0.85rem; color: #999;">None excluded</p>`;
    } else {
      html += `<div style="display: flex; flex-direction: column; gap: 6px;">`;
      for (const p of excludedPlayers) {
        const pName = p.displayName || `${p.firstName} ${p.lastName}`;
        const pSquad = squads[p.squadId];
        const pSquadName = pSquad?.shortName || pSquad?.name || '?';
        html += `
          <label style="display: flex; align-items: center; gap: 6px; font-size: 0.85rem; cursor: pointer; padding: 4px; border-radius: 3px; background: white;">
            <input type="checkbox" class="exclude-player-checkbox" value="${p.id}" style="cursor: pointer;" />
            <span>${pName}</span>
            <span style="font-size: 0.75rem; color: #999;">(${pSquadName})</span>
          </label>
        `;
      }
      html += '</div>';
    }
    html += '</div>';

    html += '</div>'; // close flex container
  } else {
    html += '</div>';
    html += '<div class="picks-formation" style="padding: 20px; text-align: center; color: #666;">';
    html += '<p>No team can be formed with the current filters. Adjust filters or select different teams.</p>';
    html += '</div>';
  }

  // Build Teams to Target section
  // Calculate fixture score for each team and group by league
  const leagueGroups = {};

  for (const squad of Object.values(squadsMap)) {
    const fixtures = fixturesBySquad[squad.id] || [];
    let fixtureScore = 0;

    for (const fixture of fixtures) {
      const opp = fixture.isHome ? fixture.awayId : fixture.homeId;
      const difficulty = getFixtureDifficulty(opp);

      let difficultyScore = 0;
      if (difficulty === 'easy') difficultyScore = 2;
      else if (difficulty === 'medium') difficultyScore = 1;
      else if (difficulty === 'hard') difficultyScore = 0;

      const homeBonus = fixture.isHome ? 1 : 0;
      fixtureScore += difficultyScore + homeBonus;
    }

    // Determine league from competitionId
    let league = 'Unknown';
    if (squad.competitionId === 10) league = 'Championship';
    else if (squad.competitionId === 11) league = 'League 1';
    else if (squad.competitionId === 12) league = 'League 2';

    if (!leagueGroups[league]) {
      leagueGroups[league] = [];
    }
    leagueGroups[league].push({ squad, fixtures, fixtureScore });
  }

  // Sort teams within each league by fixture count first, then fixture score
  for (const league of Object.keys(leagueGroups)) {
    leagueGroups[league].sort((a, b) => {
      if (b.fixtures.length !== a.fixtures.length) return b.fixtures.length - a.fixtures.length;
      if (b.fixtureScore !== a.fixtureScore) return b.fixtureScore - a.fixtureScore;
      return (a.squad.shortName || a.squad.name).localeCompare(b.squad.shortName || b.squad.name);
    });
  }

  // Sort leagues in a sensible order (Championship first, then League 1, etc)
  const leagueOrder = ['Championship', 'League 1', 'League 2'];
  const sortedLeagues = Object.keys(leagueGroups).sort((a, b) => {
    const aIdx = leagueOrder.indexOf(a);
    const bIdx = leagueOrder.indexOf(b);
    if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
    if (aIdx !== -1) return -1;
    if (bIdx !== -1) return 1;
    return a.localeCompare(b);
  });

  let teamsHtml = '<div style="margin-top: 40px; padding-top: 24px; border-top: 2px solid #ddd;">';
  teamsHtml += '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">';
  teamsHtml += '<h3 style="margin: 0; font-size: 1.2rem;">Teams to Target</h3>';
  teamsHtml += '<label style="display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 0.9rem;"><input type="checkbox" id="toggle-all-teams" style="width: 18px; height: 18px; cursor: pointer;" /> Toggle All</label>';
  teamsHtml += '</div>';
  teamsHtml += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">';

  for (const league of sortedLeagues) {
    const teams = leagueGroups[league];
    if (!teams || teams.length === 0) continue;

    teamsHtml += `<div><h4 style="margin-bottom: 12px; color: #f05a28;">${league}</h4>`;
    teamsHtml += '<div style="display: flex; flex-direction: column; gap: 8px;">';

    for (const { squad, fixtures } of teams) {
      const isExcluded = filters.excludeTeams.includes(squad.id);

      let fixturesDisplay = '';
      for (const fixture of fixtures) {
        const opp = fixture.isHome ? fixture.awayId : fixture.homeId;
        const oppSquad = squadsMap[opp];
        const oppName = oppSquad?.shortName || '?';
        const homeAway = fixture.isHome ? 'H' : 'A';
        const difficulty = getFixtureDifficulty(opp);
        const fixtureBadgeClass = `fixture-${difficulty}`;
        fixturesDisplay += `<span class="fixture-badge ${fixtureBadgeClass}" style="margin-right: 4px; display: inline-block;">${oppName}(${homeAway})</span>`;
      }

      const checkboxId = `team-check-${squad.id}`;
      teamsHtml += `
        <label style="display: flex; align-items: flex-start; gap: 10px; padding: 8px; background: ${isExcluded ? '#f0f0f0' : '#fafafa'}; border-radius: 4px; cursor: pointer;">
          <input type="checkbox" id="${checkboxId}" class="team-target-checkbox" value="${squad.id}" ${isExcluded ? '' : 'checked'} style="cursor: pointer; width: 18px; height: 18px; margin-top: 2px; flex-shrink: 0;" />
          <div style="flex: 1; ${isExcluded ? 'opacity: 0.5;' : ''}">
            <div style="font-weight: bold; font-size: 0.95rem;">${squad.shortName || squad.name}</div>
            <div style="font-size: 0.75rem; color: #666; margin-top: 2px;">${fixtures.length} fixture${fixtures.length !== 1 ? 's' : ''} • ${fixturesDisplay}</div>
          </div>
        </label>
      `;
    }

    teamsHtml += '</div></div>';
  }

  teamsHtml += '</div></div>';
  html += teamsHtml;

  // Build save filters modal
  html += `
    <div id="save-filters-modal" class="games-modal" style="display: none;">
      <div class="games-modal-content" style="max-width: 400px;">
        <span class="games-modal-close" style="cursor: pointer;">&times;</span>
        <h3 style="margin-top: 0;">Save Filter Set</h3>
        <input type="text" id="filter-name-input" placeholder="Enter filter name..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 0.9rem; box-sizing: border-box;" />
        <div style="margin-top: 16px; display: flex; gap: 8px;">
          <button id="confirm-save-filters-btn" style="flex: 1; padding: 8px; background: #f05a28; color: white; border: none; border-radius: 4px; cursor: pointer;">Save</button>
          <button id="cancel-save-filters-btn" style="flex: 1; padding: 8px; background: #999; color: white; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
        </div>
      </div>
    </div>
  `;

  // Build load filters modal
  html += `
    <div id="load-filters-modal" class="games-modal" style="display: none;">
      <div class="games-modal-content" style="max-width: 400px;">
        <span class="games-modal-close" style="cursor: pointer;">&times;</span>
        <h3 style="margin-top: 0;">Load Filter Set</h3>
        <div id="filter-list-container" style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 12px;">
          <!-- Filter list populated by JS -->
        </div>
      </div>
    </div>
  `;

  html += `
    <div id="methodology-modal" class="games-modal" style="display: none;">
      <div class="games-modal-content" style="max-width: 500px;">
        <span class="games-modal-close" style="cursor: pointer;">&times;</span>
        <h3 style="margin-top: 0;">Methodology</h3>
        <div style="font-size: 0.9rem; line-height: 1.6;">
          <p><strong>Projection Formula:</strong></p>
          <p>Each player's empirical Home/Away Points Per 90 (calculated from their actual game history)</p>

          <p style="margin-top: 16px;"><strong>How it works:</strong></p>
          <ul style="margin: 8px 0; padding-left: 20px;">
            <li>We analyze each player's actual performance in home vs away games</li>
            <li>We calculate their average points per 90 minutes for each location</li>
            <li>These real-world averages are used to project their points for upcoming fixtures</li>
          </ul>

          <p style="margin-top: 16px; padding: 10px; background: #fff3cd; border-radius: 4px; border-left: 3px solid #f0ad4e;">
            <strong>⚠️ Important:</strong> These projections assume normal playing time. <strong>You must check estimated minutes</strong> for each player in the teams themselves, as absences, injuries, or rotation can significantly impact actual performance.
          </p>
        </div>
      </div>
    </div>
  `;

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

  // Toggle All checkbox
  const toggleAllCheckbox = document.getElementById('toggle-all-teams');
  if (toggleAllCheckbox) {
    toggleAllCheckbox.addEventListener('change', (e) => {
      const allTeamIds = Array.from(document.querySelectorAll('.team-target-checkbox')).map(cb => parseInt(cb.value, 10));

      if (e.target.checked) {
        // Include all
        filters.excludeTeams = [];
      } else {
        // Exclude all
        filters.excludeTeams = allTeamIds;
      }
      renderWithFilters();
    });
  }

  // Update toggle all checkbox state after rendering
  function updateToggleAllState() {
    const toggleAllCheckbox = document.getElementById('toggle-all-teams');
    if (toggleAllCheckbox) {
      const allTeamIds = Array.from(document.querySelectorAll('.team-target-checkbox')).map(cb => parseInt(cb.value, 10));
      const allChecked = allTeamIds.length > 0 && allTeamIds.every(id => !filters.excludeTeams.includes(id));
      toggleAllCheckbox.checked = allChecked;
    }
  }

  // Team target checkboxes
  document.querySelectorAll('.team-target-checkbox').forEach(checkbox => {
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

  // Save/Load filters functionality
  const saveFiltersBtn = document.getElementById('save-filters-btn');
  const loadFiltersBtn = document.getElementById('load-filters-btn');
  const saveFiltersModal = document.getElementById('save-filters-modal');
  const loadFiltersModal = document.getElementById('load-filters-modal');
  const filterNameInput = document.getElementById('filter-name-input');
  const confirmSaveBtn = document.getElementById('confirm-save-filters-btn');
  const cancelSaveBtn = document.getElementById('cancel-save-filters-btn');

  function getSavedFilterSets() {
    try {
      const saved = localStorage.getItem('efl_filter_sets');
      return saved ? JSON.parse(saved) : {};
    } catch (err) {
      console.warn('Error loading filter sets:', err);
      return {};
    }
  }

  function saveSavedFilterSets(filterSets) {
    try {
      localStorage.setItem('efl_filter_sets', JSON.stringify(filterSets));
    } catch (err) {
      console.warn('Error saving filter sets:', err);
    }
  }

  saveFiltersBtn.addEventListener('click', () => {
    filterNameInput.value = '';
    saveFiltersModal.style.display = 'block';
    filterNameInput.focus();
  });

  confirmSaveBtn.addEventListener('click', () => {
    const name = filterNameInput.value.trim();
    if (!name) {
      alert('Please enter a filter set name');
      return;
    }
    const filterSets = getSavedFilterSets();
    filterSets[name] = JSON.parse(JSON.stringify(filters));
    saveSavedFilterSets(filterSets);
    alert(`Filter set "${name}" saved!`);
    saveFiltersModal.style.display = 'none';
  });

  cancelSaveBtn.addEventListener('click', () => {
    saveFiltersModal.style.display = 'none';
  });

  loadFiltersBtn.addEventListener('click', () => {
    const filterSets = getSavedFilterSets();
    const listContainer = document.getElementById('filter-list-container');

    if (Object.keys(filterSets).length === 0) {
      listContainer.innerHTML = '<div style="padding: 12px; color: #999; text-align: center;">No saved filter sets</div>';
    } else {
      listContainer.innerHTML = Object.keys(filterSets).map(name => `
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; border-bottom: 1px solid #eee;">
          <button id="load-${name}" style="flex: 1; text-align: left; background: none; border: none; cursor: pointer; padding: 0; color: #0066cc; text-decoration: underline;">${name}</button>
          <button id="delete-${name}" style="background: #ddd; border: none; cursor: pointer; padding: 4px 8px; border-radius: 3px; color: #d32f2f; font-size: 0.8rem;">Delete</button>
        </div>
      `).join('');

      // Add load event listeners
      Object.keys(filterSets).forEach(name => {
        document.getElementById(`load-${name}`).addEventListener('click', () => {
          Object.assign(filters, JSON.parse(JSON.stringify(filterSets[name])));
          renderWithFilters();
          loadFiltersModal.style.display = 'none';
        });

        // Add delete event listeners
        document.getElementById(`delete-${name}`).addEventListener('click', () => {
          if (confirm(`Delete "${name}"?`)) {
            delete filterSets[name];
            saveSavedFilterSets(filterSets);
            loadFiltersBtn.click(); // Refresh the list
          }
        });
      });
    }

    loadFiltersModal.style.display = 'block';
  });

  // Close modals
  document.querySelectorAll('#save-filters-modal .games-modal-close, #load-filters-modal .games-modal-close').forEach(btn => {
    btn.addEventListener('click', () => {
      saveFiltersModal.style.display = 'none';
      loadFiltersModal.style.display = 'none';
    });
  });

  saveFiltersModal.addEventListener('click', (e) => {
    if (e.target === saveFiltersModal) saveFiltersModal.style.display = 'none';
  });

  loadFiltersModal.addEventListener('click', (e) => {
    if (e.target === loadFiltersModal) loadFiltersModal.style.display = 'none';
  });

  // Methodology modal functionality
  const methodologyBtn = document.getElementById('methodology-btn');
  const methodologyModal = document.getElementById('methodology-modal');

  if (methodologyBtn) {
    methodologyBtn.addEventListener('click', () => {
      methodologyModal.style.display = 'block';
    });
  }

  const methodologyCloseBtns = methodologyModal.querySelectorAll('.games-modal-close');
  methodologyCloseBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      methodologyModal.style.display = 'none';
    });
  });

  methodologyModal.addEventListener('click', (e) => {
    if (e.target === methodologyModal) {
      methodologyModal.style.display = 'none';
    }
  });

  // Games modal functionality
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

  // Exclude player buttons
  document.querySelectorAll('.pick-exclude-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const playerId = parseInt(btn.getAttribute('data-player-id'), 10);
      if (!filters.excludePlayers.includes(playerId)) {
        filters.excludePlayers.push(playerId);
      }
      renderWithFilters();
    });
  });

  // Exclude player checkboxes (to re-include)
  document.querySelectorAll('.exclude-player-checkbox').forEach(checkbox => {
    checkbox.addEventListener('change', (e) => {
      const playerId = parseInt(e.target.value, 10);
      if (e.target.checked) {
        filters.excludePlayers = filters.excludePlayers.filter(id => id !== playerId);
      }
      renderWithFilters();
    });
  });

  // Update toggle all state after rendering
  updateToggleAllState();
}

loadPicks();
