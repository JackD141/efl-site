const COGNITO_CLIENT_ID = '4q60ebh5rv8aduhnfh383epu8u';
const COGNITO_USER_ID = 'a171833f-3556-4f4b-83ed-6eaa4318d371';

const BASE_HEADERS = {
  'Accept': 'application/json',
  'Referer': 'https://fantasy.efl.com/',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
};

async function getCognitoTokens(email, password) {
  const response = await fetch('https://cognito-idp.eu-west-1.amazonaws.com/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
    },
    body: JSON.stringify({
      AuthFlow: 'USER_PASSWORD_AUTH',
      ClientId: COGNITO_CLIENT_ID,
      AuthParameters: { USERNAME: email, PASSWORD: password },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Cognito auth failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  const { IdToken, AccessToken, RefreshToken } = data.AuthenticationResult;
  return { IdToken, AccessToken, RefreshToken };
}

function buildCognitoCookies(idToken, accessToken, refreshToken) {
  const prefix = `CognitoIdentityServiceProvider.${COGNITO_CLIENT_ID}.${COGNITO_USER_ID}`;
  return [
    `${prefix}.idToken=${idToken}`,
    `${prefix}.accessToken=${accessToken}`,
    `${prefix}.refreshToken=${refreshToken}`,
    `${prefix}.clockDrift=0`,
    `CognitoIdentityServiceProvider.${COGNITO_CLIENT_ID}.LastAuthUser=${COGNITO_USER_ID}`,
  ].join('; ');
}

async function getPlayerProfile(playerId, idToken) {
  const url = `https://fantasy.efl.com/json/fantasy/player_profiles/${playerId}.json`;
  const response = await fetch(url, {
    headers: {
      ...BASE_HEADERS,
      'Authorization': `Bearer ${idToken}`,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch player ${playerId}: ${response.status}`);
  }

  return response.json();
}

async function getAllPlayers() {
  const response = await fetch('https://fantasy.efl.com/json/fantasy/players.json', {
    headers: BASE_HEADERS,
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch players list: ${response.status}`);
  }

  return response.json();
}

function generateCSVContent(gameweek, playerStats) {
  if (playerStats.length === 0) {
    return '';
  }

  const headers = [
    'player_id', 'first_name', 'last_name', 'display_name', 'position', 'squad_id',
    'gameweek', 'minutes_played', 'goals_scored', 'hat_tricks', 'assists', 'penalty_misses',
    'own_goals', 'yellow_cards', 'red_cards', 'saves', 'penalty_saves', 'clean_sheet',
    'goals_conceded', 'clearances', 'blocks', 'tackles', 'interceptions', 'key_passes',
    'shots_on_target', 'points'
  ];

  const rows = playerStats.map(stat => [
    stat.player_id,
    stat.first_name,
    stat.last_name,
    stat.display_name,
    stat.position,
    stat.squad_id,
    stat.gameweek,
    stat.minutes_played || 0,
    stat.goals_scored || 0,
    stat.hat_tricks || 0,
    stat.assists || 0,
    stat.penalty_misses || 0,
    stat.own_goals || 0,
    stat.yellow_cards || 0,
    stat.red_cards || 0,
    stat.saves || 0,
    stat.penalty_saves || 0,
    stat.clean_sheet || 0,
    stat.goals_conceded || 0,
    stat.clearances || 0,
    stat.blocks || 0,
    stat.tackles || 0,
    stat.interceptions || 0,
    stat.key_passes || 0,
    stat.shots_on_target || 0,
    stat.points || 0,
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.map(cell => {
      // Escape quotes and wrap in quotes if contains comma
      const str = String(cell);
      if (str.includes(',') || str.includes('"') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    }).join(','))
  ].join('\n');

  return csvContent;
}

async function commitToGitHub(csvsByGameweek, githubToken) {
  const repo = 'JackD141/efl-site';
  const owner = 'JackD141';
  const branch = 'main';

  for (const [gameweek, csvContent] of Object.entries(csvsByGameweek)) {
    const filename = `data/player_stats_gw${gameweek}.csv`;

    // Get file SHA if it exists (for updating)
    let fileSha = null;
    try {
      const getResponse = await fetch(
        `https://api.github.com/repos/${repo}/contents/${filename}`,
        {
          headers: {
            'Authorization': `token ${githubToken}`,
            'Accept': 'application/vnd.github.v3+json',
          },
        }
      );

      if (getResponse.ok) {
        const fileData = await getResponse.json();
        fileSha = fileData.sha;
      }
    } catch (e) {
      // File doesn't exist yet, that's fine
    }

    // Commit the file
    const content = Buffer.from(csvContent).toString('base64');
    const commitResponse = await fetch(
      `https://api.github.com/repos/${repo}/contents/${filename}`,
      {
        method: 'PUT',
        headers: {
          'Authorization': `token ${githubToken}`,
          'Accept': 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: `Save player stats for gameweek ${gameweek}`,
          content,
          branch,
          ...(fileSha && { sha: fileSha }),
        }),
      }
    );

    if (!commitResponse.ok) {
      const error = await commitResponse.json();
      throw new Error(`Failed to commit GW${gameweek}: ${error.message}`);
    }
  }

  return Object.keys(csvsByGameweek);
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Content-Type', 'application/json');

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const email = process.env.EFL_EMAIL;
  const password = process.env.EFL_PASSWORD;
  const githubToken = process.env.GITHUB_TOKEN;

  if (!email || !password) {
    return res.status(500).json({ error: 'EFL_EMAIL or EFL_PASSWORD not set' });
  }

  if (!githubToken) {
    return res.status(500).json({ error: 'GITHUB_TOKEN not set' });
  }

  try {
    // Authenticate with EFL
    const { IdToken: idToken } = await getCognitoTokens(email, password);

    // Fetch all players
    const players = await getAllPlayers();
    console.log(`Fetched ${players.length} players`);

    // Organize stats by gameweek
    const statsByGameweek = {};

    // Fetch player profiles in parallel batches of 10 to avoid timeout
    const batchSize = 10;
    for (let i = 0; i < players.length; i += batchSize) {
      const batch = players.slice(i, i + batchSize);
      const profiles = await Promise.all(
        batch.map(p => getPlayerProfile(p.id, idToken))
      );

      batch.forEach((player, idx) => {
        const profile = profiles[idx];
        const results = profile.results || [];

        // Add each result to the corresponding gameweek
        for (const result of results) {
          const gameweek = result.roundId;
          if (!statsByGameweek[gameweek]) {
            statsByGameweek[gameweek] = [];
          }

          statsByGameweek[gameweek].push({
            player_id: player.id,
            first_name: player.firstName,
            last_name: player.lastName,
            display_name: player.displayName,
            position: player.position,
            squad_id: player.squadId,
            gameweek,
            minutes_played: result.minutesPlayed,
            goals_scored: result.goalsScored,
            hat_tricks: result.hatTricks,
            assists: result.assists,
            penalty_misses: result.penaltyMisses,
            own_goals: result.ownGoals,
            yellow_cards: result.yellowCards,
            red_cards: result.redCards,
            saves: result.saves,
            penalty_saves: result.penaltySaves,
            clean_sheet: result.cleanSheet,
            goals_conceded: result.goalsConceded,
            clearances: result.clearances,
            blocks: result.blocks,
            tackles: result.tackles,
            interceptions: result.interceptions,
            key_passes: result.keyPasses,
            shots_on_target: result.shotsOnTarget,
            points: result.points,
          });
        }
      });

      // Log progress
      console.log(`Processed ${Math.min(i + batchSize, players.length)}/${players.length} players`);
    }

    console.log(`Organized stats for ${Object.keys(statsByGameweek).length} gameweeks`);

    // Generate CSV content for each gameweek
    const csvsByGameweek = {};
    for (const [gameweek, stats] of Object.entries(statsByGameweek)) {
      csvsByGameweek[gameweek] = generateCSVContent(gameweek, stats);
    }

    // Commit to GitHub
    const committedGameweeks = await commitToGitHub(csvsByGameweek, githubToken);

    return res.status(200).json({
      success: true,
      message: `Successfully saved ${committedGameweeks.length} gameweeks`,
      gameweeks: committedGameweeks.sort((a, b) => Number(a) - Number(b)),
    });
  } catch (error) {
    console.error('Export error:', error);
    return res.status(500).json({
      error: 'Failed to export player stats',
      details: error.message,
    });
  }
};
