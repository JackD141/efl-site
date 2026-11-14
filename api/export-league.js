// Saves the top of a Fantasy EFL league table and every listed manager's team for each gameweek, so the League Rivals
// page can score their picks with our model (ml/team_strengths/league_rivals.py).
// One file per league: data/<season>/leagues/league_<id>.json, committed to GitHub like the player stats export.
// Teams of finished gameweeks never change, so they are fetched once and kept; only new ones are fetched each run.
const COGNITO_CLIENT_ID = '4q60ebh5rv8aduhnfh383epu8u';
const API = 'https://fantasy.efl.com/api/en';
const MODE = 'season';
const REPO = 'JackD141/efl-site';
const TOP_N = 20;

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
  if (!response.ok) throw new Error(`Cognito auth failed: ${response.status}`);
  return (await response.json()).AuthenticationResult;
}

// "name=value" pairs from a response's Set-Cookie headers
function cookiesOf(response) {
  let list = [];
  if (typeof response.headers.getSetCookie === 'function') list = response.headers.getSetCookie();
  else if (response.headers.get('set-cookie')) list = response.headers.get('set-cookie').split(/,(?=\s*[^;,=\s]+=)/);
  return list.map(c => c.split(';')[0].trim()).filter(Boolean);
}

// the fantasy site's own login: it exchanges the EFL account token for a session cookie
async function login(token) {
  const response = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { ...BASE_HEADERS, 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  return { status: response.status, cookies: cookiesOf(response), text: response.ok ? '' : (await response.text()).slice(0, 300) };
}

async function api(path, session) {
  const response = await fetch(`${API}/${path}`, { headers: { ...BASE_HEADERS, ...session } });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  const body = await response.json();
  return body.success;
}

function seasonLabel(date = new Date()) {
  const year = date.getUTCFullYear();
  const start = date.getUTCMonth() >= 6 ? year : year - 1;
  return `${start}_${String((start + 1) % 100).padStart(2, '0')}`;
}

const gh = token => ({ 'Authorization': `token ${token}`, 'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'efl-site-export-league' });

async function readFile(path, token) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/contents/${path}?ref=main`, { headers: { ...gh(token), 'Accept': 'application/vnd.github.raw+json' } });
  if (r.status === 404) return { data: null, sha: null };
  if (!r.ok) throw new Error(`GitHub read failed: ${r.status}`);
  const text = await r.text();
  const meta = await fetch(`https://api.github.com/repos/${REPO}/contents/${path}?ref=main`, { headers: gh(token) }).then(x => x.json());
  return { data: JSON.parse(text), sha: meta.sha };
}

async function writeFile(path, data, sha, message, token) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/contents/${path}`, {
    method: 'PUT',
    headers: { ...gh(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, content: Buffer.from(JSON.stringify(data)).toString('base64'), branch: 'main', ...(sha && { sha }) }),
  });
  if (!r.ok) throw new Error(`GitHub write failed: ${r.status}`);
}

async function inBatches(items, size, fn) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(...await Promise.all(items.slice(i, i + size).map(fn)));
  return out;
}

async function openSession(email, password) {
  const tokens = await getCognitoTokens(email, password);
  const tried = [];
  for (const [name, token] of [['idToken', tokens.IdToken], ['accessToken', tokens.AccessToken]]) {
    const r = await login(token);
    tried.push({ token: name, status: r.status, cookieNames: r.cookies.map(c => c.split('=')[0]), text: r.text });
    if (r.status < 300 && r.cookies.length) return { session: { Cookie: r.cookies.join('; ') }, tried };
  }
  return { session: null, tried };
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const email = process.env.EFL_EMAIL, password = process.env.EFL_PASSWORD, githubToken = process.env.GITHUB_TOKEN;
  if (!email || !password || !githubToken) return res.status(500).json({ error: 'Server credentials are not set.' });
  const q = req.query || {};

  try {
    const { session, tried } = await openSession(email, password);
    if (!session) return res.status(502).json({ error: 'Could not log in to Fantasy EFL.', tried });

    // one-off look at what the EFL returns (shapes only matter; used while building the page)
    if (q.probe) {
      const out = { tried };
      const attempt = async (key, path) => { try { out[key] = await api(path, session); } catch (e) { out[key] = { error: e.message }; } };
      await attempt('leagues', `${MODE}/leagues`);
      const ids = String(q.probe).split(',').map(Number).filter(Number.isInteger).slice(0, 12);
      for (const id of ids) {
        await attempt(`league_${id}`, `${MODE}/league/${id}`);
        try {
          const lad = (await api(`${MODE}/ranking/overall_ladder?leagueId=${id}`, session)).ladder;
          out[`ladder_${id}`] = { ...lad, rankings: (lad.rankings || []).slice(0, 2), count: (lad.rankings || []).length };
          const top = (lad.rankings || [])[0];
          if (top && q.team) out[`team_${id}`] = await api(`${MODE}/other-user-team/${top.userId}/${Number(q.team)}`, session).catch(e => ({ error: e.message }));
        } catch (e) { out[`ladder_${id}`] = { error: e.message }; }
      }
      return res.status(200).json(out);
    }

    if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST.' });
    const leagueId = Number(q.league);
    if (!Number.isInteger(leagueId) || leagueId < 1) return res.status(400).json({ error: 'league is required.' });

    const rounds = await fetch('https://fantasy.efl.com/json/fantasy/rounds.json', { headers: BASE_HEADERS }).then(r => r.json());
    const done = rounds.filter(r => r.status === 'completed').map(r => r.id);
    const live = rounds.filter(r => r.status === 'playing').map(r => r.id);

    const league = await api(`${MODE}/league/${leagueId}`, session).catch(() => null);
    const rankings = [];
    let extra = {};
    for (let page = 1; page <= 5 && rankings.length < TOP_N; page++) {
      const lad = (await api(`${MODE}/ranking/overall_ladder?leagueId=${leagueId}&page=${page}`, session)).ladder;
      const { rankings: rows = [], ...rest } = lad;
      extra = rest;
      const seen = new Set(rankings.map(r => r.userId));
      const fresh = rows.filter(r => !seen.has(r.userId));
      rankings.push(...fresh);
      if (!fresh.length || !rest.nextPage) break;
    }
    const top = rankings.slice(0, TOP_N);

    const path = `data/${seasonLabel()}/leagues/league_${leagueId}.json`;
    const { data: old, sha } = await readFile(path, githubToken);
    const teams = (old && old.teams) || {};
    const jobs = [];
    for (const r of top) {
      const mine = teams[r.userId] || (teams[r.userId] = {});
      for (const id of done) if (!mine[id] || mine[id].live) jobs.push([r.userId, id, false]);
      for (const id of live) jobs.push([r.userId, id, true]);
    }
    let failed = 0;
    await inBatches(jobs, 15, async ([userId, roundId, isLive]) => {
      try {
        const s = await api(`${MODE}/other-user-team/${userId}/${roundId}`, session);
        teams[userId][roundId] = { team: s.team || null, ...(isLive && { live: true }) };
      } catch (e) { failed++; }
    });

    const data = { leagueId, league, fetchedAt: new Date().toISOString(), completedRounds: done, liveRounds: live, ladder: extra, rankings: top, teams };
    await writeFile(path, data, sha, `Save league ${leagueId} table and teams`, githubToken);
    return res.status(200).json({ ok: true, path, managers: top.length, fetched: jobs.length - failed, failed });
  } catch (err) {
    return res.status(500).json({ error: 'League export failed.', details: err.message });
  }
};
