const COGNITO_CLIENT_ID = '4q60ebh5rv8aduhnfh383epu8u';
const COGNITO_USER_ID = 'a171833f-3556-4f4b-83ed-6eaa4318d371';
const EFL_USER_URL = 'https://fantasy.efl.com/api/en/user';
const EFL_API_URL = 'https://fantasy.efl.com/api/en/season/ranking/overall_ladder?leagueId=38';

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
  const { IdToken, AccessToken } = data.AuthenticationResult;
  return { IdToken, AccessToken };
}

function buildCognitoCookies(idToken, accessToken) {
  const prefix = `CognitoIdentityServiceProvider.${COGNITO_CLIENT_ID}.${COGNITO_USER_ID}`;
  return [
    `${prefix}.idToken=${idToken}`,
    `${prefix}.accessToken=${accessToken}`,
    `${prefix}.clockDrift=0`,
    `CognitoIdentityServiceProvider.${COGNITO_CLIENT_ID}.LastAuthUser=${COGNITO_USER_ID}`,
  ].join('; ');
}

function extractSetCookies(response) {
  // getSetCookie() is available in Node 18.14+ / undici
  if (typeof response.headers.getSetCookie === 'function') {
    return response.headers.getSetCookie().map(c => c.split(';')[0]).join('; ');
  }
  // fallback: single set-cookie header
  const sc = response.headers.get('set-cookie');
  return sc ? sc.split(';')[0] : '';
}

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const email = process.env.EFL_EMAIL;
  const password = process.env.EFL_PASSWORD;
  if (!email || !password) {
    return res.status(500).json({ error: 'EFL_EMAIL or EFL_PASSWORD environment variable is not set.' });
  }

  // Step 1: Authenticate with Cognito
  let idToken, accessToken;
  try {
    ({ IdToken: idToken, AccessToken: accessToken } = await getCognitoTokens(email, password));
  } catch (err) {
    return res.status(500).json({ error: 'Failed to authenticate with EFL.', details: err.message });
  }

  const cognitoCookies = buildCognitoCookies(idToken, accessToken);

  // Step 2: Get an anonymous PHPSESSID from the main site
  let anonSessionCookies = '';
  try {
    const anonRes = await fetch('https://fantasy.efl.com/', { headers: BASE_HEADERS });
    anonSessionCookies = extractSetCookies(anonRes);
  } catch (err) { /* non-fatal */ }

  // Step 3: Hit the user endpoint with Cognito + anonymous session to activate auth session
  let authSessionCookies = '';
  let userStatus = 0;
  try {
    const initCookies = anonSessionCookies ? `${cognitoCookies}; ${anonSessionCookies}` : cognitoCookies;
    const userRes = await fetch(EFL_USER_URL, {
      headers: { ...BASE_HEADERS, 'Cookie': initCookies },
    });
    userStatus = userRes.status;
    const newCookies = extractSetCookies(userRes);
    authSessionCookies = newCookies || anonSessionCookies;
  } catch (err) { /* non-fatal */ }

  const allCookies = authSessionCookies
    ? `${cognitoCookies}; ${authSessionCookies}`
    : cognitoCookies;

  // Step 4: Fetch the league table
  let eflResponse;
  try {
    eflResponse = await fetch(EFL_API_URL, {
      headers: { ...BASE_HEADERS, 'Cookie': allCookies },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to reach EFL Fantasy API.', details: err.message });
  }

  if (!eflResponse.ok) {
    // Return debug info so we can diagnose
    return res.status(eflResponse.status).json({
      error: `EFL API returned ${eflResponse.status}`,
      debug: { userStatus, hadAnonSession: !!anonSessionCookies, hadAuthSession: !!authSessionCookies },
    });
  }

  const data = await eflResponse.json();
  return res.status(200).json(data);
}
