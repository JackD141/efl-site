const COGNITO_CLIENT_ID = '4q60ebh5rv8aduhnfh383epu8u';
const COGNITO_USER_ID = process.env.COGNITO_USER_ID || 'a171833f-3556-4f4b-83ed-6eaa4318d371';
const EFL_API_URL = 'https://fantasy.efl.com/api/en/season/ranking/overall_ladder?leagueId=38';

async function getIdToken(refreshToken) {
  const response = await fetch('https://cognito-idp.eu-west-1.amazonaws.com/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
    },
    body: JSON.stringify({
      AuthFlow: 'REFRESH_TOKEN_AUTH',
      ClientId: COGNITO_CLIENT_ID,
      AuthParameters: {
        REFRESH_TOKEN: refreshToken,
      },
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Cognito auth failed: ${response.status} ${text}`);
  }

  const data = await response.json();
  return data.AuthenticationResult.IdToken;
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const refreshToken = process.env.COGNITO_REFRESH_TOKEN;
  if (!refreshToken) {
    return res.status(500).json({ error: 'COGNITO_REFRESH_TOKEN environment variable is not set.' });
  }

  let idToken;
  try {
    idToken = await getIdToken(refreshToken);
  } catch (err) {
    return res.status(500).json({ error: 'Failed to authenticate with EFL.', details: err.message });
  }

  const cookieName = `CognitoIdentityServiceProvider.${COGNITO_CLIENT_ID}.${COGNITO_USER_ID}.idToken`;

  let eflResponse;
  try {
    eflResponse = await fetch(EFL_API_URL, {
      headers: {
        'Accept': 'application/json',
        'Cookie': `${cookieName}=${idToken}`,
        'Referer': 'https://fantasy.efl.com/',
        'User-Agent': 'Mozilla/5.0',
      },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to reach EFL Fantasy API.', details: err.message });
  }

  if (!eflResponse.ok) {
    return res.status(eflResponse.status).json({ error: `EFL API returned ${eflResponse.status}` });
  }

  const data = await eflResponse.json();
  return res.status(200).json(data);
}
