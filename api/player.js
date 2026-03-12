module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');

  const { id } = req.query;
  if (!id || !/^\d+$/.test(id)) {
    return res.status(400).json({ error: 'Missing or invalid player id' });
  }

  const url = `https://fantasy.efl.com/json/fantasy/player_profiles/${id}.json`;

  let upstream;
  try {
    upstream = await fetch(url, {
      headers: {
        'Accept': 'application/json',
        'Referer': 'https://fantasy.efl.com/',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
    });
  } catch (err) {
    return res.status(500).json({ error: 'Failed to reach EFL player profile.', details: err.message });
  }

  if (!upstream.ok) {
    return res.status(upstream.status).json({ error: `EFL returned ${upstream.status}` });
  }

  const data = await upstream.json();
  res.setHeader('Cache-Control', 'public, max-age=300');
  return res.status(200).json(data);
};
