// Cloud backup for the Club Planner's picks (so they can be loaded from another browser or a private window).
// One record per profile lives in club-picks.json on the `picks-store` branch, a separate branch so that saving
// does not redeploy the site. The first save for a profile sets its passphrase; later saves and loads need it.
// The repo is public: picks and a salted PBKDF2 hash of the passphrase are readable by anyone.
const REPO = 'JackD141/efl-site';
const BRANCH = 'picks-store';
const FILE = 'club-picks.json';
const NAMES = ['Jack', 'John', 'Test']; // 'Test' is only used to check the live route without touching real profiles
const ITERATIONS = 100000;

const { webcrypto } = require('crypto');
const subtle = webcrypto.subtle;
const enc = new TextEncoder();

const toB64 = s => (typeof Buffer !== 'undefined' ? Buffer.from(s, 'utf8').toString('base64') : btoa(unescape(encodeURIComponent(s))));
const fromB64 = b => (typeof Buffer !== 'undefined' ? Buffer.from(b, 'base64').toString('utf8') : decodeURIComponent(escape(atob(b))));
const hex = buf => [...new Uint8Array(buf)].map(x => x.toString(16).padStart(2, '0')).join('');
const unhex = h => new Uint8Array(h.match(/../g).map(x => parseInt(x, 16)));

async function derive(passphrase, salt) {
  const key = await subtle.importKey('raw', enc.encode(passphrase), 'PBKDF2', false, ['deriveBits']);
  return hex(await subtle.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt, iterations: ITERATIONS }, key, 256));
}
function same(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

const API = `https://api.github.com/repos/${REPO}/contents/${FILE}`;
const ghHeaders = token => ({
  'Authorization': `token ${token}`,
  'Accept': 'application/vnd.github.v3+json',
  'User-Agent': 'efl-site-club-picks',
});

async function readStore(token) {
  const r = await fetch(`${API}?ref=${BRANCH}`, { headers: ghHeaders(token) });
  if (r.status === 404) return { store: {}, sha: null }; // file (or branch) not there yet
  if (!r.ok) throw new Error(`GitHub read failed: ${r.status}`);
  const j = await r.json();
  return { store: JSON.parse(fromB64(j.content.replace(/\n/g, ''))), sha: j.sha };
}
function writeStore(token, store, sha, message) {
  const body = { message, content: toB64(JSON.stringify(store, null, 2)), branch: BRANCH, ...(sha && { sha }) };
  return fetch(API, { method: 'PUT', headers: { ...ghHeaders(token), 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

// {gameweek: [clubId|null, clubId|null]} with sane bounds; returns null if the shape is wrong
function cleanPicks(p) {
  if (!p || typeof p !== 'object' || Array.isArray(p)) return null;
  const out = {};
  for (const [gw, ids] of Object.entries(p)) {
    const n = Number(gw);
    if (!Number.isInteger(n) || n < 1 || n > 60) return null;
    if (!Array.isArray(ids) || ids.length !== 2) return null;
    if (!ids.every(x => x === null || (Number.isInteger(x) && x > 0 && x < 1e6))) return null;
    if (ids[0] === null && ids[1] === null) continue;
    out[n] = ids;
  }
  return out;
}

module.exports = async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST.' });

  let body = req.body;
  try { if (typeof body === 'string') body = JSON.parse(body); } catch (e) { return res.status(400).json({ error: 'Bad request body.' }); }
  const { action, name, passphrase, picks } = body || {};
  if (action !== 'save' && action !== 'load') return res.status(400).json({ error: 'Unknown action.' });
  if (!NAMES.includes(name)) return res.status(400).json({ error: 'Unknown profile.' });
  if (typeof passphrase !== 'string' || passphrase.length < 4 || passphrase.length > 100) {
    return res.status(400).json({ error: 'Passphrase must be 4 to 100 characters.' });
  }
  const token = process.env.GITHUB_TOKEN;
  if (!token) return res.status(500).json({ error: 'GITHUB_TOKEN is not set on the server.' });

  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      const { store, sha } = await readStore(token);
      const rec = store[name];

      if (action === 'load') {
        if (!rec) return res.status(404).json({ error: `Nothing saved yet for ${name}.` });
        if (!same(await derive(passphrase, unhex(rec.salt)), rec.hash)) return res.status(401).json({ error: 'Wrong passphrase.' });
        return res.status(200).json({ ok: true, picks: rec.picks, savedAt: rec.savedAt });
      }

      const clean = cleanPicks(picks);
      if (!clean) return res.status(400).json({ error: 'Picks are not in the expected format.' });
      let salt, hash, created = false;
      if (rec) {
        if (!same(await derive(passphrase, unhex(rec.salt)), rec.hash)) return res.status(401).json({ error: 'Wrong passphrase.' });
        ({ salt, hash } = rec);
      } else {
        const s = webcrypto.getRandomValues(new Uint8Array(16));
        salt = hex(s);
        hash = await derive(passphrase, s);
        created = true;
      }
      const savedAt = new Date().toISOString();
      store[name] = { salt, hash, picks: clean, savedAt };
      const w = await writeStore(token, store, sha, `Save ${name}'s club picks`);
      if (w.ok) return res.status(200).json({ ok: true, savedAt, created });
      if (w.status === 404) return res.status(500).json({ error: `The ${BRANCH} branch does not exist on GitHub.` });
      if (w.status !== 409 && w.status !== 422) throw new Error(`GitHub write failed: ${w.status}`);
      // 409/422: the file changed since we read it; read again and retry once
    }
    return res.status(409).json({ error: 'Could not save just now. Try again.' });
  } catch (err) {
    return res.status(500).json({ error: 'Could not reach the picks store.', details: err.message });
  }
};
