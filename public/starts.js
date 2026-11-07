// Start chances -> minutes scenarios. Shared by Player Picks (keepers.js / defenders.js / attackers.js) and the homepage
// (xp-core.js); same maths as ml/team_strengths/check_site_maths.py.
// Each player in a plan has (minutes_model.py, start model):
//   pStart    chance he starts his club's next game
//   pFull     chance he plays 60+ once he starts (his own: a 90-minute regular is not docked for others' substitutions)
//   pSubN     chance he comes off the bench if he does not start
//   pStart2c / pStart2n   a double's second game: chance he starts it if he started / did not start the first
//   mFull     his usual minutes when he plays 60+
// A gameweek game is then: P(full game) = start x pFull (appearance 2, clean sheet counts), P(part game) = start x
// (1 - pFull) + (1 - start) x pSubN (appearance 1). "If he starts" sets the first game's start chance to 100%.
const START_KEYS = { GK: 'efl_keeper_start_v1', DEF: 'efl_defender_start_v1', MID: 'efl_mid_start_v1', FWD: 'efl_fwd_start_v1' };

function readStore(key) {
  try { const v = JSON.parse(localStorage.getItem(key) || '{}'); return v && typeof v === 'object' ? v : {}; } catch (e) { return {}; }
}
function writeStore(key, v) {
  try { localStorage.setItem(key, JSON.stringify(v)); } catch (e) { /* ignore */ }
}
// chance he starts game j (0-based) of a gameweek, given his first-game start chance s1
function startChanceAt(d, j, s1, asStarter) {
  const a = asStarter ? 1 : s1;
  if (j === 0) return a;
  return d.pStart2c == null ? a : a * d.pStart2c + (1 - a) * d.pStart2n;
}
// chance of each minutes scenario in a game he starts with chance ps
function scenarioProbs(plan, d, ps) {
  const q = d.pFull != null ? d.pFull : plan.startModel.q;
  return { pFull: ps * q, pPart: ps * (1 - q) + (1 - ps) * (d.pSubN || 0) };
}
// his expected minutes if he starts
function minsIfStarts(plan, d) {
  const q = d.pFull != null ? d.pFull : plan.startModel.q;
  return q * (d.mFull || plan.startModel.full) + (1 - q) * plan.startModel.part;
}
// the editable start-chance cell (Player Picks): value in %, then minutes if he starts and, for a double, the second game
function startCellHtml(plan, d, s1, isEdited, dataAttr, nGames) {
  const p2 = d.pStart2c != null && nGames > 1 ? startChanceAt(d, 1, s1, false) : null;
  const q = d.pFull != null ? d.pFull : plan.startModel.q;
  return `<input type="number" class="kp-mins ${isEdited ? 'kp-mins-edited' : ''}" ${dataAttr} min="0" max="100" step="5" value="${Math.round(s1 * 100)}"
      title="Chance he starts his club's next game, %. Model: ${Math.round((d.pStart || 0) * 100)}%. Change it after team news (100 = in the line-up, 0 = not); saved in this browser." />
    <div class="cp-muted cp-small" title="Minutes if he starts: his usual full game (${Math.round(d.mFull || plan.startModel.full)}), allowing a ${Math.round((1 - q) * 100)}% chance he comes off before 60">if starts: ${Math.round(minsIfStarts(plan, d))}'</div>${p2 != null
      ? `<div class="cp-muted cp-small" title="Chance he starts the second game of the double (${Math.round(d.pStart2c * 100)}% if he starts the first, ${Math.round(d.pStart2n * 100)}% if not)">2nd game: ${Math.round(p2 * 100)}%</div>` : ''}`;
}
