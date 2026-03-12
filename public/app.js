const statusEl = document.getElementById('status');
const tableContainer = document.getElementById('table-container');
const refreshBtn = document.getElementById('refresh-btn');

async function loadTable() {
  refreshBtn.disabled = true;
  statusEl.className = '';
  statusEl.textContent = 'Loading...';
  tableContainer.innerHTML = '';

  try {
    const response = await fetch('/api/league');

    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || `Server error ${response.status}`);
    }

    const data = await response.json();

    // The EFL API returns an object — try to find the array of entries.
    // Log raw data to console so we can inspect the structure if needed.
    console.log('Raw EFL API response:', data);

    const entries = extractEntries(data);

    if (!entries || entries.length === 0) {
      throw new Error('No league data found. The league may be empty or the data format has changed.');
    }

    renderTable(entries);
    const now = new Date().toLocaleTimeString('en-GB');
    statusEl.textContent = `Last updated: ${now}`;
  } catch (err) {
    statusEl.className = 'error';
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  } finally {
    refreshBtn.disabled = false;
  }
}

// Try common field name patterns used by EFL Fantasy API
function extractEntries(data) {
  // The response might be an array directly, or nested under a key
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.ladder)) return data.ladder;
  if (Array.isArray(data.entries)) return data.entries;
  if (Array.isArray(data.standings)) return data.standings;
  if (Array.isArray(data.results)) return data.results;
  if (Array.isArray(data.data)) return data.data;

  // If none match, return null — the error message will prompt us to check console
  return null;
}

function getField(entry, ...candidates) {
  for (const key of candidates) {
    if (entry[key] !== undefined && entry[key] !== null) return entry[key];
  }
  return '—';
}

function renderTable(entries) {
  // Sort by rank/position/points descending if not already sorted
  // (assume API returns them in order already)

  let html = `
    <table>
      <thead>
        <tr>
          <th class="pos">#</th>
          <th>Manager</th>
          <th>Team</th>
          <th>Points</th>
          <th>Advantage</th>
        </tr>
      </thead>
      <tbody>
  `;

  entries.forEach((entry, index) => {
    const pos = getField(entry, 'rank', 'position', 'pos') !== '—'
      ? getField(entry, 'rank', 'position', 'pos')
      : index + 1;

    const manager = getField(entry, 'player_name', 'manager_name', 'manager', 'user_name', 'username');
    const team = getField(entry, 'entry_name', 'team_name', 'name', 'squad_name');
    const points = getField(entry, 'total', 'points', 'total_points', 'score');

    const nextEntry = entries[index + 1];
    let advantage = '—';
    let advantageClass = 'dash';

    if (nextEntry !== undefined) {
      const nextPoints = getField(nextEntry, 'total', 'points', 'total_points', 'score');
      const gap = Number(points) - Number(nextPoints);
      advantage = `+${gap}`;
      advantageClass = '';
    }

    html += `
      <tr>
        <td class="pos">${pos}</td>
        <td>${manager}</td>
        <td>${team}</td>
        <td>${points}</td>
        <td class="advantage ${advantageClass}">${advantage}</td>
      </tr>
    `;
  });

  html += `</tbody></table>`;
  tableContainer.innerHTML = html;
}

// Load automatically when the page opens
loadTable();
