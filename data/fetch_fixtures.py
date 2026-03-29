"""
Fetch all remaining EFL Fantasy fixtures and save to fixtures_remaining.json.

Run this script whenever you need to update fixture data:
    python data/fetch_fixtures.py

Output: data/fixtures_remaining.json
"""
import requests
import json
from pathlib import Path

data_dir = Path(__file__).parent

# Fetch rounds and squads from EFL API
headers = {'Referer': 'https://fantasy.efl.com/', 'User-Agent': 'Mozilla/5.0'}
rounds_data = requests.get('https://fantasy.efl.com/json/fantasy/rounds.json', headers=headers).json()
squads_data = requests.get('https://fantasy.efl.com/json/fantasy/squads.json', headers=headers).json()

# Build squad lookup
squads_by_id = {s['id']: s for s in squads_data}

def get_fixture_difficulty(opponent_id):
    squad = squads_by_id.get(opponent_id)
    if not squad:
        return 'medium'
    pos = squad.get('leaguePosition', 999)
    if pos <= 8: return 'hard'
    if pos <= 16: return 'medium'
    return 'easy'

# Find completed gameweeks from existing CSV files
csv_files = sorted(data_dir.glob('player_stats_gw*.csv'))
if csv_files:
    import pandas as pd
    # Extract GW numbers from filenames
    completed_gws = set()
    for f in csv_files:
        gw_str = f.stem.replace('player_stats_gw', '')
        try:
            completed_gws.add(int(gw_str))
        except ValueError:
            pass
    max_completed = max(completed_gws) if completed_gws else 0
else:
    max_completed = 0

print(f'Max completed gameweek: {max_completed}')

# Build fixtures for remaining rounds
remaining_fixtures = {}
for rnd in rounds_data:
    rnd_num = rnd.get('roundNumber', rnd.get('id'))
    if rnd_num <= max_completed:
        continue

    fixtures = []
    for game in rnd.get('games', []):
        home_id = game['homeId']
        away_id = game['awayId']
        home_squad = squads_by_id.get(home_id, {})
        away_squad = squads_by_id.get(away_id, {})

        fixtures.append({
            'game_id': game.get('id'),
            'home_id': home_id,
            'home_name': home_squad.get('shortName', home_squad.get('name', '')),
            'away_id': away_id,
            'away_name': away_squad.get('shortName', away_squad.get('name', '')),
            'home_difficulty': get_fixture_difficulty(away_id),
            'away_difficulty': get_fixture_difficulty(home_id),
        })

    if fixtures:
        remaining_fixtures[str(rnd_num)] = fixtures

# Also save squads lookup for notebooks to use
squads_path = data_dir / 'squads.json'
with open(squads_path, 'w') as f:
    json.dump(squads_data, f, indent=2)

fixtures_path = data_dir / 'fixtures_remaining.json'
with open(fixtures_path, 'w') as f:
    json.dump(remaining_fixtures, f, indent=2)

print(f'Saved {len(remaining_fixtures)} future gameweeks to {fixtures_path}')
print(f'Saved {len(squads_data)} squads to {squads_path}')
for gw, games in sorted(remaining_fixtures.items(), key=lambda x: int(x[0])):
    print(f'  GW {gw}: {len(games)} games')
