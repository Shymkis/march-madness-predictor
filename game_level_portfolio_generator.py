"""
Portfolio generator using game-level predictions.
Creates 40 diverse brackets for Sweet 16 onwards using game-level model.
"""

import pickle
import pandas as pd
import numpy as np
import random
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load game-level models
print("Loading models...")
with open("game_level_models.pkl", "rb") as f:
    game_level_models = pickle.load(f)

# Load 2026 metrics
df_2026 = pd.read_csv("csv_files/2026.csv")

# Pre-load team metrics
team_metrics_dict = {}
for year in range(2002, 2027):
    try:
        team_metrics_dict[year] = pd.read_csv(f"csv_files/{year}.csv")
    except FileNotFoundError:
        pass

from game_level_integration import get_game_win_probability

# Sweet 16 teams confirmed
sweet16_teams = {
    "East": ["Duke", "St. John's", "Michigan St.", "Connecticut"],
    "South": ["Iowa", "Nebraska", "Illinois", "Houston"],
    "West": ["Arizona", "Arkansas", "Texas", "Purdue"],
    "Midwest": ["Michigan", "Alabama", "Tennessee", "Iowa St."]
}

def generate_simple_bracket(teams_dict, models, metrics, year=2026, deterministic=False):
    """
    Generate a simple playoff style bracket.
    Returns (champion, final_four, elite_eight) for each configuration.
    """
    results = {
        'champion': None,
        'final_four': [],
        'elite_eight': []
    }

    # Regional winners (Sweet 16 -> Elite 8)
    regional_winners = {}
    for region, teams in teams_dict.items():
        # S16: Teams[0,1] vs Teams[2,3]
        t1, t2, t3, t4 = teams[0], teams[1], teams[2], teams[3]

        # Semi-final 1: t1 vs t2
        p_1_wins, p_2_wins = get_game_win_probability(t1, t2, year, 3, models, metrics)
        semi1_winner = t1 if (deterministic and p_1_wins >= 0.5) or (random.random() < p_1_wins) else t2

        # Semi-final 2: t3 vs t4
        p_3_wins, p_4_wins = get_game_win_probability(t3, t4, year, 3, models, metrics)
        semi2_winner = t3 if (deterministic and p_3_wins >= 0.5) or (random.random() < p_3_wins) else t4

        # Regional final: semi1_winner vs semi2_winner
        p_s1_wins, p_s2_wins = get_game_win_probability(semi1_winner, semi2_winner, year, 4, models, metrics)
        regional_winner = semi1_winner if (deterministic and p_s1_wins >= 0.5) or (random.random() < p_s1_wins) else semi2_winner

        regional_winners[region] = regional_winner
        results['elite_eight'].append(regional_winner)

    # Final Four: E vs S, W vs MW
    east_winner = regional_winners["East"]
    south_winner = regional_winners["South"]
    west_winner = regional_winners["West"]
    midwest_winner = regional_winners["Midwest"]

    p_e_wins, p_s_wins = get_game_win_probability(east_winner, south_winner, year, 5, models, metrics)
    f4_winner1 = east_winner if (deterministic and p_e_wins >= 0.5) or (random.random() < p_e_wins) else south_winner

    p_w_wins, p_mw_wins = get_game_win_probability(west_winner, midwest_winner, year, 5, models, metrics)
    f4_winner2 = west_winner if (deterministic and p_w_wins >= 0.5) or (random.random() < p_w_wins) else midwest_winner

    results['final_four'] = [f4_winner1, f4_winner2]

    # Championship
    p_f1_wins, p_f2_wins = get_game_win_probability(f4_winner1, f4_winner2, year, 6, models, metrics)
    champion = f4_winner1 if (deterministic and p_f1_wins >= 0.5) or (random.random() < p_f1_wins) else f4_winner2

    results['champion'] = champion
    return results

# Generate 40 portfolios
print("\nGenerating 40 portfolios using game-level model...")
portfolios = []

for i in range(40):
    # First 10 deterministic, rest stochastic
    deterministic = (i < 10)
    results = generate_simple_bracket(sweet16_teams, game_level_models, team_metrics_dict, year=2026, deterministic=deterministic)

    portfolios.append({
        'bracket_num': i + 1,
        'group': chr(65 + (i // 10)),  # A, B, C, D
        'champion': results['champion'],
        'final_four': ', '.join(results['final_four']),
        'elite_eight': ', '.join(results['elite_eight']),
        'deterministic': deterministic
    })

    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1} brackets")

# Save results
output_df = pd.DataFrame(portfolios)
output_path = Path("results/2026/game_level_portfolios.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(output_path, index=False)

print(f"\nSaved {len(output_df)} portfolios to {output_path}")
print(f"\nChampion distribution:")
print(output_df['champion'].value_counts())
print(f"\nTop finalists:")
all_finalists = output_df['final_four'].str.split(', ', expand=True).stack().value_counts()
print(all_finalists.head(10))
