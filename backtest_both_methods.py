"""
Backtest both prediction methods on historical tournament data.
ESPN Scoring: R1=10, R2=20, R3=40, R4=80, R5=160, R6=320
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from game_level_integration import get_game_win_probability

ESPN_SCORES = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# Load game-level models
print("Loading game-level models...")
with open("game_level_models.pkl", "rb") as f:
    game_level_models = pickle.load(f)

# Load KenPom data for all years
print("Loading KenPom metrics...")
team_metrics_dict = {}
for year in range(2002, 2027):
    try:
        team_metrics_dict[year] = pd.read_csv(f"csv_files/{year}.csv")
    except FileNotFoundError:
        pass

# Load all bracket data
print("Loading bracket data...")
all_brackets = {}
for year in range(2002, 2026):
    try:
        all_brackets[year] = pd.read_csv(f"csv_files/{year}bracket.csv")
    except FileNotFoundError:
        pass

def extract_games_from_bracket(bracket_df):
    """Extract individual games from bracket CSV."""
    games = []
    for _, row in bracket_df.iterrows():
        games.append({
            'team_a': row['Team1'],
            'team_b': row['Team2'],
            'round': row['Round'],
            'actual_winner': row['Winner'],
            'score_a': row['Team1 Score'],
            'score_b': row['Team2 Score']
        })
    return games

def build_tournament_winner_ensemble(year):
    """Build ensemble for tournament-winner predictions using 10-fold CV."""
    DROP_COLS_TRAIN = ['Winner', 'Unnamed: 0', 'Seed2', 'ORtg.Rk', 'DRtg.Rk',
                       'AdjT.Rk', 'Luck.Rk', 'SOS.NetRtg.Rk', 'SOS.ORtg.Rk',
                       'SOS.DRtg.Rk', 'NCSOS.Rk', 'W', 'L', 'WinPct', 'Team', 'Conf']

    all_data = []
    for y in range(2002, year):
        if y not in all_brackets or y not in team_metrics_dict:
            continue
        bracket_df = all_brackets[y]
        kenpon_df = team_metrics_dict[y]

        for _, game in bracket_df.iterrows():
            row_a = kenpon_df[kenpon_df['Team'] == game['Team1']]
            row_b = kenpon_df[kenpon_df['Team'] == game['Team2']]
            if row_a.empty or row_b.empty:
                continue

            data = pd.concat([row_a, row_b], ignore_index=True)
            data['Winner'] = 1 if game['Winner'] == game['Team1'] else 0
            all_data.append(data)

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    X = combined_df.drop(columns=[col for col in DROP_COLS_TRAIN if col in combined_df.columns])
    X = X.select_dtypes(include=[np.number])
    y = combined_df["Winner"]

    # Train soft-voting ensemble with 10-fold CV
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = {'lr': [], 'rf': [], 'svm': [], 'xgb': []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        models['lr'].append(LogisticRegression(max_iter=5000, random_state=42).fit(X_train, y_train))
        models['rf'].append(RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train))
        models['svm'].append(SVC(probability=True, random_state=42).fit(X_train, y_train))
        models['xgb'].append(XGBClassifier(random_state=42, verbosity=0).fit(X_train, y_train))

    return {'models': models, 'X_columns': X.columns, 'feature_names': X.columns}

def predict_tournament_winner(ensemble, year, kenpon_df):
    """Predict winner probabilities using TW ensemble."""
    if ensemble is None:
        return {}

    win_probs = {}
    for team in kenpon_df['Team']:
        team_data = kenpon_df[kenpon_df['Team'] == team]
        if team_data.empty:
            continue

        X_team = team_data[ensemble['X_columns']].select_dtypes(include=[np.number])
        probs = []
        for model_type in ['lr', 'rf', 'svm', 'xgb']:
            model_list = ensemble['models'][model_type]
            # Average across CV folds
            fold_probs = [m.predict_proba(X_team)[0, 1] for m in model_list]
            probs.append(np.mean(fold_probs))

        win_probs[team] = np.mean(probs)  # Simple average of 4 models

    return win_probs

def normalize_probs(team_a, team_b, p_a, p_b):
    """Normalize two probabilities to sum to 1."""
    total = p_a + p_b
    if total == 0:
        return 0.5, 0.5
    return p_a / total, p_b / total

def simulate_game_tw(team_a, team_b, win_probs):
    """Simulate game using tournament-winner probabilities."""
    p_a = win_probs.get(team_a, 0.5)
    p_b = win_probs.get(team_b, 0.5)
    p_a, p_b = normalize_probs(team_a, team_b, p_a, p_b)
    return team_a if np.random.random() < p_a else team_b

def simulate_game_gl(team_a, team_b, year, round_num, models, metrics_dict):
    """Simulate game using game-level probabilities."""
    p_a, p_b = get_game_win_probability(team_a, team_b, year, round_num, models, metrics_dict)
    return team_a if np.random.random() < p_a else team_b

def score_bracket(predicted_games, actual_games):
    """Score bracket using ESPN scoring."""
    score = 0
    correct = 0
    for pred in predicted_games:
        actual = next((g for g in actual_games if
                      {g['team_a'], g['team_b']} == {pred['team_a'], pred['team_b']} and
                      g['round'] == pred['round']), None)
        if actual:
            if pred['winner'] == actual['actual_winner']:
                score += ESPN_SCORES[pred['round']]
                correct += 1
    return score, correct, len(predicted_games)

# Backtest on 2025 (we know Florida won)
print("\n" + "="*80)
print("BACKTESTING ON 2025 TOURNAMENT DATA (Florida was champion)")
print("="*80)

year = 2025
if year not in all_brackets or year not in team_metrics_dict:
    print(f"Data not available for {year}")
else:
    bracket_df = all_brackets[year]
    kenpon_df = team_metrics_dict[year]
    actual_games = extract_games_from_bracket(bracket_df)

    print(f"\nActual tournament had {len(actual_games)} games")
    print(f"Actual teams: {kenpon_df['Team'].tolist()[:10]}...")  # Show first 10

    # Build Tournament-Winner ensemble
    print("\nTraining Tournament-Winner ensemble...")
    tw_ensemble = build_tournament_winner_ensemble(year)
    tw_win_probs = predict_tournament_winner(tw_ensemble, year, kenpon_df)

    print(f"Top 5 TW favorites: {sorted(tw_win_probs.items(), key=lambda x: x[1], reverse=True)[:5]}")

    # Simulate tournament with both methods (deterministic for comparison)
    print("\nSimulating 2025 tournament with both methods (deterministic)...")

    # Tournament-Winner simulation
    tw_predictions = []
    for game in actual_games:
        winner = simulate_game_tw(game['team_a'], game['team_b'], tw_win_probs)
        tw_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    tw_score, tw_correct, tw_total = score_bracket(tw_predictions, actual_games)

    print(f"\nTournament-Winner Results:")
    print(f"  Correct Games: {tw_correct}/{tw_total}")
    print(f"  ESPN Score: {tw_score}/1920 ({tw_score/1920*100:.1f}%)")

    # Game-Level simulation
    gl_predictions = []
    for game in actual_games:
        winner = simulate_game_gl(game['team_a'], game['team_b'], year, game['round'],
                                  game_level_models, team_metrics_dict)
        gl_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    gl_score, gl_correct, gl_total = score_bracket(gl_predictions, actual_games)

    print(f"\nGame-Level Results:")
    print(f"  Correct Games: {gl_correct}/{gl_total}")
    print(f"  ESPN Score: {gl_score}/1920 ({gl_score/1920*100:.1f}%)")

    # Compare
    print(f"\n{'COMPARISON':-^80}")
    print(f"Tournament-Winner: {tw_score} points ({tw_correct}/{tw_total} games)")
    print(f"Game-Level:       {gl_score} points ({gl_correct}/{gl_total} games)")

    if gl_score > tw_score:
        print(f"✓ WINNER: Game-Level by {gl_score - tw_score} points")
    elif tw_score > gl_score:
        print(f"✓ WINNER: Tournament-Winner by {tw_score - gl_score} points")
    else:
        print(f"✓ TIE: Both methods scored equally")

print("\n" + "="*80)
print("TESTING MULTIPLE YEARS")
print("="*80)

backtest_results = []

for year in [2023, 2024, 2025]:
    if year not in all_brackets or year not in team_metrics_dict:
        print(f"Data not available for {year}")
        continue

    bracket_df = all_brackets[year]
    kenpon_df = team_metrics_dict[year]
    actual_games = extract_games_from_bracket(bracket_df)

    print(f"\n{year} Tournament ({len(actual_games)} games):")

    # Build Tournament-Winner ensemble
    tw_ensemble = build_tournament_winner_ensemble(year)
    tw_win_probs = predict_tournament_winner(tw_ensemble, year, kenpon_df)

    # Tournament-Winner simulation
    tw_predictions = []
    for game in actual_games:
        winner = simulate_game_tw(game['team_a'], game['team_b'], tw_win_probs)
        tw_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    tw_score, tw_correct, tw_total = score_bracket(tw_predictions, actual_games)

    # Game-Level simulation
    gl_predictions = []
    for game in actual_games:
        winner = simulate_game_gl(game['team_a'], game['team_b'], year, game['round'],
                                  game_level_models, team_metrics_dict)
        gl_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    gl_score, gl_correct, gl_total = score_bracket(gl_predictions, actual_games)

    print(f"  Tournament-Winner: {tw_score} points ({tw_correct}/{tw_total})")
    print(f"  Game-Level:       {gl_score} points ({gl_correct}/{gl_total})")
    print(f"  Winner: {'Game-Level' if gl_score > tw_score else 'Tournament-Winner' if tw_score > gl_score else 'Tie'} (+{abs(gl_score - tw_score)} points)")

    backtest_results.append({
        'year': year,
        'tw_score': tw_score,
        'gl_score': gl_score,
        'tw_correct': tw_correct,
        'gl_correct': gl_correct,
        'total_games': tw_total
    })

# Summary table
if backtest_results:
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL YEARS")
    print("="*80)
    summary_df = pd.DataFrame(backtest_results)
    summary_df['TW_Accuracy'] = (summary_df['tw_correct'] / summary_df['total_games'] * 100).round(1)
    summary_df['GL_Accuracy'] = (summary_df['gl_correct'] / summary_df['total_games'] * 100).round(1)
    summary_df['Point_Margin'] = (summary_df['gl_score'] - summary_df['tw_score']).astype(int)

    print("\n", summary_df[['year', 'tw_score', 'gl_score', 'TW_Accuracy', 'GL_Accuracy', 'Point_Margin']].to_string(index=False))

    total_tw = summary_df['tw_score'].sum()
    total_gl = summary_df['gl_score'].sum()
    print(f"\nTOTAL ACROSS {len(summary_df)} YEARS:")
    print(f"  Tournament-Winner: {total_tw} points")
    print(f"  Game-Level:       {total_gl} points")
    print(f"  Game-Level wins by: {total_gl - total_tw} points")

print("\n" + "="*80)
