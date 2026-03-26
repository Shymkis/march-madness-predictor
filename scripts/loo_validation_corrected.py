"""
Proper Leave-One-Out Cross-Validation (CORRECTED).
BOTH TW and GL models trained ONLY on years excluding the test year.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

ESPN_SCORES = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}

# Load all data
print("Loading all years...")
all_brackets = {}
team_metrics_dict = {}
for year in range(2002, 2026):
    try:
        all_brackets[year] = pd.read_csv(f"csv_files/{year}bracket.csv")
        team_metrics_dict[year] = pd.read_csv(f"csv_files/{year}.csv")
    except FileNotFoundError:
        pass

print(f"Available years: {sorted(all_brackets.keys())}\n")

def extract_games(bracket_df):
    """Extract games from bracket."""
    games = []
    for _, row in bracket_df.iterrows():
        games.append({
            'team_a': row['Team1'],
            'team_b': row['Team2'],
            'round': row['Round'],
            'actual_winner': row['Winner'],
        })
    return games

def build_gl_model_loo(train_years, all_brackets_dict, all_metrics_dict):
    """Build Game-Level model trained ONLY on specified years."""
    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]

    all_data = []
    y_list = []

    for y in train_years:
        if y not in all_brackets_dict or y not in all_metrics_dict:
            continue

        bracket_df = all_brackets_dict[y]
        kenpon_df = all_metrics_dict[y]

        for _, game in bracket_df.iterrows():
            row_a = kenpon_df[kenpon_df['Team'] == game['Team1']]
            row_b = kenpon_df[kenpon_df['Team'] == game['Team2']]

            if row_a.empty or row_b.empty:
                continue

            row_a = row_a.iloc[0]
            row_b = row_b.iloc[0]

            # Create position-invariant features (matching game_level_model.py)
            feat_dict = {"Round_Encoded": game['Round']}

            for feat in features_to_use:
                if feat not in kenpon_df.columns:
                    continue

                v_a = float(row_a[feat]) if pd.notna(row_a[feat]) else 0
                v_b = float(row_b[feat]) if pd.notna(row_b[feat]) else 0

                feat_dict[f"{feat}_Min"] = min(v_a, v_b)
                feat_dict[f"{feat}_Max"] = max(v_a, v_b)
                feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)

                if v_a > 0 and v_b > 0:
                    feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
                    feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
                else:
                    feat_dict[f"{feat}_Geo_Mean"] = 0
                    feat_dict[f"{feat}_Harmonic_Mean"] = 0

            all_data.append(feat_dict)
            # Target: did better-seeded team win?
            seed_a = float(row_a["Seed"]) if "Seed" in kenpon_df.columns and pd.notna(row_a["Seed"]) else 8
            seed_b = float(row_b["Seed"]) if "Seed" in kenpon_df.columns and pd.notna(row_b["Seed"]) else 8

            if seed_a != seed_b:
                better_seed_a = seed_a < seed_b
            else:
                netrtg_a = float(row_a["NetRtg"]) if pd.notna(row_a["NetRtg"]) else 0
                netrtg_b = float(row_b["NetRtg"]) if pd.notna(row_b["NetRtg"]) else 0
                better_seed_a = netrtg_a >= netrtg_b

            winner_is_a = game['Winner'] == game['Team1']
            y_list.append(1 if winner_is_a == better_seed_a else 0)

    if not all_data:
        return None

    X_df = pd.DataFrame(all_data)
    y = np.array(y_list)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Train RF and GB
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=150, random_state=42)

    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)

    # Calculate PR-AUC for weighting
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
    gb_probs = gb_model.predict_proba(X_scaled)[:, 1]

    precision_rf, recall_rf, _ = precision_recall_curve(y, rf_probs)
    precision_gb, recall_gb, _ = precision_recall_curve(y, gb_probs)

    rf_prauc = auc(recall_rf, precision_rf)
    gb_prauc = auc(recall_gb, precision_gb)

    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_cols': X_df.columns.tolist(),
        'rf_prauc': rf_prauc,
        'gb_prauc': gb_prauc
    }

def build_tw_ensemble_loo(train_years, all_brackets_dict, all_metrics_dict):
    """Build TW ensemble trained ONLY on specified years."""
    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]

    all_data = []
    y_list = []

    for y in train_years:
        if y not in all_brackets_dict or y not in all_metrics_dict:
            continue

        bracket_df = all_brackets_dict[y]
        kenpon_df = all_metrics_dict[y]

        for _, game in bracket_df.iterrows():
            row_a = kenpon_df[kenpon_df['Team'] == game['Team1']]
            row_b = kenpon_df[kenpon_df['Team'] == game['Team2']]

            if row_a.empty or row_b.empty:
                continue

            row_a = row_a.iloc[0]
            row_b = row_b.iloc[0]

            # Create paired features (same as GL)
            feat_dict = {}
            for feat in features_to_use:
                if feat not in kenpon_df.columns:
                    continue

                v_a = float(row_a[feat]) if pd.notna(row_a[feat]) else 0
                v_b = float(row_b[feat]) if pd.notna(row_b[feat]) else 0

                feat_dict[f"{feat}_Min"] = min(v_a, v_b)
                feat_dict[f"{feat}_Max"] = max(v_a, v_b)
                feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)

                if v_a > 0 and v_b > 0:
                    feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
                    feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
                else:
                    feat_dict[f"{feat}_Geo_Mean"] = 0
                    feat_dict[f"{feat}_Harmonic_Mean"] = 0

            all_data.append(feat_dict)
            y_list.append(1 if game['Winner'] == game['Team1'] else 0)

    if not all_data:
        return None

    X_df = pd.DataFrame(all_data)
    y = np.array(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {'lr': [], 'rf': [], 'svm': [], 'xgb': []}

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]

        models['lr'].append(LogisticRegression(max_iter=5000, random_state=42).fit(X_train, y_train))
        models['rf'].append(RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train))
        models['svm'].append(SVC(probability=True, random_state=42).fit(X_train, y_train))
        models['xgb'].append(XGBClassifier(random_state=42, verbosity=0).fit(X_train, y_train))

    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': X_df.columns.tolist()
    }

def predict_tw_matchup(team_a, team_b, year, ensemble, metrics_dict):
    """Predict TW probability."""
    if ensemble is None:
        return 0.5, 0.5

    kenpon_df = metrics_dict.get(year)
    if kenpon_df is None:
        return 0.5, 0.5

    row_a = kenpon_df[kenpon_df['Team'] == team_a]
    row_b = kenpon_df[kenpon_df['Team'] == team_b]

    if row_a.empty or row_b.empty:
        return 0.5, 0.5

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]
    feat_dict = {}

    for feat in features_to_use:
        if feat not in kenpon_df.columns:
            continue

        v_a = float(row_a[feat]) if pd.notna(row_a[feat]) else 0
        v_b = float(row_b[feat]) if pd.notna(row_b[feat]) else 0

        feat_dict[f"{feat}_Min"] = min(v_a, v_b)
        feat_dict[f"{feat}_Max"] = max(v_a, v_b)
        feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)

        if v_a > 0 and v_b > 0:
            feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
            feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
        else:
            feat_dict[f"{feat}_Geo_Mean"] = 0
            feat_dict[f"{feat}_Harmonic_Mean"] = 0

    X = pd.DataFrame([feat_dict])
    X_scaled = ensemble['scaler'].transform(X)

    probs_team_a = []
    for model_type in ['lr', 'rf', 'svm', 'xgb']:
        fold_probs = [m.predict_proba(X_scaled)[0, 1] for m in ensemble['models'][model_type]]
        probs_team_a.append(np.mean(fold_probs))

    p_a = np.mean(probs_team_a)
    return p_a, 1 - p_a

def predict_gl_matchup(team_a, team_b, year, round_num, gl_model, metrics_dict):
    """Predict GL probability."""
    if gl_model is None:
        return 0.5, 0.5

    kenpon_df = metrics_dict.get(year)
    if kenpon_df is None:
        return 0.5, 0.5

    row_a = kenpon_df[kenpon_df['Team'] == team_a]
    row_b = kenpon_df[kenpon_df['Team'] == team_b]

    if row_a.empty or row_b.empty:
        return 0.5, 0.5

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]
    feat_dict = {"Round_Encoded": round_num}

    for feat in features_to_use:
        if feat not in kenpon_df.columns:
            continue

        v_a = float(row_a[feat]) if pd.notna(row_a[feat]) else 0
        v_b = float(row_b[feat]) if pd.notna(row_b[feat]) else 0

        feat_dict[f"{feat}_Min"] = min(v_a, v_b)
        feat_dict[f"{feat}_Max"] = max(v_a, v_b)
        feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)

        if v_a > 0 and v_b > 0:
            feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
            feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
        else:
            feat_dict[f"{feat}_Geo_Mean"] = 0
            feat_dict[f"{feat}_Harmonic_Mean"] = 0

    X = pd.DataFrame([feat_dict])
    X_scaled = gl_model['scaler'].transform(X)

    rf_prob = gl_model['rf_model'].predict_proba(X_scaled)[0, 1]
    gb_prob = gl_model['gb_model'].predict_proba(X_scaled)[0, 1]

    rf_weight = gl_model['rf_prauc']
    gb_weight = gl_model['gb_prauc']
    total = rf_weight + gb_weight
    prob_better_wins = (rf_prob * rf_weight + gb_prob * gb_weight) / total

    # Which team is "better"?
    seed_a = float(row_a["Seed"]) if "Seed" in kenpon_df.columns and pd.notna(row_a["Seed"]) else 8
    seed_b = float(row_b["Seed"]) if "Seed" in kenpon_df.columns and pd.notna(row_b["Seed"]) else 8

    if seed_a != seed_b:
        team_a_is_better = seed_a < seed_b
    else:
        netrtg_a = float(row_a["NetRtg"]) if pd.notna(row_a["NetRtg"]) else 0
        netrtg_b = float(row_b["NetRtg"]) if pd.notna(row_b["NetRtg"]) else 0
        team_a_is_better = netrtg_a >= netrtg_b

    if team_a_is_better:
        return prob_better_wins, 1 - prob_better_wins
    else:
        return 1 - prob_better_wins, prob_better_wins

def score_bracket(predicted_games, actual_games):
    """Score bracket."""
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

# LOO CV
print("="*80)
print("LEAVE-ONE-OUT CROSS-VALIDATION (CORRECTED - BOTH METHODS RETRAINED)")
print("="*80)

loo_results = []
test_years = sorted([y for y in all_brackets.keys() if y >= 2010])

for test_year in test_years:
    train_years = [y for y in all_brackets.keys() if y != test_year and y < test_year]

    if len(train_years) < 5:
        continue

    bracket_df = all_brackets[test_year]
    actual_games = extract_games(bracket_df)

    print(f"\n{'='*80}")
    print(f"TEST YEAR: {test_year} ({len(actual_games)} games)")
    print(f"TRAINING: {len(train_years)} years ({min(train_years)}-{max(train_years)})")
    print(f"{'='*80}")

    # Build both models on training years only
    tw_ensemble = build_tw_ensemble_loo(train_years, all_brackets, team_metrics_dict)
    gl_model = build_gl_model_loo(train_years, all_brackets, team_metrics_dict)

    if tw_ensemble is None or gl_model is None:
        print(f"  Could not build models")
        continue

    # Predict with TW
    tw_predictions = []
    for game in actual_games:
        p_a, p_b = predict_tw_matchup(game['team_a'], game['team_b'], test_year, tw_ensemble, team_metrics_dict)
        winner = game['team_a'] if p_a > 0.5 else game['team_b']
        tw_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    tw_score, tw_correct, tw_total = score_bracket(tw_predictions, actual_games)

    # Predict with GL
    gl_predictions = []
    for game in actual_games:
        p_a, p_b = predict_gl_matchup(game['team_a'], game['team_b'], test_year, game['round'],
                                      gl_model, team_metrics_dict)
        winner = game['team_a'] if p_a > 0.5 else game['team_b']
        gl_predictions.append({
            'team_a': game['team_a'],
            'team_b': game['team_b'],
            'round': game['round'],
            'winner': winner
        })

    gl_score, gl_correct, gl_total = score_bracket(gl_predictions, actual_games)

    print(f"Tournament-Winner: {tw_correct}/{tw_total} ({tw_correct/tw_total*100:.1f}%), {tw_score} pts")
    print(f"Game-Level:       {gl_correct}/{gl_total} ({gl_correct/gl_total*100:.1f}%), {gl_score} pts")

    margin = gl_score - tw_score
    margin_str = f"GL +{margin}" if margin >= 0 else f"TW +{-margin}"
    print(f"Winner: {margin_str}")

    loo_results.append({
        'year': test_year,
        'tw_score': tw_score,
        'gl_score': gl_score,
        'tw_correct': tw_correct,
        'gl_correct': gl_correct,
        'total': tw_total,
        'tw_pct': tw_correct / tw_total * 100,
        'gl_pct': gl_correct / gl_total * 100
    })

# Summary
if loo_results:
    print("\n" + "="*80)
    print("SUMMARY - BOTH METHODS TRAINED PER FOLD")
    print("="*80)

    summary_df = pd.DataFrame(loo_results)
    summary_df['Margin'] = (summary_df['gl_score'] - summary_df['tw_score']).astype(int)
    summary_df['Winner'] = summary_df.apply(lambda r: 'GL' if r['gl_score'] > r['tw_score'] else ('TW' if r['tw_score'] > r['gl_score'] else 'TIE'), axis=1)

    print("\n", summary_df[['year', 'tw_correct', 'gl_correct', 'tw_pct', 'gl_pct', 'tw_score', 'gl_score', 'Margin', 'Winner']].to_string(index=False))

    print("\n" + "-"*80)
    total_games = summary_df['total'].sum()
    total_tw_correct = summary_df['tw_correct'].sum()
    total_gl_correct = summary_df['gl_correct'].sum()
    total_tw_score = summary_df['tw_score'].sum()
    total_gl_score = summary_df['gl_score'].sum()

    print(f"TOTAL ({len(summary_df)} years):")
    print(f"  Tournament-Winner: {total_tw_correct}/{total_games} ({total_tw_correct/total_games*100:.1f}%), {total_tw_score} points")
    print(f"  Game-Level:       {total_gl_correct}/{total_games} ({total_gl_correct/total_games*100:.1f}%), {total_gl_score} points")
    print(f"  Difference:       {total_gl_correct-total_tw_correct:+d} games, {total_gl_score-total_tw_score:+d} points")

    tw_wins = (summary_df['tw_score'] > summary_df['gl_score']).sum()
    gl_wins = (summary_df['gl_score'] > summary_df['tw_score']).sum()
    ties = (summary_df['tw_score'] == summary_df['gl_score']).sum()

    print(f"\nYear-by-year wins:")
    print(f"  TW wins: {tw_wins}")
    print(f"  GL wins: {gl_wins}")
    print(f"  Ties:    {ties}")

print("\n" + "="*80)
