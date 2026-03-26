"""
Game-Level Predictive Model for March Madness

Builds position-invariant models for:
1. Win probability in head-to-head matchups
2. Score differential prediction

Uses historical bracket data (2002-2025) combined with KenPom team metrics.
"""

import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_recall_curve, mean_squared_error
from pathlib import Path

SEED = 42
DATA_DIR = Path("csv_files")

def load_all_bracket_data():
    """Load and combine all historical bracket games (2002-2025)."""
    bracket_files = sorted(DATA_DIR.glob("*bracket.csv"))
    games = []

    for bracket_file in bracket_files:
        year = int(bracket_file.stem.split("bracket")[0])
        df = pd.read_csv(bracket_file)
        df["Year"] = year
        games.append(df)

    return pd.concat(games, ignore_index=True)

def load_team_metrics(year):
    """Load KenPom team metrics for a given year."""
    csv_file = DATA_DIR / f"{year}.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return pd.DataFrame()

def create_game_features(games_df):
    """
    Create TRUE position-invariant features for each game.

    Position-invariant means: features don't depend on team order at all.
    We create features like:
    - min/max of metrics (symmetric)
    - geometric mean of metrics (symmetric)
    - Absolute differences (symmetric)

    The target "Team1_Won" will be naturally balanced by always recording
    the higher-seed or alphabetically-first team as "Team1".
    """
    all_games = []

    for _, game in games_df.iterrows():
        year = game["Year"]
        team1 = game["Team1"]
        team2 = game["Team2"]
        round_num = game["Round"]

        metrics = load_team_metrics(year)
        if metrics.empty:
            continue

        t1_data = metrics[metrics["Team"] == team1].iloc[0] if team1 in metrics["Team"].values else None
        t2_data = metrics[metrics["Team"] == team2].iloc[0] if team2 in metrics["Team"].values else None

        if t1_data is None or t2_data is None:
            continue

        features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]

        game_features = {
            "Team1": team1,
            "Team2": team2,
            "Year": year,
            "Round": round_num,
            "Winner": game["Winner"],
            "Team1_Won": 1 if game["Winner"] == team1 else 0,
            "Score1": game["Team1 Score"],
            "Score2": game["Team2 Score"],
            "Score_Diff": abs(game["Team1 Score"] - game["Team2 Score"]),  # Absolute difference for symmetry
            "Higher_Seed_Won": 1 if (
                (float(t1_data["Seed"]) < float(t2_data["Seed"]) and game["Winner"] == team1) or
                (float(t2_data["Seed"]) < float(t1_data["Seed"]) and game["Winner"] == team2)
            ) else 0
        }

        # Create truly position-invariant features
        for feat in features_to_use:
            if feat not in metrics.columns:
                continue

            v1 = float(t1_data[feat]) if pd.notna(t1_data[feat]) else 0
            v2 = float(t2_data[feat]) if pd.notna(t2_data[feat]) else 0

            # Symmetric operations only
            game_features[f"{feat}_Min"] = min(v1, v2)
            game_features[f"{feat}_Max"] = max(v1, v2)
            game_features[f"{feat}_Diff_Abs"] = abs(v1 - v2)

            # Geometric mean (symmetric)
            if v1 > 0 and v2 > 0:
                game_features[f"{feat}_Geo_Mean"] = (v1 * v2) ** 0.5
            else:
                game_features[f"{feat}_Geo_Mean"] = 0

            # Harmonic mean (symmetric)
            if v1 > 0 and v2 > 0:
                game_features[f"{feat}_Harmonic_Mean"] = 2 * v1 * v2 / (v1 + v2)
            else:
                game_features[f"{feat}_Harmonic_Mean"] = 0

        # Add round encoding (not position-dependent)
        game_features["Round_Encoded"] = round_num

        all_games.append(game_features)

    return pd.DataFrame(all_games)

def build_win_probability_model(game_features_df):
    """
    Build a model predicting win probability (who wins the game).
    Predicts: does the better-seeded team win?
    Returns trained ensemble model and feature columns used.
    """
    # Select feature columns
    feat_cols = [col for col in game_features_df.columns
                 if col.endswith(("_Min", "_Max", "_Diff_Abs", "_Geo_Mean", "_Harmonic_Mean", "_Encoded"))]

    X = game_features_df[feat_cols].fillna(0)
    # Use "Higher_Seed_Won" for position invariance
    y = game_features_df["Higher_Seed_Won"]

    # Handle class imbalance
    print(f"\nWin probability model:")
    print(f"  Training samples: {len(X)}")
    print(f"  Better team wins: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Features used: {len(feat_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ensemble: Random Forest + Gradient Boosting for robustness
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=SEED, max_features='sqrt')
    gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=SEED, subsample=0.8)

    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)

    # Compute PR-AUC for each
    rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
    gb_probs = gb_model.predict_proba(X_scaled)[:, 1]

    rf_prec, rf_rec, _ = precision_recall_curve(y, rf_probs)
    rf_prauc = auc(rf_rec, rf_prec)
    gb_prec, gb_rec, _ = precision_recall_curve(y, gb_probs)
    gb_prauc = auc(gb_rec, gb_prec)

    print(f"  RF PR-AUC: {rf_prauc:.4f}")
    print(f"  GB PR-AUC: {gb_prauc:.4f}")

    return {
        "rf_model": rf_model,
        "gb_model": gb_model,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "rf_prauc": rf_prauc,
        "gb_prauc": gb_prauc
    }

def build_score_differential_model(game_features_df):
    """
    Build a model predicting score differential (Team1 Score - Team2 Score).
    Useful for calibrating final score predictions.
    """
    feat_cols = [col for col in game_features_df.columns
                 if col.endswith(("_Min", "_Max", "_Diff", "_Ratio_Log", "_Encoded"))]

    X = game_features_df[feat_cols].fillna(0)
    y = game_features_df["Score_Diff"].fillna(10)  # Fill missing with median (~11)

    # Remove any remaining NaN
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"\nScore differential model:")
    print(f"  Training samples: {len(X)}")
    print(f"  Mean score diff: {y.mean():.2f} ± {y.std():.2f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use gradient boosting for regression (handles heteroscedastic residuals better)
    model = GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=SEED)
    model.fit(X_scaled, y)

    # Evaluate
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  RMSE: {rmse:.2f}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "rmse": rmse
    }

def predict_game_winner(team_a, team_b, year, round_num, win_prob_model, team_metrics_dict):
    """
    Predict win probability for Team A vs Team B at a given round.

    Position-invariant: always returns (prob_team_a_wins, prob_team_b_wins)
    Regardless of which team had better metrics.

    Returns: (prob_team_a_wins, prob_team_b_wins)
    """
    metrics = team_metrics_dict.get(year, pd.DataFrame())
    if metrics.empty:
        return 0.5, 0.5

    t_a_data = metrics[metrics["Team"] == team_a]
    t_b_data = metrics[metrics["Team"] == team_b]

    if t_a_data.empty or t_b_data.empty:
        return 0.5, 0.5

    t_a_data = t_a_data.iloc[0]
    t_b_data = t_b_data.iloc[0]

    # Create truly position-invariant features
    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]
    feat_dict = {"Round_Encoded": round_num}

    for feat in features_to_use:
        if feat not in metrics.columns:
            continue

        v_a = float(t_a_data[feat]) if pd.notna(t_a_data[feat]) else 0
        v_b = float(t_b_data[feat]) if pd.notna(t_b_data[feat]) else 0

        # Symmetric features only
        feat_dict[f"{feat}_Min"] = min(v_a, v_b)
        feat_dict[f"{feat}_Max"] = max(v_a, v_b)
        feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)

        if v_a > 0 and v_b > 0:
            feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
        else:
            feat_dict[f"{feat}_Geo_Mean"] = 0

        if v_a > 0 and v_b > 0:
            feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
        else:
            feat_dict[f"{feat}_Harmonic_Mean"] = 0

    # Create feature vector in same order as training
    feat_cols = win_prob_model["feature_cols"]
    X = np.array([[feat_dict.get(col, 0) for col in feat_cols]], dtype=float)
    X_scaled = win_prob_model["scaler"].transform(X)

    # Get prediction (prob that higher-seed/better team wins)
    rf_prob = win_prob_model["rf_model"].predict_proba(X_scaled)[0, 1]
    gb_prob = win_prob_model["gb_model"].predict_proba(X_scaled)[0, 1]

    rf_weight = win_prob_model["rf_prauc"]
    gb_weight = win_prob_model["gb_prauc"]
    total = rf_weight + gb_weight

    prob_better_team_wins = (rf_prob * rf_weight + gb_prob * gb_weight) / total

    # Determine which team is "better" (higher seed = lower seed number)
    seed_a = float(t_a_data["Seed"]) if "Seed" in metrics.columns else 8
    seed_b = float(t_b_data["Seed"]) if "Seed" in metrics.columns else 8

    # Better team = lower seed number
    team_a_is_better = seed_a < seed_b

    if team_a_is_better:
        return prob_better_team_wins, 1 - prob_better_team_wins
    else:
        return 1 - prob_better_team_wins, prob_better_team_wins

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("GAME-LEVEL PREDICTIVE MODEL")
    print("="*80)

    # Load data
    print("\nLoading historical bracket games...")
    games_df = load_all_bracket_data()
    print(f"Loaded {len(games_df)} games from {games_df['Year'].nunique()} years")

    # Create features
    print("\nCreating position-invariant features...")
    game_features_df = create_game_features(games_df)
    print(f"Created {len(game_features_df)} complete game records with features")
    print(f"Features: {[col for col in game_features_df.columns if col.endswith(('_Min', '_Max', '_Diff', '_Ratio_Log'))][:5]}...")

    # Build models
    print("\n" + "="*80)
    print("BUILDING WIN PROBABILITY MODEL")
    print("="*80)
    win_prob_model = build_win_probability_model(game_features_df)

    print("\n" + "="*80)
    print("BUILDING SCORE DIFFERENTIAL MODEL")
    print("="*80)
    score_diff_model = build_score_differential_model(game_features_df)

    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    import pickle

    with open("game_level_models.pkl", "wb") as f:
        pickle.dump({
            "win_prob": win_prob_model,
            "score_diff": score_diff_model,
            "game_features": game_features_df
        }, f)

    print("Models saved to game_level_models.pkl")

    # Example: Test position-invariance
    print("\n" + "="*80)
    print("TESTING POSITION INVARIANCE")
    print("="*80)

    # Load team metrics for examples
    team_metrics_dict = {}
    for year in range(2002, 2026):
        metrics = load_team_metrics(year)
        if not metrics.empty:
            team_metrics_dict[year] = metrics

    # Test: Pick a recent game and swap team order
    recent_games = game_features_df[game_features_df["Year"] == 2025].head(5)

    for _, game in recent_games.iterrows():
        team1 = game["Team1"]
        team2 = game["Team2"]
        year = int(game["Year"])
        round_num = int(game["Round"])

        # Forward: Team1 vs Team2
        p1_fwd, p2_fwd = predict_game_winner(team1, team2, year, round_num, win_prob_model, team_metrics_dict)

        # Reverse: Team2 vs Team1
        p2_rev, p1_rev = predict_game_winner(team2, team1, year, round_num, win_prob_model, team_metrics_dict)

        print(f"\n{team1} ({p1_fwd:.3f}) vs {team2} ({p2_fwd:.3f})")
        print(f"  Reversed: {team2} ({p2_rev:.3f}) vs {team1} ({p1_rev:.3f})")
        print(f"  Invariant? {abs(p1_fwd - p1_rev) < 0.01 and abs(p2_fwd - p2_rev) < 0.01}")
