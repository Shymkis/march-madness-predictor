"""
Integration code for game-level model with tournament bracket simulation.
To be used in analyze_data.ipynb
"""

import pickle
import numpy as np
import pandas as pd

def load_game_level_models():
    """Load precomputed game-level models."""
    with open("game_level_models.pkl", "rb") as f:
        return pickle.load(f)

def get_game_win_probability(team_a, team_b, year, round_num, models, team_metrics_dict):
    """
    Get win probability for Team A vs Team B using game-level model.
    
    Parameters
    ----------
    team_a, team_b : str
        Team names
    year : int
        Tournament year
    round_num : int
        Round number (1=Round of 64, ..., 6=Championship)
    models : dict
        Loaded models from load_game_level_models()
    team_metrics_dict : dict
        Mapping year -> team metrics dataframe
    
    Returns
    -------
    (prob_team_a_wins, prob_team_b_wins) : tuple
        Win probabilities (sums to 1.0)
    """
    win_prob_model = models["win_prob"]
    team_metrics = team_metrics_dict.get(year, None)
    
    if team_metrics is None or team_metrics.empty:
        return 0.5, 0.5
    
    # Get team data
    t_a = team_metrics[team_metrics["Team"] == team_a]
    t_b = team_metrics[team_metrics["Team"] == team_b]
    
    if t_a.empty or t_b.empty:
        return 0.5, 0.5
    
    t_a = t_a.iloc[0]
    t_b = t_b.iloc[0]
    
    # Build position-invariant features
    features_to_use = ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "Seed"]
    feat_dict = {"Round_Encoded": round_num}
    
    for feat in features_to_use:
        if feat not in team_metrics.columns:
            continue
        
        v_a = float(t_a[feat]) if pd.notna(t_a[feat]) else 0
        v_b = float(t_b[feat]) if pd.notna(t_b[feat]) else 0
        
        feat_dict[f"{feat}_Min"] = min(v_a, v_b)
        feat_dict[f"{feat}_Max"] = max(v_a, v_b)
        feat_dict[f"{feat}_Diff_Abs"] = abs(v_a - v_b)
        
        if v_a > 0 and v_b > 0:
            feat_dict[f"{feat}_Geo_Mean"] = (v_a * v_b) ** 0.5
            feat_dict[f"{feat}_Harmonic_Mean"] = 2 * v_a * v_b / (v_a + v_b)
        else:
            feat_dict[f"{feat}_Geo_Mean"] = 0
            feat_dict[f"{feat}_Harmonic_Mean"] = 0
    
    # Create feature vector
    feat_cols = win_prob_model["feature_cols"]
    X = np.array([[feat_dict.get(col, 0) for col in feat_cols]], dtype=float)
    X_scaled = win_prob_model["scaler"].transform(X)
    
    # Get predictions from both models
    rf_prob = win_prob_model["rf_model"].predict_proba(X_scaled)[0, 1]
    gb_prob = win_prob_model["gb_model"].predict_proba(X_scaled)[0, 1]
    
    # Blend models
    rf_weight = win_prob_model["rf_prauc"]
    gb_weight = win_prob_model["gb_prauc"]
    total = rf_weight + gb_weight
    prob_better_wins = (rf_prob * rf_weight + gb_prob * gb_weight) / total
    
    # Determine which team is "better" (lower seed, or higher NetRtg if tied)
    seed_a = float(t_a["Seed"]) if "Seed" in team_metrics.columns else 8
    seed_b = float(t_b["Seed"]) if "Seed" in team_metrics.columns else 8

    if seed_a != seed_b:
        team_a_is_better = seed_a < seed_b
    else:
        # Tiebreaker: higher NetRtg = better team
        netrtg_a = float(t_a["NetRtg"]) if "NetRtg" in team_metrics.columns else 0
        netrtg_b = float(t_b["NetRtg"]) if "NetRtg" in team_metrics.columns else 0
        team_a_is_better = netrtg_a >= netrtg_b
    
    if team_a_is_better:
        return prob_better_wins, 1 - prob_better_wins
    else:
        return 1 - prob_better_wins, prob_better_wins

# Example usage:
# 
# models = load_game_level_models()
# team_metrics_2026 = pd.read_csv("csv_files/2026.csv")
# team_metrics_dict = {2026: team_metrics_2026}
# 
# # Predict Duke vs Creighton in Round of 32
# p_duke, p_creighton = get_game_win_probability(
#     "Duke", "Creighton", 2026, 2, models, team_metrics_dict
# )
# print(f"Duke: {p_duke:.1%}, Creighton: {p_creighton:.1%}")

