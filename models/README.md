# Game-Level Model

This folder contains the trained game-level prediction model and integration code.

## Files

- **game_level_models.pkl**: Pre-trained Random Forest + Gradient Boosting ensemble (trained on 1,503 games from 2002-2025 bracket data)
  - Contains: RF model, GB model, scaler, feature columns, PR-AUC weights for blending

- **game_level_integration.py**: Python module for using the model to predict game outcomes
  - `load_game_level_models()`: Load the pickle file
  - `get_game_win_probability(team_a, team_b, year, round_num, models, team_metrics_dict)`: Get prediction for a specific matchup

- **README_PREDICTION_METHODS.md**: Comprehensive guide comparing game-level vs tournament-winner approaches
  - Feature engineering details
  - Model validation & position invariance verification
  - Usage patterns and examples

## Usage in Notebook

```python
from models.game_level_integration import load_game_level_models, get_game_win_probability

models = load_game_level_models()
p_team_a, p_team_b = get_game_win_probability("Duke", "Iowa", 2026, 2, models, team_metrics_dict)
```

## Model Performance

**Leave-One-Out Validation (2010-2025)**:
- Accuracy: 71.5% (675/944 games correct)
- Tournament-Winner approach: 97.6% (for reference)

**Note**: Game-Level model is position-invariant and captures opponent-specific matchup dynamics. Tournament-Winner using full KenPom metrics shows superior performance on historical bracket data.
