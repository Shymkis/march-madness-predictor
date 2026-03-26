# Scripts

This folder contains utility and training scripts. Not needed for routine bracket predictions, but useful for model validation and portfolio generation.

## Files

- **game_level_model.py**: Training script to rebuild game-level models from scratch
  - Loads all bracket CSV files (2002-2025)
  - Creates 31 position-invariant features from 6 KenPom metrics
  - Trains Random Forest (200 trees) and Gradient Boosting (150 estimators)
  - Outputs: `../models/game_level_models.pkl`
  - **Usage**: `python game_level_model.py` (from project root, or adjust paths)

- **game_level_portfolio_generator.py**: Generates diverse bracket portfolios using game-level model
  - Creates 40 Sweet 16→Finals brackets with varying determinism
  - Outputs: `../results/2026/game_level_portfolios.csv`
  - **Usage**: `python scripts/game_level_portfolio_generator.py`

- **loo_validation_corrected.py**: Leave-One-Out cross-validation test
  - Tests both TW and GL methods under proper isolation
  - Trains each method only on years excluding the test year
  - Provides performance comparison across 2010-2025
  - Reference for understanding method performance
  - **Usage**: `python scripts/loo_validation_corrected.py` (informational only)

## When to Use

- **game_level_model.py**: Only if you need to retrain the model with new data
- **game_level_portfolio_generator.py**: To create new portfolios for ESP bracket contests
- **loo_validation_corrected.py**: For validation/research, not part of prediction pipeline
