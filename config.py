"""
Configuration file for Monte Carlo Portfolio Risk Simulator
Centralizes all configuration parameters for easy modification.
"""

from typing import Dict
import numpy as np

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Assets to analyze
TICKERS: list[str] = ["SPY", "BTC-USD", "GLD", "BND"]

# Historical data period (years)
# Note: BTC-USD data starts ~2014, so 8-10 years ensures complete data
PERIOD_YEARS: int = 12

# ============================================================================
# PORTFOLIO CONFIGURATIONS
# ============================================================================

# Portfolio allocation strategies
# Format: [SPY, BTC-USD, GLD, BND]
PORTFOLIO_CONFIGS: Dict[str, np.ndarray] = {
    "Bond Heavy": np.array([0.30, 0.00, 0.10, 0.60]),
    "Equity + Crypto": np.array([0.80, 0.20, 0.00, 0.00]),
    "Gold Heavy": np.array([0.20, 0.10, 0.50, 0.20]),
}

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Simulation horizon (trading days)
# 252 days = 1 year, 2520 days = 10 years
HORIZON_DAYS: int = 5475  # 15 years

# Number of Monte Carlo simulations
N_SIMULATIONS: int = 10_000

# Risk-free rate (annual)
RISK_FREE_RATE: float = 0.02  # 2%

# Random seed for reproducibility
RANDOM_SEED: int = 42

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Winsorization percentile (for outlier handling)
# Set to None to disable winsorization
WINSORIZATION_PERCENTILE: float | None = 0.01  # 1% winsorization

# Covariance matrix regularization (small value added to diagonal)
# Prevents Cholesky decomposition failures
COV_REGULARIZATION: float = 1e-8

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output directory
RESULTS_DIR: str = "results"

# Image export settings
IMAGE_WIDTH: int = 1600
IMAGE_HEIGHT: int = 900
IMAGE_SCALE: int = 3  # 300 DPI equivalent

# Export interactive HTML files (in addition to PNGs)
EXPORT_HTML: bool = True

