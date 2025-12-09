# Monte Carlo Portfolio Risk Simulator

A Python simulator for analyzing portfolio risk using Monte Carlo simulation with Geometric Brownian Motion (GBM) and correlated shocks (Cholesky decomposition). The code is fully vectorized with NumPy to handle 10,000 simulations efficiently without slow loops.

## Features

- **Modular Architecture**: Well-organized classes for data fetching, portfolio optimization, Monte Carlo simulation, and risk metrics
- **Vectorized Implementation**: Uses NumPy matrix operations for fast computation of 10,000+ simulations
- **Correlated Asset Modeling**: Implements Cholesky decomposition to simulate correlated price movements
- **Risk Metrics**: Calculates Value at Risk (VaR) and Conditional VaR (CVaR)
- **Interactive Visualizations**: Generates three Plotly charts for analysis

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `plotly`, `yfinance`

### Quick Installation

```bash
pip install numpy pandas plotly yfinance
```

## Usage

```bash
python portfolio_simulator.py
```

## What the Script Does

1. **Data Download**: Fetches 5 years of adjusted price data for SPY (S&P 500), BTC-USD (Bitcoin), and GLD (Gold ETF)
2. **Statistical Analysis**: Calculates logarithmic returns, mean returns, covariance matrix, and correlation matrix
3. **Monte Carlo Simulation**: Simulates 10,000 price paths over 1 year (252 trading days) using correlated GBM
4. **Risk Calculation**: Computes VaR 95% and CVaR 95% for the final portfolio return
5. **Visualizations**: Generates three interactive charts:
   - **Correlation Heatmap**: Shows how assets move together
   - **Spaghetti Plot**: Displays a sample of 100 portfolio evolution paths
   - **Return Distribution Histogram**: Shows the distribution of final returns with mean and VaR thresholds marked

## Portfolio Configuration

Default portfolio uses equal weights (1/3 each) for the three assets. You can modify the weights in the `main()` function of `portfolio_simulator.py`:

```python
weights = np.array([0.5, 0.3, 0.2])  # Custom weights for SPY, BTC-USD, GLD
```

## Technical Details

- **Geometric Brownian Motion**: Models asset prices using the standard financial model
- **Cholesky Decomposition**: Enables efficient generation of correlated random shocks
- **Vectorization**: All simulations run simultaneously using NumPy matrix operations
- **Logarithmic Returns**: Used for better statistical properties and GBM compatibility