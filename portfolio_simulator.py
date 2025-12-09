"""
Monte Carlo portfolio risk simulator using vectorized NumPy operations.

Components
- DataFetcher: downloads historical data and computes log-returns.
- PortfolioOptimizer: estimates mean returns, covariance, and correlation.
- MonteCarloSimulator: generates correlated GBM price paths.
- RiskMetrics: computes VaR and CVaR on simulated portfolio outcomes.
- Plot helpers: correlation heatmap, spaghetti plot, histogram of final returns.

Run:
    python portfolio_simulator.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

TickerList = list[str]


class DataFetcher:
    """
    Downloads historical price data and computes logarithmic returns.
    
    Logarithmic returns are preferred in quantitative finance because:
    1. They are additive over time: log(S_t/S_0) = sum(log(S_i/S_{i-1}))
    2. They better model price distributions with Geometric Brownian Motion (GBM)
    3. They are symmetric: log(1/r) = -log(r)
    """

    def __init__(self, tickers: TickerList, period_years: int = 5) -> None:
        """
        Initialize the data fetcher with ticker symbols and historical period.
        
        Args:
            tickers: List of asset symbols (e.g., ['SPY', 'BTC-USD', 'GLD'])
            period_years: Number of years of historical data to download
        """
        self.tickers = tickers
        self.period_years = period_years

    def fetch_prices(self) -> pd.DataFrame:
        """
        Download historical adjusted prices for the specified assets.
        
        Uses yfinance to fetch data. Adjusted prices account for splits and dividends,
        making them suitable for return calculations.
        
        Returns:
            DataFrame with adjusted prices. Each column represents an asset,
            each row represents a trading date.
            
        Raises:
            ValueError: If no data could be downloaded
            KeyError: If expected price columns are not found
        """
        # Download historical data using yfinance
        # auto_adjust=False keeps 'Adj Close' column explicitly available
        prices = yf.download(
            tickers=self.tickers,
            period=f"{self.period_years}y",
            auto_adjust=False,  # Keep 'Adj Close' column available
            progress=False,  # Suppress progress bar
        )
        
        # Validate that data was successfully downloaded
        if prices.empty:
            raise ValueError("No price data downloaded; check tickers or connectivity.")

        # yfinance returns a MultiIndex DataFrame (field, ticker) when multiple tickers are requested
        # We need to extract only the 'Adj Close' or 'Close' column
        if isinstance(prices.columns, pd.MultiIndex):
            # Multiple tickers case: MultiIndex structure
            if ("Adj Close" in prices.columns.get_level_values(0)):
                # Extract 'Adj Close' column for all tickers
                prices = prices["Adj Close"]
            elif ("Close" in prices.columns.get_level_values(0)):
                # Fallback to 'Close' if 'Adj Close' is not available
                prices = prices["Close"]
            else:
                raise KeyError("Expected 'Adj Close' or 'Close' columns in download.")
        else:
            # Single ticker case: simple column structure
            if "Adj Close" in prices.columns:
                # Rename column with ticker name for consistency
                prices = prices.rename(columns={"Adj Close": self.tickers[0]})
            elif "Close" in prices.columns:
                # Fallback to 'Close' if 'Adj Close' is not available
                prices = prices.rename(columns={"Close": self.tickers[0]})
            else:
                raise KeyError("Expected 'Adj Close' or 'Close' columns in download.")

        # Remove rows with missing values (NaN) to ensure clean data
        return prices.dropna()

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily logarithmic returns: log(P_t / P_{t-1}).
        
        Formula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
        
        Args:
            prices: DataFrame with historical prices
            
        Returns:
            DataFrame with logarithmic returns (first row will be NaN and is dropped)
        """
        # shift(1) shifts prices one day backward
        # Element-wise division calculates P_t / P_{t-1}
        # np.log applies natural logarithm
        # dropna() removes the first row which becomes NaN
        return np.log(prices / prices.shift(1)).dropna()


class PortfolioOptimizer:
    """
    Estimates key portfolio statistics from historical logarithmic returns.
    
    Calculates:
    - Mean daily returns (drift): estimation of expected return per day
    - Covariance matrix: measures how assets vary together
    - Correlation matrix: normalizes covariance to values between -1 and 1
    """

    def __init__(self, log_returns: pd.DataFrame) -> None:
        """
        Initialize the optimizer by calculating all necessary statistics.
        
        Args:
            log_returns: DataFrame with historical logarithmic returns
        """
        self.log_returns = log_returns
        
        # Mean daily return per asset (GBM drift parameter)
        # Used as estimation of expected return μ in the GBM formula
        self.mean_returns = log_returns.mean().to_numpy()
        
        # Daily covariance matrix: Σ[i,j] = Cov(r_i, r_j)
        # Necessary to simulate correlated movements between assets
        self.cov_matrix = log_returns.cov().to_numpy()
        
        # Correlation matrix: ρ[i,j] = Corr(r_i, r_j) = Σ[i,j] / (σ_i * σ_j)
        # Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation)
        self.corr_matrix = log_returns.corr()


class MonteCarloSimulator:
    """
    Vectorized Geometric Brownian Motion (GBM) simulator with correlated shocks.
    
    The GBM is the standard model for modeling financial asset prices:
        dS_t = μ * S_t * dt + σ * S_t * dW_t
    
    In logarithmic form and discretized for daily steps:
        log(S_t) = log(S_{t-1}) + (μ - 0.5*σ²)*dt + L @ z
    
    Where:
    - μ: mean daily return (drift)
    - σ²: daily variance (diagonal of covariance matrix)
    - L: Cholesky factor of covariance matrix (enables correlation)
    - z: vector of independent random shocks ~ N(0, I)
    
    Vectorization allows simulating all paths simultaneously without slow loops.
    """

    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        start_prices: np.ndarray,
        steps: int = 252,
        n_sims: int = 10_000,
    ) -> None:
        """
        Initialize the simulator with necessary parameters.
        
        Args:
            mean_returns: 1D array with mean daily returns per asset
            cov_matrix: Daily covariance matrix (n_assets x n_assets)
            start_prices: Initial prices for each asset
            steps: Number of trading days in the horizon (252 = 1 year)
            n_sims: Number of Monte Carlo simulations to perform
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.start_prices = start_prices
        self.steps = steps
        self.n_sims = n_sims
        self.n_assets = start_prices.shape[0]
        
        # Cholesky decomposition: cov_matrix = L @ L.T
        # Allows generating correlated shocks from independent shocks
        # If z ~ N(0, I), then L @ z ~ N(0, cov_matrix)
        self.cholesky = np.linalg.cholesky(cov_matrix)

    def simulate_paths(self, seed: int | None = 42) -> np.ndarray:
        """
        Generate simulated price paths using vectorized GBM.
        
        The implementation is fully vectorized:
        - Generates all random shocks at once
        - Applies correlation through matrix multiplication
        - Calculates all paths simultaneously
        
        Args:
            seed: Random seed for reproducibility (None = random)
            
        Returns:
            3D array of shape (n_sims, steps+1, n_assets)
            - First dimension: simulation number
            - Second dimension: day (includes initial day)
            - Third dimension: asset
        """
        # Random number generator with seed for reproducibility
        rng = np.random.default_rng(seed)

        # Generate independent random shocks: (simulations, days, assets)
        # Each element follows a standard normal distribution N(0, 1)
        shocks = rng.standard_normal((self.n_sims, self.steps, self.n_assets))
        
        # Apply correlation through matrix multiplication with Cholesky factor
        # shocks @ cholesky.T transforms independent shocks into correlated ones
        # This operation is fully vectorized: all simulations at once
        correlated_shocks = shocks @ self.cholesky.T

        # Calculate adjusted drift: μ - 0.5*σ²
        # The -0.5*σ² term comes from Itô's lemma applied to the logarithm
        # np.diag extracts variances (diagonal of covariance matrix)
        drift = self.mean_returns - 0.5 * np.diag(self.cov_matrix)
        
        # Reshape for broadcasting: (1, 1, n_assets)
        # Allows adding drift to all days and simulations
        drift = drift.reshape(1, 1, self.n_assets)

        # Calculate logarithmic returns per step: drift + correlated shock
        log_returns = drift + correlated_shocks
        
        # Cumulative sum along the days axis (axis=1)
        # Builds the cumulative GBM exponent: log(S_t) - log(S_0)
        log_paths = np.cumsum(log_returns, axis=1)

        # Initialize price array: (simulations, days+1, assets)
        prices = np.empty((self.n_sims, self.steps + 1, self.n_assets))
        
        # Set initial prices (day 0)
        prices[:, 0, :] = self.start_prices
        
        # Calculate future prices: S_t = S_0 * exp(log(S_t) - log(S_0))
        # prices[:, 1:, :] contains prices from day 1 to the end
        prices[:, 1:, :] = np.exp(np.log(self.start_prices) + log_paths)
        
        return prices


class RiskMetrics:
    """
    Computes portfolio-level Value at Risk (VaR) and Conditional VaR (CVaR) 
    from simulated price paths.
    
    VaR: Maximum expected loss at a given confidence level (e.g., 95%)
    CVaR: Expected loss conditional on exceeding VaR threshold
    """

    def __init__(self, weights: np.ndarray) -> None:
        """
        Initialize risk metrics calculator with portfolio weights.
        
        Args:
            weights: Array of portfolio weights (must sum to 1.0)
        """
        self.weights = weights

    def portfolio_growth(self, price_paths: np.ndarray) -> np.ndarray:
        """
        Calculate portfolio value growth path from individual asset price paths.
        
        Computes weighted portfolio returns and accumulates them to get the
        portfolio value evolution over time.
        
        Args:
            price_paths: 3D array of shape (n_sims, days+1, n_assets) with price paths
            
        Returns:
            2D array of shape (n_sims, days) with portfolio value paths
            (normalized to start at 1.0)
        """
        # Calculate daily returns for each asset: (P_t / P_{t-1}) - 1
        asset_returns = price_paths[:, 1:, :] / price_paths[:, :-1, :] - 1
        
        # Weighted daily returns: vectorized dot product over the asset axis
        # np.tensordot performs weighted sum: sum(w_i * r_i) for each day and simulation
        portfolio_returns = np.tensordot(
            asset_returns, self.weights, axes=([2], [0])
        )
        
        # Cumulative product of (1 + r_t) gives portfolio value path
        # Each element represents portfolio value relative to initial value
        return np.cumprod(1 + portfolio_returns, axis=1)

    def var_cvar(self, final_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR) from final returns.
        
        Args:
            final_returns: Array of final portfolio returns from all simulations
            alpha: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            Tuple of (VaR, CVaR) values
        """
        # Calculate percentile cutoff (e.g., 5th percentile for 95% VaR)
        cutoff = 100 * (1 - alpha)
        
        # VaR is the return at the cutoff percentile (negative = loss)
        var = np.percentile(final_returns, cutoff)
        
        # CVaR is the mean of all returns that are worse than VaR
        cvar = final_returns[final_returns <= var].mean()
        
        return var, cvar


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    """
    Create an interactive correlation heatmap visualization.
    
    Shows how assets move together. Values range from -1 (perfect negative correlation)
    to +1 (perfect positive correlation). Red indicates positive correlation,
    blue indicates negative correlation.
    
    Args:
        corr: DataFrame with correlation matrix between assets
    """
    fig = px.imshow(
        corr,
        text_auto=".2f",  # Display correlation values with 2 decimal places
        color_continuous_scale="RdBu",  # Red-Blue color scale
        origin="lower",  # Origin at bottom-left
        title="Correlation Heatmap",
    )
    fig.update_layout(width=600, height=500)
    fig.show()


def plot_spaghetti(portfolio_growth: np.ndarray, n_paths: int = 100) -> None:
    """
    Create a spaghetti plot showing a sample of Monte Carlo simulation paths.
    
    Visualizes the dispersion and variability of portfolio evolution over time.
    Each line represents one simulated path of portfolio value.
    
    Args:
        portfolio_growth: 2D array of shape (n_sims, days) with portfolio value paths
        n_paths: Number of paths to display (default: 100 for clarity)
    """
    # Limit number of paths to display (avoid overcrowding)
    n_paths = min(n_paths, portfolio_growth.shape[0])
    
    # Randomly select paths to display
    idx = np.random.choice(portfolio_growth.shape[0], size=n_paths, replace=False)
    
    fig = go.Figure()
    # Add each selected path as a line trace
    for i in idx:
        fig.add_trace(
            go.Scatter(
                y=portfolio_growth[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.6,  # Semi-transparent for better visibility of overlapping paths
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Monte Carlo Portfolio Paths (sample)",
        xaxis_title="Days",
        yaxis_title="Portfolio Value (start = 1)",
    )
    fig.show()


def plot_final_distribution(final_returns: np.ndarray, mean_ret: float, var95: float) -> None:
    """
    Create a histogram of final portfolio returns with risk metrics marked.
    
    Shows the distribution of outcomes from all simulations, with vertical lines
    indicating the mean return and VaR threshold.
    
    Args:
        final_returns: Array of final portfolio returns from all simulations
        mean_ret: Mean of final returns
        var95: Value at Risk at 95% confidence level
    """
    fig = go.Figure()
    # Create histogram of final returns
    fig.add_trace(
        go.Histogram(
            x=final_returns,
            nbinsx=60,  # Number of bins for histogram
            name="Final Returns",
            marker=dict(color="#4C78A8"),
            opacity=0.75,
        )
    )
    # Add vertical line for mean return (green, dashed)
    fig.add_vline(x=mean_ret, line=dict(color="green", dash="dash"), annotation_text="Mean")
    # Add vertical line for VaR threshold (red, dashed)
    fig.add_vline(x=var95, line=dict(color="red", dash="dash"), annotation_text="VaR 95%")
    fig.update_layout(
        title="Distribution of Final 1Y Portfolio Returns",
        xaxis_title="Return",
        yaxis_title="Frequency",
        bargap=0.02,  # Gap between bars
    )
    fig.show()


def main() -> None:
    """
    Main execution function that orchestrates the Monte Carlo portfolio risk analysis.
    
    Workflow:
    1. Download historical data for portfolio assets
    2. Calculate log returns and portfolio statistics
    3. Simulate future price paths using GBM
    4. Compute portfolio-level risk metrics (VaR, CVaR)
    5. Generate visualizations
    """
    # Portfolio configuration
    tickers = ["SPY", "BTC-USD", "GLD"]  # S&P 500, Bitcoin, Gold ETF
    weights = np.array([1 / 3, 1 / 3, 1 / 3])  # Equal weights (can be modified)
    horizon_days = 252  # 1 year of trading days
    n_sims = 10_000  # Number of Monte Carlo simulations

    # Step 1: Download and process historical data
    fetcher = DataFetcher(tickers)
    prices = fetcher.fetch_prices()
    log_returns = fetcher.compute_log_returns(prices)

    # Step 2: Calculate portfolio statistics
    optimizer = PortfolioOptimizer(log_returns)
    start_prices = prices.iloc[-1].to_numpy()  # Use most recent prices as starting point

    # Step 3: Run Monte Carlo simulation
    simulator = MonteCarloSimulator(
        mean_returns=optimizer.mean_returns,
        cov_matrix=optimizer.cov_matrix,
        start_prices=start_prices,
        steps=horizon_days,
        n_sims=n_sims,
    )
    price_paths = simulator.simulate_paths(seed=42)  # Fixed seed for reproducibility

    # Step 4: Calculate portfolio-level risk metrics
    risk = RiskMetrics(weights)
    portfolio_growth = risk.portfolio_growth(price_paths)
    final_returns = portfolio_growth[:, -1] - 1  # Final returns (portfolio value - 1)
    var95, cvar95 = risk.var_cvar(final_returns, alpha=0.95)
    mean_ret = final_returns.mean()

    # Step 5: Print summary statistics
    print("----- Portfolio Risk Summary -----")
    print(f"Tickers: {tickers}")
    print(f"Weights: {weights.round(3)}")
    print(f"Mean final return: {mean_ret:.4f}")
    print(f"VaR 95% (loss threshold): {var95:.4f}")
    print(f"CVaR 95% (expected loss beyond VaR): {cvar95:.4f}")

    # Step 6: Generate visualizations
    plot_correlation_heatmap(optimizer.corr_matrix)
    plot_spaghetti(portfolio_growth)
    plot_final_distribution(final_returns, mean_ret=mean_ret, var95=var95)


if __name__ == "__main__":
    main()

