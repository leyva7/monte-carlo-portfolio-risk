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
    """Download prices and compute log-returns."""

    def __init__(self, tickers: TickerList, period_years: int = 5) -> None:
        self.tickers = tickers
        self.period_years = period_years

    def fetch_prices(self) -> pd.DataFrame:
        prices = yf.download(
            tickers=self.tickers,
            period=f"{self.period_years}y",
            auto_adjust=False,  # keep Adj Close available
            progress=False,
        )
        if prices.empty:
            raise ValueError("No price data downloaded; check tickers or connectivity.")

        # yfinance returns MultiIndex (field, ticker) when multiple tickers.
        if isinstance(prices.columns, pd.MultiIndex):
            if ("Adj Close" in prices.columns.get_level_values(0)):
                prices = prices["Adj Close"]
            elif ("Close" in prices.columns.get_level_values(0)):
                prices = prices["Close"]
            else:
                raise KeyError("Expected 'Adj Close' or 'Close' columns in download.")
        else:
            if "Adj Close" in prices.columns:
                prices = prices.rename(columns={"Adj Close": self.tickers[0]})
            elif "Close" in prices.columns:
                prices = prices.rename(columns={"Close": self.tickers[0]})
            else:
                raise KeyError("Expected 'Adj Close' or 'Close' columns in download.")

        return prices.dropna()

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return np.log(prices / prices.shift(1)).dropna()


class PortfolioOptimizer:
    """Estimate portfolio statistics from historical log-returns."""

    def __init__(self, log_returns: pd.DataFrame) -> None:
        self.log_returns = log_returns
        self.mean_returns = log_returns.mean().to_numpy()  # daily drift estimate
        self.cov_matrix = log_returns.cov().to_numpy()  # daily covariance
        self.corr_matrix = log_returns.corr()


class MonteCarloSimulator:
    """
    Vectorized GBM simulator with correlated shocks.

    We discretize GBM with daily steps:
        log(S_t) = log(S_{t-1}) + (mu - 0.5*var)*dt + L @ z
    where L is the Cholesky factor of the covariance matrix and z ~ N(0, I).
    Daily covariance from historical data is used (dt = 1 day), so no loops
    over assets or simulations are needed.
    """

    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        start_prices: np.ndarray,
        steps: int = 252,
        n_sims: int = 10_000,
    ) -> None:
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.start_prices = start_prices
        self.steps = steps
        self.n_sims = n_sims
        self.n_assets = start_prices.shape[0]
        self.cholesky = np.linalg.cholesky(cov_matrix)

    def simulate_paths(self, seed: int | None = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # Uncorrelated draws: (sims, days, assets)
        shocks = rng.standard_normal((self.n_sims, self.steps, self.n_assets))
        # Correlate shocks via Cholesky; matrix multiply is fully vectorized.
        correlated_shocks = shocks @ self.cholesky.T

        drift = self.mean_returns - 0.5 * np.diag(self.cov_matrix)
        drift = drift.reshape(1, 1, self.n_assets)

        # Log-returns per step; cumulative sum builds the GBM exponent.
        log_returns = drift + correlated_shocks
        log_paths = np.cumsum(log_returns, axis=1)

        prices = np.empty((self.n_sims, self.steps + 1, self.n_assets))
        prices[:, 0, :] = self.start_prices
        prices[:, 1:, :] = np.exp(np.log(self.start_prices) + log_paths)
        return prices


class RiskMetrics:
    """Compute portfolio-level VaR and CVaR from simulated price paths."""

    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def portfolio_growth(self, price_paths: np.ndarray) -> np.ndarray:
        asset_returns = price_paths[:, 1:, :] / price_paths[:, :-1, :] - 1
        # Weighted daily returns: vectorized dot over the asset axis.
        portfolio_returns = np.tensordot(
            asset_returns, self.weights, axes=([2], [0])
        )
        # Cumulative product of (1 + r_t) gives portfolio value path.
        return np.cumprod(1 + portfolio_returns, axis=1)

    def var_cvar(self, final_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
        cutoff = 100 * (1 - alpha)  # e.g., 5th percentile for 95% VaR
        var = np.percentile(final_returns, cutoff)
        cvar = final_returns[final_returns <= var].mean()
        return var, cvar


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        origin="lower",
        title="Correlation Heatmap",
    )
    fig.update_layout(width=600, height=500)
    fig.show()


def plot_spaghetti(portfolio_growth: np.ndarray, n_paths: int = 100) -> None:
    n_paths = min(n_paths, portfolio_growth.shape[0])
    idx = np.random.choice(portfolio_growth.shape[0], size=n_paths, replace=False)
    fig = go.Figure()
    for i in idx:
        fig.add_trace(
            go.Scatter(
                y=portfolio_growth[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
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
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=final_returns,
            nbinsx=60,
            name="Final Returns",
            marker=dict(color="#4C78A8"),
            opacity=0.75,
        )
    )
    fig.add_vline(x=mean_ret, line=dict(color="green", dash="dash"), annotation_text="Mean")
    fig.add_vline(x=var95, line=dict(color="red", dash="dash"), annotation_text="VaR 95%")
    fig.update_layout(
        title="Distribution of Final 1Y Portfolio Returns",
        xaxis_title="Return",
        yaxis_title="Frequency",
        bargap=0.02,
    )
    fig.show()


def main() -> None:
    tickers = ["SPY", "BTC-USD", "GLD"]
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    horizon_days = 252
    n_sims = 10_000

    fetcher = DataFetcher(tickers)
    prices = fetcher.fetch_prices()
    log_returns = fetcher.compute_log_returns(prices)

    optimizer = PortfolioOptimizer(log_returns)
    start_prices = prices.iloc[-1].to_numpy()

    simulator = MonteCarloSimulator(
        mean_returns=optimizer.mean_returns,
        cov_matrix=optimizer.cov_matrix,
        start_prices=start_prices,
        steps=horizon_days,
        n_sims=n_sims,
    )
    price_paths = simulator.simulate_paths(seed=42)

    risk = RiskMetrics(weights)
    portfolio_growth = risk.portfolio_growth(price_paths)
    final_returns = portfolio_growth[:, -1] - 1
    var95, cvar95 = risk.var_cvar(final_returns, alpha=0.95)
    mean_ret = final_returns.mean()

    print("----- Portfolio Risk Summary -----")
    print(f"Tickers: {tickers}")
    print(f"Weights: {weights.round(3)}")
    print(f"Mean final return: {mean_ret:.4f}")
    print(f"VaR 95% (loss threshold): {var95:.4f}")
    print(f"CVaR 95% (expected loss beyond VaR): {cvar95:.4f}")

    # Visualizations
    plot_correlation_heatmap(optimizer.corr_matrix)
    plot_spaghetti(portfolio_growth)
    plot_final_distribution(final_returns, mean_ret=mean_ret, var95=var95)


if __name__ == "__main__":
    main()

