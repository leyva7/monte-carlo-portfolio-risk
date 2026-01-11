"""
Monte Carlo Portfolio Risk Simulator
====================================
A professional-grade portfolio risk analysis tool using Monte Carlo simulation
with Geometric Brownian Motion (GBM) and correlated asset movements.

Author: Portfolio Project
Purpose: Quantitative Finance & Risk Analysis
"""

from __future__ import annotations
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Professional color palette for LinkedIn-ready visualizations
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',     # Orange
    'success': '#2ca02c',       # Green
    'warning': '#d62728',       # Red
    'info': '#9467bd',          # Purple
    'light': '#8c564b',         # Brown
    'dark': '#2C3E50',          # Dark blue-gray
    'accent': '#17a2b8',        # Teal
}

# Professional color scheme for portfolios
PORTFOLIO_COLORS = ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6']

TickerList = list[str]

# -------------------------
# Helper: setup result folders
# -------------------------
def clean_results_dir(base_dir: str = "results") -> None:
    """Remove all contents from results directory before new run."""
    results_path = Path(base_dir)
    if results_path.exists() and results_path.is_dir():
        try:
            shutil.rmtree(results_path)
            print(f"[*] Cleaned existing results directory: {base_dir}")
        except Exception as e:
            print(f"[WARNING] Could not clean results directory: {e}")

def setup_results_dirs(base_dir: str = "results") -> dict[str, Path]:
    """Create organized directory structure for results."""
    dirs = {
        "base": Path(base_dir),
        "paths": Path(base_dir) / "paths",
        "distribution": Path(base_dir) / "distribution",
        "heatmaps": Path(base_dir) / "heatmaps",
        "comparison": Path(base_dir) / "comparison",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

# -------------------------
# DataFetcher
# -------------------------
class DataFetcher:
    """Handles downloading and preprocessing of financial data."""
    
    def __init__(self, tickers: TickerList, period_years: int = 8) -> None:
        """
        Initialize data fetcher.
        
        Note: Using 8 years to ensure all assets (including BTC-USD) have
        complete overlapping data. BTC data on Yahoo Finance starts ~2014.
        """
        self.tickers = tickers
        self.period_years = period_years

    def fetch_prices(self, retries: int = 3) -> pd.DataFrame:
        """Download historical price data with retry logic."""
        last_error = None
        for attempt in range(retries):
            try:
                prices = yf.download(
                    tickers=self.tickers,
                    period=f"{self.period_years}y",
                    auto_adjust=False,
                    progress=False,
                )
                if prices is not None and not prices.empty:
                    break
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    continue
                else:
                    raise ValueError(f"Failed download: {e}") from e

        if isinstance(prices.columns, pd.MultiIndex):
            level0_values = prices.columns.get_level_values(0)
            if "Adj Close" in level0_values:
                prices = prices["Adj Close"]
            elif "Close" in level0_values:
                prices = prices["Close"]
            else:
                raise KeyError(f"Columns not found: {list(set(level0_values))}")
        else:
            if "Adj Close" in prices.columns:
                prices = prices.rename(columns={"Adj Close": self.tickers[0]})
            elif "Close" in prices.columns:
                prices = prices.rename(columns={"Close": self.tickers[0]})
            else:
                raise KeyError(f"Columns not found: {list(prices.columns)}")

        prices_clean = prices.dropna()
        if prices_clean.empty:
            raise ValueError(f"No data available after cleaning for {self.tickers}")
        return prices_clean

    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate logarithmic returns for GBM modeling."""
        return np.log(prices / prices.shift(1)).dropna()

# -------------------------
# PortfolioOptimizer
# -------------------------
class PortfolioOptimizer:
    """Calculates key portfolio statistics from historical returns."""
    
    def __init__(self, log_returns: pd.DataFrame) -> None:
        self.log_returns = log_returns
        self.mean_returns = log_returns.mean().to_numpy()
        self.cov_matrix = log_returns.cov().to_numpy()
        self.corr_matrix = log_returns.corr()

# -------------------------
# MonteCarloSimulator
# -------------------------
class MonteCarloSimulator:
    """
    Implements vectorized Monte Carlo simulation using Geometric Brownian Motion.
    Uses Cholesky decomposition for efficient correlated random number generation.
    """
    
    def __init__(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                 start_prices: np.ndarray, steps: int = 252, n_sims: int = 10_000) -> None:
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.start_prices = start_prices
        self.steps = steps
        self.n_sims = n_sims
        self.n_assets = start_prices.shape[0]
        # Cholesky decomposition enables efficient generation of correlated shocks
        self.cholesky = np.linalg.cholesky(cov_matrix)

    def simulate_paths(self, seed: int | None = 42) -> np.ndarray:
        """
        Simulate asset price paths using GBM with correlated shocks.
        
        GBM Formula: dS = μS dt + σS dW
        Where:
        - μ = drift (mean return)
        - σ = volatility (from covariance matrix)
        - dW = correlated Wiener process (via Cholesky decomposition)
        
        Returns:
            Array of shape (n_sims, steps+1, n_assets) with price paths
        """
        rng = np.random.default_rng(seed)
        # Generate uncorrelated standard normal shocks
        shocks = rng.standard_normal((self.n_sims, self.steps, self.n_assets))
        # Transform to correlated shocks using Cholesky decomposition
        correlated_shocks = shocks @ self.cholesky.T
        # Calculate drift term: μ - 0.5*σ² (Itô's lemma correction)
        drift = self.mean_returns - 0.5 * np.diag(self.cov_matrix)
        drift = drift.reshape(1, 1, self.n_assets)
        # Calculate log returns: drift + correlated shocks
        log_returns = drift + correlated_shocks
        # Cumulative sum to get log price paths
        log_paths = np.cumsum(log_returns, axis=1)
        # Convert to price paths
        prices = np.empty((self.n_sims, self.steps + 1, self.n_assets))
        prices[:, 0, :] = self.start_prices
        prices[:, 1:, :] = np.exp(np.log(self.start_prices) + log_paths)
        return prices

# -------------------------
# RiskMetrics
# -------------------------
class RiskMetrics:
    """Calculates comprehensive risk and return metrics for portfolios."""
    
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    def portfolio_growth(self, price_paths: np.ndarray) -> np.ndarray:
        """Calculate portfolio value over time for each simulation."""
        asset_returns = price_paths[:, 1:, :] / price_paths[:, :-1, :] - 1
        portfolio_returns = np.tensordot(asset_returns, self.weights, axes=([2], [0]))
        return np.cumprod(1 + portfolio_returns, axis=1)

    def var_cvar(self, final_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)."""
        cutoff = 100 * (1 - alpha)
        var = np.percentile(final_returns, cutoff)
        cvar = final_returns[final_returns <= var].mean()
        return var, cvar

    def sharpe_ratio(self, portfolio_growth: np.ndarray, risk_free_rate: float = 0.0, horizon_years: float = 1.0) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Formula: Sharpe = (R_p - R_f) / σ_p
        Where R_p and σ_p are annualized portfolio return and volatility.
        
        Args:
            portfolio_growth: Portfolio value paths (n_sims, n_days)
            risk_free_rate: Annual risk-free rate
            horizon_years: Investment horizon in years (for proper annualization)
        """
        final_returns = portfolio_growth[:, -1] - 1  # Cumulative returns
        
        # Convert cumulative returns to annualized
        # (1 + total_return)^(1/years) - 1
        annualized_returns = np.power(1 + final_returns, 1.0 / horizon_years) - 1
        mean_annual_return = annualized_returns.mean()
        
        # Annualized volatility: std of annualized returns
        std_annual_return = np.std(annualized_returns)
        
        if std_annual_return == 0:
            return 0.0
        return (mean_annual_return - risk_free_rate) / std_annual_return

    def max_drawdown(self, portfolio_growth: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate maximum drawdown for each simulation path."""
        n_sims, n_days = portfolio_growth.shape
        max_drawdowns = np.zeros(n_sims)
        for i in range(n_sims):
            path = portfolio_growth[i]
            running_max = np.maximum.accumulate(path)
            drawdowns = (running_max - path) / running_max
            max_drawdowns[i] = drawdowns.max()
        return max_drawdowns.mean(), max_drawdowns

    def compute_all_metrics(self, portfolio_growth: np.ndarray, risk_free_rate: float = 0.0, horizon_years: float = 1.0) -> dict[str, float]:
        """
        Compute all risk and return metrics.
        
        Returns include both cumulative (total) and annualized metrics for clarity.
        """
        final_returns = portfolio_growth[:, -1] - 1  # Cumulative returns
        var95, cvar95 = self.var_cvar(final_returns)
        
        # Cumulative metrics
        mean_ret_cumulative = final_returns.mean()
        std_ret_cumulative = final_returns.std()
        
        # Annualized metrics
        annualized_returns = np.power(1 + final_returns, 1.0 / horizon_years) - 1
        mean_ret_annualized = annualized_returns.mean()
        std_ret_annualized = np.std(annualized_returns)
        
        sharpe = self.sharpe_ratio(portfolio_growth, risk_free_rate, horizon_years)
        mean_dd, max_dds = self.max_drawdown(portfolio_growth)
        
        return {
            "mean_return_cumulative": mean_ret_cumulative,
            "mean_return_annualized": mean_ret_annualized,
            "std_return_cumulative": std_ret_cumulative,
            "std_return_annualized": std_ret_annualized,
            "var_95": var95,
            "cvar_95": cvar95,
            "sharpe_ratio": sharpe,
            "mean_max_drawdown": mean_dd,
            "max_drawdown_95": np.percentile(max_dds, 95),
        }

# -------------------------
# Professional Plotting Functions
# -------------------------
def get_professional_layout(title: str, xaxis_title: str = "", yaxis_title: str = "") -> dict:
    """Get professional layout template for LinkedIn-ready plots."""
    return {
        'title': {
            'text': title,
            'font': {'size': 24, 'family': 'Arial, sans-serif', 'color': COLORS['dark']},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'title': {'text': xaxis_title, 'font': {'size': 16, 'family': 'Arial, sans-serif'}},
            'gridcolor': '#e0e0e0',
            'showgrid': True,
            'linecolor': '#b0b0b0',
            'linewidth': 1
        },
        'yaxis': {
            'title': {'text': yaxis_title, 'font': {'size': 16, 'family': 'Arial, sans-serif'}},
            'gridcolor': '#e0e0e0',
            'showgrid': True,
            'linecolor': '#b0b0b0',
            'linewidth': 1
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80},
    }

def save_figure(fig: go.Figure, path: Path, width: int = 1600, height: int = 900) -> None:
    """Save figure as high-resolution PNG for LinkedIn posts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), width=width, height=height, scale=3)
        print(f"[OK] Saved: {path}")
    except Exception as e:
        print(f"[WARNING] Could not save figure {path}: {e}")
        print("   Make sure kaleido is installed: pip install kaleido")

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, tickers: list[str], save_path: Path) -> go.Figure:
    """Create professional correlation heatmap."""
    # Format ticker names for display
    ticker_labels = [t.replace('-USD', '') if '-USD' in t else t for t in tickers]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=ticker_labels,
        y=ticker_labels,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={'size': 14, 'color': 'white'},
        colorbar=dict(
            title=dict(text="Correlation", font={'size': 14}),
            tickfont={'size': 12}
        )
    ))
    
    layout = get_professional_layout(
        "Asset Correlation Matrix",
        "Assets",
        "Assets"
    )
    layout.update({'width': 900, 'height': 800})
    fig.update_layout(**layout)
    
    save_figure(fig, save_path, width=900, height=800)
    return fig

def plot_portfolio_paths_with_percentiles(
    portfolio_growth: np.ndarray,
    portfolio_name: str,
    weights: np.ndarray,
    tickers: list[str],
    save_path: Path
) -> go.Figure:
    """Create professional portfolio path visualization with percentiles and shading."""
    n_days = portfolio_growth.shape[1]
    days = np.arange(n_days)
    years = days / 252  # Convert trading days to years
    
    # Calculate percentiles
    p5 = np.percentile(portfolio_growth, 5, axis=0)
    p25 = np.percentile(portfolio_growth, 25, axis=0)
    p50 = np.percentile(portfolio_growth, 50, axis=0)
    p75 = np.percentile(portfolio_growth, 75, axis=0)
    p95 = np.percentile(portfolio_growth, 95, axis=0)
    
    fig = go.Figure()
    
    # Add shaded regions for confidence intervals
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p5, p95[::-1]]),
        fill='toself',
        fillcolor='rgba(44, 62, 80, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='5th-95th Percentile',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p25, p75[::-1]]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='25th-75th Percentile',
        showlegend=True
    ))
    
    # Add percentile lines
    fig.add_trace(go.Scatter(
        x=days, y=p50,
        mode='lines',
        name='Median (50th)',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    # Add sample paths (100 random simulations)
    n_sample = min(100, portfolio_growth.shape[0])
    sample_indices = np.random.choice(portfolio_growth.shape[0], n_sample, replace=False)
    for idx in sample_indices[:20]:  # Show 20 sample paths
        fig.add_trace(go.Scatter(
            x=days, y=portfolio_growth[idx],
            mode='lines',
            line=dict(color='rgba(200,200,200,0.3)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Format portfolio weights for subtitle
    ticker_labels = [t.replace('-USD', '') if '-USD' in t else t for t in tickers]
    weight_str = " | ".join([f"{t}: {w*100:.1f}%" for t, w in zip(ticker_labels, weights)])
    horizon_years = n_days / 252
    
    # Create custom x-axis labels showing years
    tick_vals = [i * 252 for i in range(int(horizon_years) + 1)]
    tick_text = [f"{i}Y" for i in range(int(horizon_years) + 1)]
    
    layout = get_professional_layout(
        f"{portfolio_name} Portfolio Simulation - {horizon_years:.0f} Year Forecast<br><sub>{weight_str} | 10,000 Monte Carlo Simulations</sub>",
        "Time Horizon (Years)",
        "Portfolio Value (Normalized)"
    )
    layout.update({
        'width': 1600,
        'height': 900,
        'legend': {'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(255,255,255,0.8)'},
        'xaxis': {
            **layout['xaxis'],
            'tickmode': 'array',
            'tickvals': tick_vals,
            'ticktext': tick_text
        }
    })
    fig.update_layout(**layout)
    
    save_figure(fig, save_path)
    return fig

def plot_final_distribution(
    final_returns: np.ndarray,
    portfolio_name: str,
    metrics: dict[str, float],
    save_path: Path
) -> go.Figure:
    """Create professional return distribution histogram with key metrics."""
    # Use cumulative return for distribution (since final_returns are cumulative)
    mean_ret = metrics['mean_return_cumulative']
    mean_ret_annual = metrics['mean_return_annualized']
    var95 = metrics['var_95']
    cvar95 = metrics['cvar_95']
    
    # Calculate percentiles
    percentiles = {
        'p5': np.percentile(final_returns, 5),
        'p25': np.percentile(final_returns, 25),
        'p50': np.percentile(final_returns, 50),
        'p75': np.percentile(final_returns, 75),
        'p95': np.percentile(final_returns, 95),
    }
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=final_returns,
        nbinsx=80,
        name='Distribution',
        marker_color=COLORS['primary'],
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Add vertical lines for key metrics
    fig.add_vline(
        x=mean_ret,
        line_dash="dash",
        line_color=COLORS['success'],
        line_width=3,
        annotation_text=f"Mean: {mean_ret:.2%}",
        annotation_position="top",
        annotation_font_size=14
    )
    
    fig.add_vline(
        x=var95,
        line_dash="dash",
        line_color=COLORS['warning'],
        line_width=3,
        annotation_text=f"VaR 95%: {var95:.2%}",
        annotation_position="top",
        annotation_font_size=14
    )
    
    fig.add_vline(
        x=percentiles['p50'],
        line_dash="dot",
        line_color=COLORS['dark'],
        line_width=2,
        annotation_text=f"Median: {percentiles['p50']:.2%}",
        annotation_position="bottom",
        annotation_font_size=12
    )
    
    # Add text box with key metrics
    metrics_text = (
        f"<b>Key Metrics:</b><br>"
        f"Cumulative Return: {mean_ret:.2%}<br>"
        f"Annualized Return: {mean_ret_annual:.2%}<br>"
        f"Std Dev (Cumulative): {metrics['std_return_cumulative']:.2%}<br>"
        f"VaR 95%: {var95:.2%}<br>"
        f"CVaR 95%: {cvar95:.2%}<br>"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br>"
        f"Max Drawdown: {metrics['mean_max_drawdown']:.2%}"
    )
    
    # Estimate horizon from number of simulations (assuming 252 days per year)
    # We'll use 3 years as default since that's our current setting
    horizon_years = 3
    layout = get_professional_layout(
        f"{portfolio_name} - Final Return Distribution ({horizon_years:.0f} Year Forecast)<br><sub>10,000 Monte Carlo Simulations</sub>",
        "Portfolio Return",
        "Probability Density"
    )
    layout.update({
        'width': 1600,
        'height': 900,
        'annotations': [{
            'text': metrics_text,
            'x': 0.98,
            'y': 0.98,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'align': 'right',
            'bgcolor': 'rgba(255,255,255,0.9)',
            'bordercolor': COLORS['dark'],
            'borderwidth': 2,
            'font': {'size': 12}
        }]
    })
    fig.update_layout(**layout)
    
    save_figure(fig, save_path)
    return fig

def plot_portfolio_comparison(results: dict[str, dict], save_path: Path) -> go.Figure:
    """Create professional comparison of median portfolio paths."""
    fig = go.Figure()
    
    for idx, (name, data) in enumerate(results.items()):
        portfolio_growth = data["portfolio_growth"]
        median_path = np.percentile(portfolio_growth, 50, axis=0)
        days = np.arange(portfolio_growth.shape[1])
        
        fig.add_trace(go.Scatter(
            x=days,
            y=median_path,
            mode="lines",
            name=name,
            line=dict(
                color=PORTFOLIO_COLORS[idx % len(PORTFOLIO_COLORS)],
                width=3
            ),
            hovertemplate=f"<b>{name}</b><br>Day: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
        ))
    
    n_days = portfolio_growth.shape[1]
    horizon_years = n_days / 252
    tick_vals = [i * 252 for i in range(int(horizon_years) + 1)]
    tick_text = [f"{i}Y" for i in range(int(horizon_years) + 1)]
    
    layout = get_professional_layout(
        f"Portfolio Comparison - {horizon_years:.0f} Year Forecast<br><sub>10,000 Monte Carlo Simulations per Portfolio</sub>",
        "Time Horizon (Years)",
        "Portfolio Value (Normalized)"
    )
    layout.update({
        'width': 1600,
        'height': 900,
        'legend': {'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(255,255,255,0.8)'},
        'xaxis': {
            **layout['xaxis'],
            'tickmode': 'array',
            'tickvals': tick_vals,
            'ticktext': tick_text
        }
    })
    fig.update_layout(**layout)
    
    save_figure(fig, save_path)
    return fig

def plot_metrics_comparison(results: dict[str, dict], save_path: Path) -> go.Figure:
    """Create professional comparison of key metrics across portfolios."""
    metrics_df = pd.DataFrame({name: data["metrics"] for name, data in results.items()}).T
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Return",
            "Sharpe Ratio",
            "Value at Risk (95%)",
            "Mean Maximum Drawdown"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Annualized Return (more professional for comparison)
    fig.add_trace(
        go.Bar(
            x=metrics_df.index,
            y=metrics_df["mean_return_annualized"],
            name="Annualized Return",
            marker_color=COLORS['primary'],
            text=[f"{v:.2%}" for v in metrics_df["mean_return_annualized"]],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Bar(
            x=metrics_df.index,
            y=metrics_df["sharpe_ratio"],
            name="Sharpe Ratio",
            marker_color=COLORS['success'],
            text=[f"{v:.2f}" for v in metrics_df["sharpe_ratio"]],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    # VaR 95%
    fig.add_trace(
        go.Bar(
            x=metrics_df.index,
            y=metrics_df["var_95"],
            name="VaR 95%",
            marker_color=COLORS['warning'],
            text=[f"{v:.2%}" for v in metrics_df["var_95"]],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # Mean Max Drawdown
    fig.add_trace(
        go.Bar(
            x=metrics_df.index,
            y=metrics_df["mean_max_drawdown"],
            name="Mean Max Drawdown",
            marker_color=COLORS['info'],
            text=[f"{v:.2%}" for v in metrics_df["mean_max_drawdown"]],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Annualized Return", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=2)
    
    layout = get_professional_layout(
        "Portfolio Metrics Comparison<br><sub>Risk-Return Analysis Across Strategies</sub>",
        "",
        ""
    )
    layout.update({
        'width': 1600,
        'height': 1000,
        'showlegend': False
    })
    fig.update_layout(**layout)
    
    save_figure(fig, save_path, width=1600, height=1000)
    return fig

# -------------------------
# Executive Summary
# -------------------------
def print_executive_summary(results: dict[str, dict], tickers: list[str]) -> None:
    """Print professional executive summary of results."""
    print("\n" + "="*80)
    print(" " * 25 + "EXECUTIVE SUMMARY")
    print("="*80)
    print("\n[*] PORTFOLIO RISK ANALYSIS - MONTE CARLO SIMULATION")
    print(f"Assets Analyzed: {', '.join(tickers)}")
    print(f"Simulations per Portfolio: 10,000")
    horizon_days = results[list(results.keys())[0]]['portfolio_growth'].shape[1]
    horizon_years = horizon_days / 252
    print(f"Time Horizon: {horizon_days} trading days ({horizon_years:.1f} years)")
    
    print("\n" + "-"*80)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("-"*80)
    
    # Create comparison table with both cumulative and annualized returns
    comparison_data = []
    for name, data in results.items():
        m = data['metrics']
        comparison_data.append({
            'Portfolio': name,
            'Cumulative Return': f"{m['mean_return_cumulative']:.2%}",
            'Annualized Return': f"{m['mean_return_annualized']:.2%}",
            'Sharpe Ratio': f"{m['sharpe_ratio']:.2f}",
            'VaR 95%': f"{m['var_95']:.2%}",
            'Max DD': f"{m['mean_max_drawdown']:.2%}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print(f"\nNote: Cumulative Return = Total return over {horizon_years:.1f} years")
    print(f"      Annualized Return = Average annual return (geometric mean)")
    
    # Find best performers
    print("\n" + "-"*80)
    print("KEY INSIGHTS")
    print("-"*80)
    
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    best_return_annual = max(results.items(), key=lambda x: x[1]['metrics']['mean_return_annualized'])
    best_return_cumulative = max(results.items(), key=lambda x: x[1]['metrics']['mean_return_cumulative'])
    lowest_var = min(results.items(), key=lambda x: x[1]['metrics']['var_95'])
    lowest_dd = min(results.items(), key=lambda x: x[1]['metrics']['mean_max_drawdown'])
    
    print(f"[*] Best Risk-Adjusted Return (Sharpe): {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.2f})")
    print(f"[*] Highest Annualized Return: {best_return_annual[0]} ({best_return_annual[1]['metrics']['mean_return_annualized']:.2%})")
    print(f"[*] Highest Cumulative Return: {best_return_cumulative[0]} ({best_return_cumulative[1]['metrics']['mean_return_cumulative']:.2%})")
    print(f"[*] Lowest Risk (VaR 95%): {lowest_var[0]} ({lowest_var[1]['metrics']['var_95']:.2%})")
    print(f"[*] Most Stable (Lowest Drawdown): {lowest_dd[0]} ({lowest_dd[1]['metrics']['mean_max_drawdown']:.2%})")
    
    print("\n" + "="*80 + "\n")

# -------------------------
# Compare portfolios function
# -------------------------
def compare_portfolios(
    tickers: list[str],
    portfolio_configs: dict[str, np.ndarray],
    horizon_days: int = 252,
    n_sims: int = 10_000,
    risk_free_rate: float = 0.0,
    seed: int = 42
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Run Monte Carlo simulations for multiple portfolio configurations."""
    print("[*] Downloading historical data...")
    fetcher = DataFetcher(tickers)
    prices = fetcher.fetch_prices()
    log_returns = fetcher.compute_log_returns(prices)
    optimizer = PortfolioOptimizer(log_returns)
    start_prices = prices.iloc[-1].to_numpy()
    
    print(f"[OK] Data downloaded: {len(prices)} days of history")
    print(f"[OK] Running {n_sims:,} simulations for {len(portfolio_configs)} portfolios...")
    
    results = {}
    for name, weights in portfolio_configs.items():
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        sim = MonteCarloSimulator(
            optimizer.mean_returns,
            optimizer.cov_matrix,
            start_prices,
            steps=horizon_days,
            n_sims=n_sims
        )
        paths = sim.simulate_paths(seed=seed)
        risk = RiskMetrics(weights)
        growth = risk.portfolio_growth(paths)
        horizon_years = horizon_days / 252
        metrics = risk.compute_all_metrics(growth, risk_free_rate, horizon_years)
        results[name] = {
            "weights": weights,
            "metrics": metrics,
            "portfolio_growth": growth,
            "final_returns": growth[:, -1] - 1
        }
        print(f"  [OK] Completed: {name}")
    
    return results, optimizer.corr_matrix

# -------------------------
# Main
# -------------------------
def main() -> None:
    """Main execution function."""
    print("\n" + "="*80)
    print(" " * 20 + "MONTE CARLO PORTFOLIO RISK SIMULATOR")
    print("="*80 + "\n")
    
    # Configuration
    tickers = ["SPY", "BTC-USD", "GLD", "BND"]

    portfolio_configs = {
        # Distribution: [SPY, BTC-USD, GLD, BND]
        
        # CONSERVATIVE: High bond allocation (60%) and minimal crypto (2%) 
        # to lower the portfolio's overall standard deviation.
        "Conservative": np.array([0.28, 0.02, 0.10, 0.60]),
        
        # AGGRESSIVE: Focus on growth (SPY) and high-volatility assets (BTC)
        "Aggressive": np.array([0.50, 0.20, 0.20, 0.10]),
        
        # GOLD HEAVY: Focused on inflation protection and hedging
        "Gold Heavy": np.array([0.20, 0.10, 0.50, 0.20]),
    }
    horizon_days = 3650  # 10 years (3650 trading days per year)
    n_sims = 10_000
    risk_free_rate = 0.02  # 2% annual risk-free rate
    
    # Clean previous results
    clean_results_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = setup_results_dirs()
    
    # Run simulations
    results, corr_matrix = compare_portfolios(
        tickers,
        portfolio_configs,
        horizon_days,
        n_sims,
        risk_free_rate
    )
    
    # Print executive summary
    print_executive_summary(results, tickers)
    
    # Generate visualizations
    print("[*] Generating professional visualizations...\n")
    
    # 1. Correlation heatmap
    print("  -> Correlation Heatmap...")
    heatmap_path = dirs["heatmaps"] / f"correlation_matrix_{timestamp}.png"
    plot_correlation_heatmap(corr_matrix, tickers, heatmap_path)
    
    # 2. Individual portfolio paths with percentiles
    print("  -> Portfolio Path Simulations...")
    for name, data in results.items():
        path_png = dirs["paths"] / f"{name.replace(' ', '_')}_{timestamp}.png"
        plot_portfolio_paths_with_percentiles(
            data["portfolio_growth"],
            name,
            data["weights"],
            tickers,
            path_png
        )
    
    # 3. Final return distributions
    print("  -> Return Distributions...")
    for name, data in results.items():
        dist_png = dirs["distribution"] / f"{name.replace(' ', '_')}_dist_{timestamp}.png"
        plot_final_distribution(
            data["final_returns"],
            name,
            data["metrics"],
            dist_png
        )
    
    # 4. Comparison plots
    print("  -> Comparison Charts...")
    comparison_paths = {
        "paths": dirs["comparison"] / f"median_paths_{timestamp}.png",
        "metrics": dirs["comparison"] / f"metrics_{timestamp}.png"
    }
    plot_portfolio_comparison(results, comparison_paths["paths"])
    plot_metrics_comparison(results, comparison_paths["metrics"])
    
    print("\n" + "="*80)
    print(f"[SUCCESS] All visualizations saved to: {dirs['base']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
