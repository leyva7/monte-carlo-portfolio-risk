# 📊 Monte Carlo Portfolio Risk Simulator

A professional-grade quantitative finance tool for portfolio risk analysis using Monte Carlo simulation with Geometric Brownian Motion (GBM). This project demonstrates advanced skills in **quantitative programming**, **predictive modeling**, and **risk optimization** for FinTech applications.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)

---

## 🎯 Project Overview

This simulator analyzes portfolio risk by running **10,000 Monte Carlo simulations** to model potential future returns across multiple asset allocation strategies. The implementation uses **vectorized NumPy operations** and **Cholesky decomposition** for efficient correlated asset modeling, making it suitable for real-world quantitative finance applications.

### Key Features

- **🔬 Advanced Monte Carlo Simulation**: 10,000+ simulations using Geometric Brownian Motion (GBM)
- **⚡ Vectorized Implementation**: Fully optimized with NumPy matrix operations (no slow Python loops)
- **🔗 Correlated Asset Modeling**: Cholesky decomposition for realistic correlated price movements
- **📈 Comprehensive Risk Metrics**: VaR, CVaR, Sharpe Ratio, Maximum Drawdown
- **📊 Professional Visualizations**: LinkedIn-ready high-resolution charts with percentiles and confidence intervals
- **🎨 Multiple Portfolio Scenarios**: Compare different asset allocation strategies side-by-side
- **📋 Executive Summary**: Automated analysis and insights generation

---

## 🚀 Technical Highlights

### Mathematical Foundation

**Geometric Brownian Motion (GBM)**
```
dS = μS dt + σS dW
```
Where:
- `μ` = drift (mean return)
- `σ` = volatility (from covariance matrix)
- `dW` = correlated Wiener process (via Cholesky decomposition)

**Cholesky Decomposition**
- Enables efficient generation of correlated random shocks
- Transforms uncorrelated standard normal variables into correlated ones
- Essential for realistic multi-asset portfolio simulation

### Architecture

The project follows a **modular, object-oriented design**:

- **`DataFetcher`**: Downloads and preprocesses historical financial data
- **`PortfolioOptimizer`**: Calculates key statistics (mean returns, covariance matrix)
- **`MonteCarloSimulator`**: Implements vectorized GBM simulation
- **`RiskMetrics`**: Computes comprehensive risk and return metrics

---

## 📦 Installation

### Requirements

- Python 3.10+
- NumPy, Pandas, Plotly, yfinance, Kaleido

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd monte-carlo-portfolio-risk

# Install dependencies
pip install -r requirements.txt
```

**Note**: `kaleido` is required for exporting high-resolution PNG images. Install it separately if needed:
```bash
pip install kaleido
```

---

## 💻 Usage

Simply run the main script:

```bash
python portfolio_simulator.py
```

The script will:
1. ✅ Download 5 years of historical data for SPY, BTC-USD, and GLD
2. ✅ Run 10,000 Monte Carlo simulations for each portfolio configuration
3. ✅ Calculate comprehensive risk metrics (VaR, CVaR, Sharpe, Drawdown)
4. ✅ Generate professional visualizations
5. ✅ Print an executive summary with key insights

---

## 📊 Portfolio Scenarios

The simulator compares four distinct allocation strategies:

| Portfolio | SPY | BTC-USD | GLD | Strategy |
|-----------|-----|---------|-----|----------|
| **Equal Weight** | 33.3% | 33.3% | 33.3% | Balanced diversification |
| **Conservative** | 60% | 20% | 20% | Equity-heavy, low crypto |
| **Aggressive** | 20% | 60% | 20% | Crypto-heavy, high risk |
| **Gold Heavy** | 30% | 20% | 50% | Safe-haven focused |

You can easily modify these configurations in the `main()` function.

---

## 📈 Output & Visualizations

All results are saved to the `results/` directory with timestamped filenames:

### Generated Charts

1. **Correlation Heatmap** (`heatmaps/`)
   - Visual representation of asset correlations
   - Professional color scheme optimized for presentations

2. **Portfolio Path Simulations** (`paths/`)
   - Median path with confidence intervals (5th-95th, 25th-75th percentiles)
   - Sample simulation paths for visual context
   - Shaded regions showing uncertainty bands

3. **Return Distributions** (`distribution/`)
   - Histogram of 10,000 final returns
   - Key metrics overlaid (Mean, VaR 95%, Median)
   - Statistical summary box

4. **Comparison Charts** (`comparison/`)
   - Side-by-side median path comparison
   - Metrics bar charts (Return, Sharpe, VaR, Drawdown)

### Console Output

The script prints a comprehensive **Executive Summary** including:
- Portfolio performance comparison table
- Best performers by metric (Sharpe, Return, Risk, Stability)
- Key insights and recommendations

---

## 📊 Risk Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mean Return** | Expected portfolio return | Higher is better |
| **Standard Deviation** | Volatility of returns | Lower = more stable |
| **Sharpe Ratio** | Risk-adjusted return | >1 = good, >2 = excellent |
| **VaR 95%** | Maximum expected loss at 95% confidence | Lower (less negative) = safer |
| **CVaR 95%** | Expected loss if VaR is exceeded | Measures tail risk |
| **Maximum Drawdown** | Largest peak-to-trough decline | Lower = more stable |

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- ✅ **Quantitative Finance**: GBM modeling, risk metrics, portfolio theory
- ✅ **Advanced NumPy**: Vectorization, matrix operations, Cholesky decomposition
- ✅ **Data Science**: Financial data processing, statistical analysis
- ✅ **Software Engineering**: Modular design, error handling, code organization
- ✅ **Data Visualization**: Professional charting for business presentations
- ✅ **Predictive Modeling**: Monte Carlo simulation for scenario analysis

---

## 🔧 Technical Implementation Details

### Vectorization Strategy

All simulations run simultaneously using NumPy's tensor operations:
- **10,000 simulations × 252 days × 3 assets** = processed in a single matrix operation
- No Python loops in the simulation core
- Cholesky decomposition applied once, reused for all simulations

### Performance

- **Simulation Time**: ~2-5 seconds for 10,000 simulations
- **Memory Efficient**: Uses in-place operations where possible
- **Scalable**: Can handle 50+ assets and 100,000+ simulations

---

## 📝 Code Quality

- ✅ Fully documented with docstrings
- ✅ Type hints throughout
- ✅ Modular, reusable classes
- ✅ Error handling and validation
- ✅ Professional code structure

---

## 🎯 Use Cases

This tool is suitable for:

- **Portfolio Risk Assessment**: Evaluate different allocation strategies
- **Risk Management**: Understand potential losses and volatility
- **Investment Research**: Compare asset allocation approaches
- **Educational Purposes**: Learn Monte Carlo simulation in finance
- **Quantitative Analysis**: Foundation for more advanced models

---

## 📚 References

- **Geometric Brownian Motion**: Standard model for asset price evolution
- **Monte Carlo Methods**: Statistical simulation techniques
- **Cholesky Decomposition**: Matrix factorization for correlated random variables
- **Modern Portfolio Theory**: Risk-return optimization framework

---

## 👤 Author

**Portfolio Project** - Quantitative Finance & Data Science

Demonstrating expertise in:
- Quantitative Programming (FinTech)
- Predictive Modeling
- Risk Analysis & Optimization

---

## 📄 License

This project is open source and available for educational and portfolio purposes.

---

## 🌟 Future Enhancements

Potential improvements for production use:

- [ ] Portfolio optimization (Markowitz, Black-Litterman)
- [ ] Real-time data integration
- [ ] Interactive web dashboard
- [ ] Additional risk metrics (Sortino, Calmar ratio)
- [ ] Backtesting framework
- [ ] Multi-period rebalancing strategies

---

**Built with ❤️ for quantitative finance and data science**
