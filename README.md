# Monte Carlo Portfolio Risk

Simulador en Python para analizar riesgo de una cartera (SPY, BTC-USD, GLD) usando un enfoque de Monte Carlo con Movimiento Browniano Geométrico y shocks correlacionados (Cholesky). El código está vectorizado con NumPy para manejar 10,000 simulaciones sin bucles lentos.

## Requisitos
- Python 3.10+
- `numpy`, `pandas`, `plotly`, `yfinance`

Instalación rápida:
```
pip install numpy pandas plotly yfinance
```

## Uso
```
python portfolio_simulator.py
```

Lo que hace el script:
- Descarga 5 años de precios ajustados.
- Calcula retornos logarítmicos, medias, covarianzas y matriz de correlación.
- Simula 10,000 trayectorias a 1 año (252 días) con GBM correlacionado.
- Calcula VaR 95% y CVaR 95% del retorno final de la cartera.
- Muestra:
  - Mapa de calor de correlación.
  - Spaghetti plot de 100 trayectorias del portafolio.
  - Histograma de retornos finales con media y VaR marcados.

Los pesos por defecto son iguales entre los tres activos; puedes modificarlos en `portfolio_simulator.py`.