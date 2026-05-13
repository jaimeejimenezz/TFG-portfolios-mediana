import pandas as pd
import numpy as np

BT_PATH = "data/processed/backtest_equal_weight.csv"
TRADING_DAYS = 252
RF = 0.0  # tipo libre de riesgo (ponemos 0 para empezar)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax() #maximo acumulado alcanzado hasta ese momento
    dd = (equity / peak) - 1.0
    return float(dd.min()) #se queda con la peor caida de todas

if __name__ == "__main__":
    bt = pd.read_csv(BT_PATH, index_col=0, parse_dates=True)
    r = bt["portfolio_return"].astype(float) #rendimiento diario de la cartera.
    equity = bt["equity"].astype(float) #valor acumulado de la cartera.

    n_days = len(r)
    n_years = n_days / TRADING_DAYS

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0) #convierte todo a una tasa anual comparable

    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(r.mean() * TRADING_DAYS)

    sharpe = float((mean_ann - RF) / vol) if vol > 0 else np.nan #Cuanta rentabilidad obtenemos por unidad de riesgo
    mdd = max_drawdown(equity)

    print("=== 1/N (Equal Weight) ===")
    print(f"Días: {n_days} | Años aprox: {n_years:.2f}")
    print(f"Retorno total: {total_return:.3f}")
    print(f"CAGR: {cagr:.3%}")
    print(f"Volatilidad anualizada: {vol:.3%}")
    print(f"Sharpe (RF=0): {sharpe:.3f}")
    print(f"Max Drawdown: {mdd:.3%}")