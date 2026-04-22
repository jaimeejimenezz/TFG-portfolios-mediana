import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/returns.csv"

LOOKBACK_DAYS = 252      # no se usa en 1/N, pero lo dejamos para consistencia futura
REBALANCE_DAYS = 21      # ~1 mes de trading

def load_returns(path: str) -> pd.DataFrame:
    r = pd.read_csv(path, index_col=0, parse_dates=True)
    r = r.sort_index()
    return r

def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets

def run_backtest_equal_weight(returns: pd.DataFrame, rebalance_days: int = 21) -> pd.DataFrame:
    tickers = list(returns.columns)
    n = len(tickers)

    w = equal_weight(n)  # pesos iniciales
    port_rets = []
    dates = []

    for i, dt in enumerate(returns.index):
        # rebalanceo cada X días (al inicio también)
        if i % rebalance_days == 0:
            w = equal_weight(n)

        r_t = returns.iloc[i].values  # rendimientos del día
        rp_t = float(np.dot(w, r_t))  # retorno de cartera ese día

        dates.append(dt)
        port_rets.append(rp_t)

    out = pd.DataFrame({"portfolio_return": port_rets}, index=pd.DatetimeIndex(dates))
    out["equity"] = (1 + out["portfolio_return"]).cumprod()
    return out

if __name__ == "__main__":
    rets = load_returns(RETURNS_PATH)
    bt = run_backtest_equal_weight(rets, REBALANCE_DAYS)

    bt.to_csv("data/processed/backtest_equal_weight.csv")
    print("OK - Backtest 1/N generado")
    print(bt.head(3))
    print(bt.tail(3))
    print("Equity final:", bt["equity"].iloc[-1])