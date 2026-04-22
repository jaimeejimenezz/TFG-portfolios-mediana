import numpy as np
import pandas as pd

RETURNS_PATH = "data/processed/returns.csv"

LOOKBACK_DAYS = 252
REBALANCE_DAYS = 21

N_SAMPLES = 3000      # cuántos pesos aleatorios pruebas por rebalanceo
MAD_LIMIT = 0.0080    # umbral de riesgo (ajustable)
SEED = 42


def load_returns(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Proyección simple: recorta negativos y normaliza."""
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s == 0:
        return np.ones_like(v) / len(v)
    return v / s


def portfolio_returns(window: np.ndarray, w: np.ndarray) -> np.ndarray:
    # window: (T, N), w: (N,)
    return window @ w


def median_ret(rp: np.ndarray) -> float:
    return float(np.median(rp))


def mad_risk(rp: np.ndarray) -> float:
    m = np.median(rp)
    return float(np.mean(np.abs(rp - m)))


def sample_weights_dirichlet(n: int, rng: np.random.Generator) -> np.ndarray:
    # genera pesos positivos que suman 1 directamente
    return rng.dirichlet(alpha=np.ones(n))


def optimize_median_mad(window_returns: pd.DataFrame, n_samples: int, mad_limit: float, rng) -> np.ndarray:
    X = window_returns.dropna().values
    n = X.shape[1]

    best_w = np.ones(n) / n
    best_score = -1e18

    # fallback: si no encontramos factibles, minimizamos MAD
    best_mad = 1e18

    for _ in range(n_samples):
        w = sample_weights_dirichlet(n, rng)
        rp = portfolio_returns(X, w)

        mad = mad_risk(rp)
        med = median_ret(rp)

        if mad <= mad_limit:
            if med > best_score:
                best_score = med
                best_w = w
        else:
            # guardamos la mejor (menor) MAD por si ninguna cumple
            if mad < best_mad:
                best_mad = mad
                best_w = w

    return best_w


def run_backtest(returns: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    tickers = list(returns.columns)
    n = len(tickers)

    w = np.ones(n) / n
    port_rets = []
    dates = []
    weights_hist = []

    for i, dt in enumerate(returns.index):
        if i % REBALANCE_DAYS == 0 and i >= LOOKBACK_DAYS:
            window = returns.iloc[i - LOOKBACK_DAYS:i]
            w = optimize_median_mad(window, N_SAMPLES, MAD_LIMIT, rng)
            w = project_simplex(w)  # por seguridad
            weights_hist.append((dt, w.copy()))

        rp_t = float(np.dot(w, returns.iloc[i].values))
        dates.append(dt)
        port_rets.append(rp_t)

    bt = pd.DataFrame({"portfolio_return": port_rets}, index=pd.DatetimeIndex(dates))
    bt["equity"] = (1 + bt["portfolio_return"]).cumprod()

    w_hist = pd.DataFrame(
        np.vstack([w for _, w in weights_hist]),
        columns=tickers,
        index=[d for d, _ in weights_hist],
    )
    w_hist.index.name = "rebalance_date"

    return bt, w_hist


if __name__ == "__main__":
    rets = load_returns(RETURNS_PATH)
    bt, w_hist = run_backtest(rets)

    bt.to_csv("data/processed/backtest_median_mad.csv")
    w_hist.to_csv("data/processed/weights_median_mad.csv")

    print("OK - Backtest Mediana+MAD generado")
    print(bt.tail(3))
    print("Equity final:", bt["equity"].iloc[-1])

    print("\nÚltimos pesos (último rebalanceo):")
    print(w_hist.iloc[-1].sort_values(ascending=False).head(6))
    print("\nMAD_LIMIT:", MAD_LIMIT, "| N_SAMPLES:", N_SAMPLES)