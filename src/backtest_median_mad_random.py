import numpy as np
import pandas as pd

RETURNS_PATH = "data/processed/returns.csv"

LOOKBACK_DAYS = 252
REBALANCE_DAYS = 21

N_SAMPLES = 3000
MAD_LIMIT = 0.0080
CAP = 0.10
SEED = 42


def load_returns(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def project_capped_simplex(v: np.ndarray, cap: float | None = None) -> np.ndarray:
    """
    Proyecta un vector de pesos al conjunto factible:

        w_i >= 0
        sum(w_i) = 1
        w_i <= cap, si cap no es None

    Si cap=None, simplemente recorta negativos y normaliza.
    Si cap existe, usa bisección para imponer simplex + límite superior.
    """
    v = np.asarray(v, dtype=float)
    n = v.size

    if cap is None:
        v = np.clip(v, 0.0, None)
        s = v.sum()
        if s == 0:
            return np.ones(n) / n
        return v / s

    cap = float(cap)

    if cap * n < 1.0 - 1e-12:
        raise ValueError(
            f"CAP demasiado bajo: cap*n={cap * n:.3f} < 1. "
            "No existe una cartera factible con esos límites."
        )

    # Buscamos lambda tal que:
    # sum(clip(v - lambda, 0, cap)) = 1
    lo = v.min() - cap
    hi = v.max()

    for _ in range(100):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, 0.0, cap)
        s = w.sum()

        if s > 1.0:
            lo = mid
        else:
            hi = mid

    w = np.clip(v - hi, 0.0, cap)

    # Ajuste numérico final para que sume exactamente 1 sin romper el cap
    diff = 1.0 - w.sum()

    if abs(diff) > 1e-10:
        free = (w > 1e-12) & (w < cap - 1e-12)

        if free.any():
            w[free] += diff / free.sum()
            w = np.clip(w, 0.0, cap)

    # Última normalización defensiva si el error numérico es minúsculo
    if abs(w.sum() - 1.0) > 1e-8:
        free = w < cap - 1e-12

        if free.any():
            w[free] += (1.0 - w.sum()) / free.sum()
            w = np.clip(w, 0.0, cap)

    return w


def portfolio_returns(window: np.ndarray, w: np.ndarray) -> np.ndarray:
    return window @ w


def median_ret(rp: np.ndarray) -> float:
    return float(np.median(rp))


def mad_risk(rp: np.ndarray) -> float:
    m = np.median(rp)
    return float(np.mean(np.abs(rp - m)))


def sample_weights_capped(n: int, rng: np.random.Generator, cap: float | None) -> np.ndarray:
    """
    Genera una cartera aleatoria factible.
    Primero genera pesos aleatorios y después los proyecta al simplex con cap.
    """
    raw = rng.random(n)
    return project_capped_simplex(raw, cap)


def optimize_median_mad(
    window_returns: pd.DataFrame,
    n_samples: int,
    mad_limit: float,
    cap: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    X = window_returns.dropna().values
    n = X.shape[1]

    best_w = project_capped_simplex(np.ones(n) / n, cap)
    best_score = -1e18

    # Fallback: si ninguna cartera cumple MAD_LIMIT, nos quedamos con la de menor MAD
    best_mad = 1e18
    best_mad_w = best_w.copy()

    for _ in range(n_samples):
        w = sample_weights_capped(n, rng, cap)

        rp = portfolio_returns(X, w)

        mad = mad_risk(rp)
        med = median_ret(rp)

        if mad <= mad_limit:
            if med > best_score:
                best_score = med
                best_w = w
        else:
            if mad < best_mad:
                best_mad = mad
                best_mad_w = w

    # Si no hubo ninguna solución factible por MAD, usamos la de menor MAD
    if best_score == -1e18:
        best_w = best_mad_w

    return project_capped_simplex(best_w, cap)


def run_backtest(returns: pd.DataFrame):
    rng = np.random.default_rng(SEED)
    tickers = list(returns.columns)
    n = len(tickers)

    w = project_capped_simplex(np.ones(n) / n, CAP)

    port_rets = []
    dates = []
    weights_hist = []

    for i, dt in enumerate(returns.index):
        if i % REBALANCE_DAYS == 0 and i >= LOOKBACK_DAYS:
            window = returns.iloc[i - LOOKBACK_DAYS:i]
            w = optimize_median_mad(window, N_SAMPLES, MAD_LIMIT, CAP, rng)
            weights_hist.append((dt, w.copy()))

        rp_t = float(np.dot(w, returns.iloc[i].values))

        dates.append(dt)
        port_rets.append(rp_t)

    bt = pd.DataFrame(
        {"portfolio_return": port_rets},
        index=pd.DatetimeIndex(dates),
    )

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
    print(w_hist.iloc[-1].sort_values(ascending=False).head(12))

    print("\nComprobaciones:")
    print("Suma pesos último rebalanceo:", w_hist.iloc[-1].sum())
    print("Peso máximo último rebalanceo:", w_hist.iloc[-1].max())

    print("\nMAD_LIMIT:", MAD_LIMIT, "| N_SAMPLES:", N_SAMPLES, "| CAP:", CAP)