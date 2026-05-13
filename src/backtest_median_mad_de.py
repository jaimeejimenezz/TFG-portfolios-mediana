import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

RETURNS_PATH = "data/processed/returns.csv"

LOOKBACK_DAYS = 252
REBALANCE_DAYS = 21

MAD_LIMIT = 0.0080
CAP = 0.10
SEED = 42

# Parámetros de Differential Evolution
#parametros primera ejecucion
#MAXITER = 40
#POPSIZE = 6
#TOL = 1e-6
#POLISH = False

#parametros segunda ejecucion
MAXITER = 80
POPSIZE = 8
TOL = 1e-7
POLISH = False

# Penalización si MAD supera el umbral
PENALTY = 10.0


def load_returns(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def project_capped_simplex(v: np.ndarray, cap: float | None = None) -> np.ndarray:
    """
    Proyecta un vector al conjunto:
        w_i >= 0
        sum(w_i) = 1
        w_i <= cap, si cap no es None
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
            "No existe una cartera factible."
        )

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

    diff = 1.0 - w.sum()

    if abs(diff) > 1e-10:
        free = (w > 1e-12) & (w < cap - 1e-12)

        if free.any():
            w[free] += diff / free.sum()
            w = np.clip(w, 0.0, cap)

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


def evaluate_solution(X: np.ndarray, w: np.ndarray) -> dict:
    rp = portfolio_returns(X, w)
    med = median_ret(rp)
    mad = mad_risk(rp)

    return {
        "median": med,
        "mad": mad,
    }


def objective_factory(X: np.ndarray, cap: float | None, mad_limit: float):
    """
    Crea la función objetivo para Differential Evolution.

    Como scipy minimiza:
        objetivo = -mediana + penalización

    Si MAD <= MAD_LIMIT:
        penalización = 0

    Si MAD > MAD_LIMIT:
        penalización = PENALTY * exceso
    """

    def objective(x: np.ndarray) -> float:
        w = project_capped_simplex(x, cap)
        stats = evaluate_solution(X, w)

        med = stats["median"]
        mad = stats["mad"]

        violation = max(0.0, mad - mad_limit)
        penalty = PENALTY * violation

        return -med + penalty

    return objective


def optimize_median_mad_de(
    window_returns: pd.DataFrame,
    mad_limit: float,
    cap: float | None,
    seed: int,
):
    X = window_returns.dropna().values
    n = X.shape[1]

    bounds = [(0.0, 1.0)] * n

    objective = objective_factory(X, cap, mad_limit)

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=TOL,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        polish=POLISH,
        updating="immediate",
        workers=1,
    )

    w = project_capped_simplex(result.x, cap)
    stats = evaluate_solution(X, w)

    opt_info = {
        "objective": float(result.fun),
        "median": float(stats["median"]),
        "mad": float(stats["mad"]),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
    }

    return w, opt_info


def run_backtest(returns: pd.DataFrame):
    rng = np.random.default_rng(SEED)

    tickers = list(returns.columns)
    n = len(tickers)

    w = project_capped_simplex(np.ones(n) / n, CAP)

    port_rets = []
    dates = []
    weights_hist = []
    optimization_stats = []

    for i, dt in enumerate(returns.index):
        if i % REBALANCE_DAYS == 0 and i >= LOOKBACK_DAYS:
            window = returns.iloc[i - LOOKBACK_DAYS:i]

            de_seed = int(rng.integers(0, 1_000_000_000))

            w, opt_info = optimize_median_mad_de(
                window_returns=window,
                mad_limit=MAD_LIMIT,
                cap=CAP,
                seed=de_seed,
            )

            weights_hist.append((dt, w.copy()))

            optimization_stats.append({
                "rebalance_date": dt,
                **opt_info,
                "weight_sum": float(w.sum()),
                "weight_max": float(w.max()),
            })

        rp_t = float(np.dot(w, returns.iloc[i].values))

        dates.append(dt)
        port_rets.append(rp_t)

    bt = pd.DataFrame(
        {"portfolio_return": port_rets},
        index=pd.DatetimeIndex(dates),
    )

    bt["equity"] = (1.0 + bt["portfolio_return"]).cumprod()

    w_hist = pd.DataFrame(
        np.vstack([w for _, w in weights_hist]),
        columns=tickers,
        index=[d for d, _ in weights_hist],
    )

    w_hist.index.name = "rebalance_date"

    stats_df = pd.DataFrame(optimization_stats)
    stats_df["rebalance_date"] = pd.to_datetime(stats_df["rebalance_date"])
    stats_df = stats_df.set_index("rebalance_date")

    return bt, w_hist, stats_df


if __name__ == "__main__":
    rets = load_returns(RETURNS_PATH)

    bt, w_hist, stats_df = run_backtest(rets)

    bt.to_csv("data/processed/backtest_median_mad_de.csv")
    w_hist.to_csv("data/processed/weights_median_mad_de.csv")
    stats_df.to_csv("data/processed/optimization_stats_median_mad_de.csv")

    print("OK - Backtest Mediana+MAD Differential Evolution generado")
    print(bt.tail(3))
    print("Equity final:", bt["equity"].iloc[-1])

    print("\nÚltimos pesos (último rebalanceo):")
    print(w_hist.iloc[-1].sort_values(ascending=False).head(12))

    print("\nComprobaciones:")
    print("Suma pesos último rebalanceo:", w_hist.iloc[-1].sum())
    print("Peso máximo último rebalanceo:", w_hist.iloc[-1].max())

    print("\nÚltimas estadísticas de optimización:")
    print(stats_df.tail(3))

    print("\nParámetros:")
    print("MAD_LIMIT:", MAD_LIMIT)
    print("CAP:", CAP)
    print("MAXITER:", MAXITER)
    print("POPSIZE:", POPSIZE)
    print("PENALTY:", PENALTY)