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
MAXITER = 80
POPSIZE = 8
TOL = 1e-7
POLISH = False

# Penalización si MAD supera el umbral
PENALTY = 10.0

# Valores de penalización explícita por turnover
# 0.0000 ya lo tienes como DE base del Paso 8, por eso aquí probamos solo extensiones.
TURNOVER_PENALTIES = [0.0005, 0.0010, 0.0020]

TRADING_DAYS = 252
RF = 0.0


def load_returns(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def penalty_tag(turnover_penalty: float) -> str:
    """
    Convierte 0.0005 -> tp_0p0005 para usarlo en nombres de ficheros.
    """
    return f"tp_{turnover_penalty:.4f}".replace(".", "p")


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


def turnover_distance(w_new: np.ndarray, w_previous: np.ndarray) -> float:
    """
    Distancia L1 entre los pesos candidatos y los pesos anteriores.
    Equivale al turnover necesario para pasar de una cartera a otra.
    """
    return float(np.abs(w_new - w_previous).sum())


def evaluate_solution(X: np.ndarray, w: np.ndarray) -> dict:
    rp = portfolio_returns(X, w)
    med = median_ret(rp)
    mad = mad_risk(rp)

    return {
        "median": med,
        "mad": mad,
    }


def objective_factory(
    X: np.ndarray,
    cap: float | None,
    mad_limit: float,
    previous_weights: np.ndarray,
    turnover_penalty: float,
):
    """
    Crea la función objetivo para Differential Evolution con penalización explícita por turnover.

    Como scipy minimiza:

        objetivo = -mediana
                   + penalización por superar MAD_LIMIT
                   + penalización por turnover

    La penalización por turnover es:

        turnover_penalty * sum(abs(w_new - w_previous))

    donde w_previous son los pesos vigentes antes del rebalanceo.
    """

    previous_weights = np.asarray(previous_weights, dtype=float)

    def objective(x: np.ndarray) -> float:
        w = project_capped_simplex(x, cap)
        stats = evaluate_solution(X, w)

        med = stats["median"]
        mad = stats["mad"]

        mad_violation = max(0.0, mad - mad_limit)
        mad_penalty = PENALTY * mad_violation

        turnover = turnover_distance(w, previous_weights)
        turnover_cost = turnover_penalty * turnover

        return -med + mad_penalty + turnover_cost

    return objective


def optimize_median_mad_de_turnover(
    window_returns: pd.DataFrame,
    mad_limit: float,
    cap: float | None,
    seed: int,
    previous_weights: np.ndarray,
    turnover_penalty: float,
):
    X = window_returns.dropna().values
    n = X.shape[1]

    bounds = [(0.0, 1.0)] * n

    objective = objective_factory(
        X=X,
        cap=cap,
        mad_limit=mad_limit,
        previous_weights=previous_weights,
        turnover_penalty=turnover_penalty,
    )

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

    realized_turnover = turnover_distance(w, previous_weights)

    opt_info = {
        "objective": float(result.fun),
        "median": float(stats["median"]),
        "mad": float(stats["mad"]),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "message": str(result.message),
        "turnover_to_previous": float(realized_turnover),
        "turnover_penalty": float(turnover_penalty),
    }

    return w, opt_info


def run_backtest(returns: pd.DataFrame, turnover_penalty: float):
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

            previous_weights = w.copy()
            de_seed = int(rng.integers(0, 1_000_000_000))

            w, opt_info = optimize_median_mad_de_turnover(
                window_returns=window,
                mad_limit=MAD_LIMIT,
                cap=CAP,
                seed=de_seed,
                previous_weights=previous_weights,
                turnover_penalty=turnover_penalty,
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


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics_from_bt(bt: pd.DataFrame) -> dict:
    r = bt["portfolio_return"].astype(float)
    equity = bt["equity"].astype(float)

    n_days = len(r)
    n_years = n_days / TRADING_DAYS

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(r.mean() * TRADING_DAYS)
    sharpe = float((mean_ann - RF) / vol) if vol > 0 else np.nan
    mdd = max_drawdown(equity)

    return {
        "Equity final": float(equity.iloc[-1]),
        "Retorno total": total_return,
        "CAGR": cagr,
        "Vol anual": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
    }


def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    pretty = df.copy()

    pretty["Equity final"] = pretty["Equity final"].round(3)
    pretty["Retorno total"] = (pretty["Retorno total"] * 100).round(2).astype(str) + "%"
    pretty["CAGR"] = (pretty["CAGR"] * 100).round(2).astype(str) + "%"
    pretty["Vol anual"] = (pretty["Vol anual"] * 100).round(2).astype(str) + "%"
    pretty["Sharpe"] = pretty["Sharpe"].round(3)
    pretty["MaxDD"] = (pretty["MaxDD"] * 100).round(2).astype(str) + "%"
    pretty["Turnover optimización total"] = pretty["Turnover optimización total"].round(3)
    pretty["Turnover optimización medio"] = pretty["Turnover optimización medio"].round(3)

    return pretty


if __name__ == "__main__":
    rets = load_returns(RETURNS_PATH)

    summary_rows = []

    for turnover_penalty in TURNOVER_PENALTIES:
        tag = penalty_tag(turnover_penalty)

        print("\n" + "=" * 80)
        print(f"Ejecutando Mediana+MAD DE con penalización de turnover: {turnover_penalty}")
        print("=" * 80)

        bt, w_hist, stats_df = run_backtest(
            returns=rets,
            turnover_penalty=turnover_penalty,
        )

        bt_path = f"data/processed/backtest_median_mad_de_turnover_{tag}.csv"
        weights_path = f"data/processed/weights_median_mad_de_turnover_{tag}.csv"
        stats_path = f"data/processed/optimization_stats_median_mad_de_turnover_{tag}.csv"

        bt.to_csv(bt_path)
        w_hist.to_csv(weights_path)
        stats_df.to_csv(stats_path)

        metrics = compute_metrics_from_bt(bt)

        turnover_total = float(stats_df["turnover_to_previous"].sum())
        turnover_mean = float(stats_df["turnover_to_previous"].mean())

        summary_rows.append({
            "Método": f"Mediana+MAD DE turnover penalty {turnover_penalty}",
            "TURNOVER_PENALTY": turnover_penalty,
            **metrics,
            "Turnover optimización total": turnover_total,
            "Turnover optimización medio": turnover_mean,
            "Último MAD": float(stats_df["mad"].iloc[-1]),
            "Última suma pesos": float(w_hist.iloc[-1].sum()),
            "Último peso máximo": float(w_hist.iloc[-1].max()),
        })

        print("OK - Backtest generado")
        print("Fichero:", bt_path)
        print("Equity final:", bt["equity"].iloc[-1])
        print("Sharpe bruto:", metrics["Sharpe"])
        print("MaxDD bruto:", metrics["MaxDD"])
        print("Turnover total frente a pesos anteriores:", turnover_total)
        print("Último MAD:", stats_df["mad"].iloc[-1])
        print("Suma pesos último rebalanceo:", w_hist.iloc[-1].sum())
        print("Peso máximo último rebalanceo:", w_hist.iloc[-1].max())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("data/processed/metrics_de_turnover_penalty_gross_raw.csv", index=False)

    pretty = format_summary(summary)
    pretty.to_csv("data/processed/metrics_de_turnover_penalty_gross.csv", index=False)

    print("\n=== Resumen bruto de DE con penalización por turnover ===")
    print(pretty)

    print("\nGuardado:")
    print("data/processed/metrics_de_turnover_penalty_gross.csv")
    print("data/processed/metrics_de_turnover_penalty_gross_raw.csv")

    