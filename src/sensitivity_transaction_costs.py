import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.0

COST_RATES = {
    "0 bps": 0.0000,
    "5 bps": 0.0005,
    "10 bps": 0.0010,
    "20 bps": 0.0020,
    "50 bps": 0.0050,
}


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def compute_metrics(bt: pd.DataFrame) -> dict:
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

    total_turnover = float(bt["turnover"].sum()) if "turnover" in bt.columns else 0.0
    total_cost = float(bt["transaction_cost"].sum()) if "transaction_cost" in bt.columns else 0.0

    if "turnover" in bt.columns:
        rebalance_turnover = bt.loc[bt["turnover"] > 0, "turnover"]
        avg_turnover = float(rebalance_turnover.mean()) if len(rebalance_turnover) > 0 else 0.0
    else:
        avg_turnover = 0.0

    return {
        "Equity final": float(equity.iloc[-1]),
        "Retorno total": total_return,
        "CAGR": cagr,
        "Vol anual": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Turnover total": total_turnover,
        "Turnover medio": avg_turnover,
        "Coste total": total_cost,
    }


def load_gross_backtest(bt_path: str) -> pd.DataFrame:
    bt = pd.read_csv(bt_path, index_col=0, parse_dates=True).sort_index()

    # Nos aseguramos de partir siempre de la rentabilidad bruta original
    if "portfolio_return_gross" in bt.columns:
        bt["portfolio_return"] = bt["portfolio_return_gross"].astype(float)
    else:
        bt["portfolio_return"] = bt["portfolio_return"].astype(float)

    bt["equity"] = (1.0 + bt["portfolio_return"]).cumprod()

    return bt


def apply_costs_in_memory(
    bt_path: str,
    weights_path: str,
    cost_rate: float,
) -> pd.DataFrame:
    bt = load_gross_backtest(bt_path)
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True).sort_index()

    bt["portfolio_return_gross"] = bt["portfolio_return"].astype(float)
    bt["turnover"] = 0.0
    bt["transaction_cost"] = 0.0

    n_assets = weights.shape[1]

    # Cartera inicial 1/N antes del primer rebalanceo
    previous_w = np.ones(n_assets) / n_assets

    for dt, row in weights.iterrows():
        if dt not in bt.index:
            continue

        new_w = row.astype(float).values
        new_w = np.nan_to_num(new_w, nan=0.0)

        turnover = float(np.sum(np.abs(new_w - previous_w)))
        transaction_cost = float(turnover * cost_rate)

        bt.loc[dt, "turnover"] = turnover
        bt.loc[dt, "transaction_cost"] = transaction_cost

        previous_w = new_w.copy()

    bt["portfolio_return"] = bt["portfolio_return_gross"] - bt["transaction_cost"]
    bt["equity"] = (1.0 + bt["portfolio_return"]).cumprod()

    return bt


def pretty_format(df: pd.DataFrame) -> pd.DataFrame:
    pretty = df.copy()

    pretty["Equity final"] = pretty["Equity final"].round(3)
    pretty["Retorno total"] = (pretty["Retorno total"] * 100).round(2).astype(str) + "%"
    pretty["CAGR"] = (pretty["CAGR"] * 100).round(2).astype(str) + "%"
    pretty["Vol anual"] = (pretty["Vol anual"] * 100).round(2).astype(str) + "%"
    pretty["MaxDD"] = (pretty["MaxDD"] * 100).round(2).astype(str) + "%"
    pretty["Sharpe"] = pretty["Sharpe"].round(3)
    pretty["Turnover total"] = pretty["Turnover total"].round(3)
    pretty["Turnover medio"] = pretty["Turnover medio"].round(3)
    pretty["Coste total"] = (pretty["Coste total"] * 100).round(2).astype(str) + "%"

    return pretty


if __name__ == "__main__":
    rows = []

    # Benchmark 1/N: lo mantenemos sin costes explícitos.
    # Esto hace la comparación conservadora, porque no penalizamos al benchmark.
    bt_equal = load_gross_backtest("data/processed/backtest_equal_weight.csv")
    metrics_equal = compute_metrics(bt_equal)

    for cost_label, cost_rate in COST_RATES.items():
        rows.append({
            "Coste": cost_label,
            "Método": "1/N",
            **metrics_equal,
        })

        bt_markowitz = apply_costs_in_memory(
            bt_path="data/processed/backtest_markowitz_cap10_gross.csv",
            weights_path="data/processed/weights_markowitz_cap10_gross.csv",
            cost_rate=cost_rate,
        )

        rows.append({
            "Coste": cost_label,
            "Método": "Markowitz CAP10",
            **compute_metrics(bt_markowitz),
        })

        bt_median_mad = apply_costs_in_memory(
            bt_path="data/processed/backtest_median_mad_cap10_gross.csv",
            weights_path="data/processed/weights_median_mad_cap10_gross.csv",
            cost_rate=cost_rate,
        )

        rows.append({
            "Coste": cost_label,
            "Método": "Mediana+MAD CAP10",
            **compute_metrics(bt_median_mad),
        })

    df = pd.DataFrame(rows)

    pretty = pretty_format(df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n=== Sensibilidad a costes de transacción ===")
    print(pretty)

    pretty.to_csv("data/processed/sensitivity_transaction_costs.csv", index=False)
    print("\nGuardado: data/processed/sensitivity_transaction_costs.csv")

    # Tabla resumida solo con equity final
    pivot_equity = df.pivot(index="Coste", columns="Método", values="Equity final")
    pivot_sharpe = df.pivot(index="Coste", columns="Método", values="Sharpe")

    print("\n=== Resumen Equity final ===")
    print(pivot_equity.round(3))

    print("\n=== Resumen Sharpe ===")
    print(pivot_sharpe.round(3))

    # Ganador por equity en cada nivel de coste
    winners = df.loc[df.groupby("Coste")["Equity final"].idxmax(), ["Coste", "Método", "Equity final", "Sharpe", "MaxDD"]]

    print("\n=== Ganador por coste según Equity final ===")
    print(winners.round(4))