import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def compute_metrics(bt_path: str):
    bt = pd.read_csv(bt_path, index_col=0, parse_dates=True)

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
        turnover_reb = bt.loc[bt["turnover"] > 0, "turnover"]
        avg_turnover = float(turnover_reb.mean()) if len(turnover_reb) > 0 else 0.0
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


if __name__ == "__main__":
    rows = [
        (
            "1/N",
            compute_metrics("data/processed/backtest_equal_weight.csv"),
        ),
        (
            "Markowitz CAP10 + costes",
            compute_metrics("data/processed/backtest_markowitz_cap10_costs.csv"),
        ),
        (
            "Mediana+MAD CAP10 + costes",
            compute_metrics("data/processed/backtest_median_mad_cap10_costs.csv"),
        ),
    ]

    df = pd.DataFrame({name: vals for name, vals in rows}).T

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

    print("\n=== Comparativa con CAP 10% y costes de transacción ===")
    print(pretty)

    pretty.to_csv("data/processed/metrics_3_methods_cap10_costs.csv")
    print("\nGuardado: data/processed/metrics_3_methods_cap10_costs.csv")