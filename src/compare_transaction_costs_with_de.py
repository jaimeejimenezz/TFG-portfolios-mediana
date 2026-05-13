import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.0
COST_RATE = 0.001  # 10 bps


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No se encontró ninguno de estos ficheros:\n"
        + "\n".join(paths)
    )


def load_backtest(path: str) -> pd.DataFrame:
    bt = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    if "portfolio_return" not in bt.columns:
        raise ValueError(f"El fichero {path} no contiene portfolio_return")

    if "equity" not in bt.columns:
        bt["equity"] = (1.0 + bt["portfolio_return"]).cumprod()

    return bt


def apply_transaction_costs(
    bt_path: str,
    weights_path: str,
    output_path: str,
    cost_rate: float,
) -> str:
    """
    Aplica costes proporcionales al turnover en cada fecha de rebalanceo.

    Se asume como cartera inicial una cartera equiponderada.
    El coste se descuenta directamente de la rentabilidad diaria
    en la fecha de rebalanceo.
    """

    bt = load_backtest(bt_path)
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True).sort_index()

    n_assets = weights.shape[1]
    previous_weights = np.ones(n_assets) / n_assets

    bt["portfolio_return_gross"] = bt["portfolio_return"].astype(float)
    bt["turnover"] = 0.0
    bt["transaction_cost"] = 0.0

    for rebalance_date, row in weights.iterrows():
        if rebalance_date not in bt.index:
            continue

        new_weights = row.astype(float).values

        turnover = float(np.abs(new_weights - previous_weights).sum())
        transaction_cost = float(cost_rate * turnover)

        bt.loc[rebalance_date, "turnover"] = turnover
        bt.loc[rebalance_date, "transaction_cost"] = transaction_cost

        previous_weights = new_weights

    bt["portfolio_return"] = (
        bt["portfolio_return_gross"] - bt["transaction_cost"]
    )

    bt["equity"] = (1.0 + bt["portfolio_return"]).cumprod()

    bt.to_csv(output_path)

    return output_path


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics(bt_path: str) -> dict:
    bt = load_backtest(bt_path)

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

    turnover_total = float(bt["turnover"].sum()) if "turnover" in bt.columns else 0.0
    turnover_mean = float(bt.loc[bt["turnover"] > 0, "turnover"].mean()) if "turnover" in bt.columns and (bt["turnover"] > 0).any() else 0.0
    cost_total = float(bt["transaction_cost"].sum()) if "transaction_cost" in bt.columns else 0.0

    return {
        "Equity final": float(equity.iloc[-1]),
        "Retorno total": total_return,
        "CAGR": cagr,
        "Vol anual": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Turnover total": turnover_total,
        "Turnover medio": turnover_mean,
        "Coste total": cost_total,
    }


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pretty = df.copy()

    pretty["Equity final"] = pretty["Equity final"].round(3)
    pretty["Retorno total"] = (pretty["Retorno total"] * 100).round(2).astype(str) + "%"
    pretty["CAGR"] = (pretty["CAGR"] * 100).round(2).astype(str) + "%"
    pretty["Vol anual"] = (pretty["Vol anual"] * 100).round(2).astype(str) + "%"
    pretty["Sharpe"] = pretty["Sharpe"].round(3)
    pretty["MaxDD"] = (pretty["MaxDD"] * 100).round(2).astype(str) + "%"
    pretty["Turnover total"] = pretty["Turnover total"].round(3)
    pretty["Turnover medio"] = pretty["Turnover medio"].round(3)
    pretty["Coste total"] = (pretty["Coste total"] * 100).round(2).astype(str) + "%"

    return pretty


if __name__ == "__main__":

    # 1) Ficheros brutos y pesos
    equal_bt = first_existing([
        "data/processed/backtest_equal_weight.csv",
    ])

    markowitz_bt = first_existing([
        "data/processed/backtest_markowitz_cap10_gross.csv",
        "data/processed/backtest_markowitz_cap10.csv",
    ])

    markowitz_w = first_existing([
        "data/processed/weights_markowitz_cap10_gross.csv",
        "data/processed/weights_markowitz_cap10.csv",
    ])

    random_bt = first_existing([
        "data/processed/backtest_median_mad_random_cap10_gross.csv",
        "data/processed/backtest_median_mad.csv",
    ])

    random_w = first_existing([
        "data/processed/weights_median_mad_random_cap10_gross.csv",
        "data/processed/weights_median_mad.csv",
    ])

    de_bt = first_existing([
        "data/processed/backtest_median_mad_de_cap10_gross.csv",
        "data/processed/backtest_median_mad_de.csv",
    ])

    de_w = first_existing([
        "data/processed/weights_median_mad_de_cap10_gross.csv",
        "data/processed/weights_median_mad_de.csv",
    ])

    # 2) Aplicar costes de 10 bps
    markowitz_costs = apply_transaction_costs(
        bt_path=markowitz_bt,
        weights_path=markowitz_w,
        output_path="data/processed/backtest_markowitz_cap10_costs_10bps.csv",
        cost_rate=COST_RATE,
    )

    random_costs = apply_transaction_costs(
        bt_path=random_bt,
        weights_path=random_w,
        output_path="data/processed/backtest_median_mad_random_cap10_costs_10bps.csv",
        cost_rate=COST_RATE,
    )

    de_costs = apply_transaction_costs(
        bt_path=de_bt,
        weights_path=de_w,
        output_path="data/processed/backtest_median_mad_de_cap10_costs_10bps.csv",
        cost_rate=COST_RATE,
    )

    # 3) Comparar métricas
    rows = [
        ("1/N", compute_metrics(equal_bt)),
        ("Markowitz CAP10 + costes", compute_metrics(markowitz_costs)),
        ("Mediana+MAD Random CAP10 + costes", compute_metrics(random_costs)),
        ("Mediana+MAD DE CAP10 + costes", compute_metrics(de_costs)),
    ]

    df = pd.DataFrame({name: vals for name, vals in rows}).T

    df.to_csv("data/processed/metrics_transaction_costs_with_de_10bps_raw.csv")

    pretty = format_metrics(df)
    pretty.to_csv("data/processed/metrics_transaction_costs_with_de_10bps.csv")

    print("\n=== Comparativa con CAP10 y costes de transacción de 10 bps incluyendo DE ===")
    print(pretty)

    print("\nGuardado:")
    print("data/processed/backtest_markowitz_cap10_costs_10bps.csv")
    print("data/processed/backtest_median_mad_random_cap10_costs_10bps.csv")
    print("data/processed/backtest_median_mad_de_cap10_costs_10bps.csv")
    print("data/processed/metrics_transaction_costs_with_de_10bps.csv")