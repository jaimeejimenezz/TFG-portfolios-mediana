import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.0

OUTPUT_RAW = "data/processed/metrics_subperiod_robustness_10bps_raw.csv"
OUTPUT_PRETTY = "data/processed/metrics_subperiod_robustness_10bps.csv"


SUBPERIODS = [
    {
        "Subperiodo": "2017-2019",
        "Interpretación": "Mercado previo al COVID",
        "start": "2017-01-01",
        "end": "2019-12-31",
    },
    {
        "Subperiodo": "2020",
        "Interpretación": "Shock COVID y recuperación",
        "start": "2020-01-01",
        "end": "2020-12-31",
    },
    {
        "Subperiodo": "2021-2022",
        "Interpretación": "Inflación, subidas de tipos y mercado más difícil",
        "start": "2021-01-01",
        "end": "2022-12-31",
    },
    {
        "Subperiodo": "2023-2026",
        "Interpretación": "Recuperación y nuevo ciclo alcista",
        "start": "2023-01-01",
        "end": "2026-02-27",
    },
]


METHOD_FILES = [
    {
        "Método": "1/N",
        "paths": [
            "data/processed/backtest_equal_weight.csv",
        ],
    },
    {
        "Método": "Markowitz CAP10 + costes",
        "paths": [
            "data/processed/backtest_markowitz_cap10_costs_10bps_step9.csv",
            "data/processed/backtest_markowitz_cap10_costs_10bps.csv",
            "data/processed/backtest_markowitz_cap10_costs.csv",
        ],
    },
    {
        "Método": "Mediana+MAD Random CAP10 + costes",
        "paths": [
            "data/processed/backtest_median_mad_random_cap10_costs_10bps_step9.csv",
            "data/processed/backtest_median_mad_random_cap10_costs_10bps.csv",
            "data/processed/backtest_median_mad_cap10_costs.csv",
        ],
    },
    {
        "Método": "Mediana+MAD DE CAP10 + costes",
        "paths": [
            "data/processed/backtest_median_mad_de_cap10_costs_10bps_step9.csv",
            "data/processed/backtest_median_mad_de_cap10_costs_10bps.csv",
        ],
    },
    {
        "Método": "Mediana+MAD DE + η = 0,0010",
        "paths": [
            "data/processed/backtest_median_mad_de_turnover_tp_0p0010_costs_10bps.csv",
        ],
    },
]


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

    if "turnover" not in bt.columns:
        bt["turnover"] = 0.0

    if "transaction_cost" not in bt.columns:
        bt["transaction_cost"] = 0.0

    return bt


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def compute_subperiod_metrics(bt: pd.DataFrame, start: str, end: str) -> dict:
    sub = bt.loc[start:end].copy()

    if sub.empty:
        raise ValueError(f"No hay datos para el subperiodo {start} - {end}")

    returns = sub["portfolio_return"].astype(float)
    equity = (1.0 + returns).cumprod()

    n_days = len(returns)
    n_years = n_days / TRADING_DAYS

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean_ann = float(returns.mean() * TRADING_DAYS)
    sharpe = float((mean_ann - RF) / vol) if vol > 0 else np.nan
    mdd = max_drawdown(equity)

    turnover_total = float(sub["turnover"].sum())
    cost_total = float(sub["transaction_cost"].sum())

    return {
        "N días": n_days,
        "Retorno total": total_return,
        "CAGR": cagr,
        "Vol anual": vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Turnover total": turnover_total,
        "Coste total": cost_total,
    }


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pretty = df.copy()

    pretty["Retorno total"] = (
        (pretty["Retorno total"] * 100).round(2).astype(str) + "%"
    )
    pretty["CAGR"] = (
        (pretty["CAGR"] * 100).round(2).astype(str) + "%"
    )
    pretty["Vol anual"] = (
        (pretty["Vol anual"] * 100).round(2).astype(str) + "%"
    )
    pretty["Sharpe"] = pretty["Sharpe"].round(3)
    pretty["MaxDD"] = (
        (pretty["MaxDD"] * 100).round(2).astype(str) + "%"
    )
    pretty["Turnover total"] = pretty["Turnover total"].round(3)
    pretty["Coste total"] = (
        (pretty["Coste total"] * 100).round(2).astype(str) + "%"
    )

    return pretty


def main():
    rows = []

    for method_info in METHOD_FILES:
        method = method_info["Método"]
        path = first_existing(method_info["paths"])
        bt = load_backtest(path)

        for subperiod in SUBPERIODS:
            metrics = compute_subperiod_metrics(
                bt=bt,
                start=subperiod["start"],
                end=subperiod["end"],
            )

            rows.append({
                "Subperiodo": subperiod["Subperiodo"],
                "Interpretación": subperiod["Interpretación"],
                "Método": method,
                **metrics,
            })

    raw = pd.DataFrame(rows)

    os.makedirs("data/processed", exist_ok=True)

    raw.to_csv(OUTPUT_RAW, index=False)

    pretty = format_metrics(raw)
    pretty.to_csv(OUTPUT_PRETTY, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    print("\n=== Análisis de robustez por subperiodos de mercado ===")
    print(pretty.to_string(index=False))

    print("\nGuardado:")
    print(OUTPUT_RAW)
    print(OUTPUT_PRETTY)


if __name__ == "__main__":
    main()