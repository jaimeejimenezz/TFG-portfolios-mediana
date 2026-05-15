import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF = 0.0
COST_RATE = 0.001  # 10 bps

TURNOVER_PENALTIES = [0.0005, 0.0010, 0.0020]


def penalty_tag(turnover_penalty: float) -> str:
    return f"tp_{turnover_penalty:.4f}".replace(".", "p")


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


def apply_transaction_costs_to_df(
    bt: pd.DataFrame,
    weights: pd.DataFrame,
    cost_rate: float,
) -> pd.DataFrame:
    out = bt.copy()

    n_assets = weights.shape[1]
    previous_weights = np.ones(n_assets) / n_assets

    out["portfolio_return_gross"] = out["portfolio_return"].astype(float)
    out["turnover"] = 0.0
    out["transaction_cost"] = 0.0

    for rebalance_date, row in weights.iterrows():
        if rebalance_date not in out.index:
            continue

        new_weights = row.astype(float).values

        turnover = float(np.abs(new_weights - previous_weights).sum())
        transaction_cost = float(cost_rate * turnover)

        out.loc[rebalance_date, "turnover"] = turnover
        out.loc[rebalance_date, "transaction_cost"] = transaction_cost

        previous_weights = new_weights

    out["portfolio_return"] = (
        out["portfolio_return_gross"] - out["transaction_cost"]
    )

    out["equity"] = (1.0 + out["portfolio_return"]).cumprod()

    return out


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_metrics_from_df(bt: pd.DataFrame) -> dict:
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
    turnover_mean = (
        float(bt.loc[bt["turnover"] > 0, "turnover"].mean())
        if "turnover" in bt.columns and (bt["turnover"] > 0).any()
        else 0.0
    )
    cost_total = (
        float(bt["transaction_cost"].sum())
        if "transaction_cost" in bt.columns
        else 0.0
    )

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


def load_weights(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


if __name__ == "__main__":
    rows = []

    # 1) 1/N sin costes explícitos
    equal_bt_path = first_existing([
        "data/processed/backtest_equal_weight.csv",
    ])

    equal_bt = load_backtest(equal_bt_path)
    rows.append({
        "Método": "1/N",
        "TURNOVER_PENALTY": np.nan,
        **compute_metrics_from_df(equal_bt),
    })

    # 2) Markowitz con costes
    markowitz_bt_path = first_existing([
        "data/processed/backtest_markowitz_cap10_gross.csv",
        "data/processed/backtest_markowitz_cap10.csv",
    ])

    markowitz_w_path = first_existing([
        "data/processed/weights_markowitz_cap10_gross.csv",
        "data/processed/weights_markowitz_cap10.csv",
    ])

    markowitz_bt = load_backtest(markowitz_bt_path)
    markowitz_w = load_weights(markowitz_w_path)

    markowitz_costs = apply_transaction_costs_to_df(
        bt=markowitz_bt,
        weights=markowitz_w,
        cost_rate=COST_RATE,
    )

    markowitz_costs.to_csv(
        "data/processed/backtest_markowitz_cap10_costs_10bps_step9.csv"
    )

    rows.append({
        "Método": "Markowitz CAP10 + costes",
        "TURNOVER_PENALTY": np.nan,
        **compute_metrics_from_df(markowitz_costs),
    })

    # 3) Random con costes
    random_bt_path = first_existing([
        "data/processed/backtest_median_mad_random_cap10_gross.csv",
        "data/processed/backtest_median_mad.csv",
    ])

    random_w_path = first_existing([
        "data/processed/weights_median_mad_random_cap10_gross.csv",
        "data/processed/weights_median_mad.csv",
    ])

    random_bt = load_backtest(random_bt_path)
    random_w = load_weights(random_w_path)

    random_costs = apply_transaction_costs_to_df(
        bt=random_bt,
        weights=random_w,
        cost_rate=COST_RATE,
    )

    random_costs.to_csv(
        "data/processed/backtest_median_mad_random_cap10_costs_10bps_step9.csv"
    )

    rows.append({
        "Método": "Mediana+MAD Random CAP10 + costes",
        "TURNOVER_PENALTY": np.nan,
        **compute_metrics_from_df(random_costs),
    })

    # 4) DE base con costes
    de_bt_path = first_existing([
        "data/processed/backtest_median_mad_de_cap10_gross.csv",
        "data/processed/backtest_median_mad_de.csv",
    ])

    de_w_path = first_existing([
        "data/processed/weights_median_mad_de_cap10_gross.csv",
        "data/processed/weights_median_mad_de.csv",
    ])

    de_bt = load_backtest(de_bt_path)
    de_w = load_weights(de_w_path)

    de_costs = apply_transaction_costs_to_df(
        bt=de_bt,
        weights=de_w,
        cost_rate=COST_RATE,
    )

    de_costs.to_csv(
        "data/processed/backtest_median_mad_de_cap10_costs_10bps_step9.csv"
    )

    rows.append({
        "Método": "Mediana+MAD DE CAP10 + costes",
        "TURNOVER_PENALTY": 0.0,
        **compute_metrics_from_df(de_costs),
    })

    # 5) DE con penalización explícita por turnover
    for turnover_penalty in TURNOVER_PENALTIES:
        tag = penalty_tag(turnover_penalty)

        bt_path = first_existing([
            f"data/processed/backtest_median_mad_de_turnover_{tag}.csv",
        ])

        weights_path = first_existing([
            f"data/processed/weights_median_mad_de_turnover_{tag}.csv",
        ])

        bt = load_backtest(bt_path)
        weights = load_weights(weights_path)

        bt_costs = apply_transaction_costs_to_df(
            bt=bt,
            weights=weights,
            cost_rate=COST_RATE,
        )

        output_path = (
            f"data/processed/backtest_median_mad_de_turnover_{tag}_costs_10bps.csv"
        )

        bt_costs.to_csv(output_path)

        rows.append({
            "Método": f"Mediana+MAD DE + turnover penalty {turnover_penalty}",
            "TURNOVER_PENALTY": turnover_penalty,
            **compute_metrics_from_df(bt_costs),
        })

    df = pd.DataFrame(rows)

    df.to_csv("data/processed/metrics_de_turnover_penalty_10bps_raw.csv", index=False)

    pretty = format_metrics(df)
    pretty.to_csv("data/processed/metrics_de_turnover_penalty_10bps.csv", index=False)

    print("\n=== Comparativa DE con penalización por turnover y costes de 10 bps ===")
    print(pretty)

    print("\nGuardado:")
    print("data/processed/metrics_de_turnover_penalty_10bps.csv")
    print("data/processed/metrics_de_turnover_penalty_10bps_raw.csv")