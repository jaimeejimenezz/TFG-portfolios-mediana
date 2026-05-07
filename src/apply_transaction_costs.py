import numpy as np
import pandas as pd

COST_RATE = 0.001  # 10 bps = 0.10% por unidad de turnover


def apply_costs(
    bt_path: str,
    weights_path: str,
    output_path: str,
    cost_rate: float = COST_RATE,
) -> pd.DataFrame:
    """
    Aplica costes de transacción a un backtest ya calculado.

    El backtest debe tener:
        - portfolio_return
        - equity

    El archivo de pesos debe tener:
        - índice = fechas de rebalanceo
        - columnas = activos
        - valores = pesos objetivo

    Se calcula:
        turnover_t = sum(abs(w_new - w_old))
        transaction_cost_t = turnover_t * cost_rate
        portfolio_return_net_t = portfolio_return_gross_t - transaction_cost_t
    """

    bt = pd.read_csv(bt_path, index_col=0, parse_dates=True).sort_index()
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True).sort_index()

    bt["portfolio_return_gross"] = bt["portfolio_return"].astype(float)
    bt["turnover"] = 0.0
    bt["transaction_cost"] = 0.0

    n_assets = weights.shape[1]

    # Suponemos que la cartera inicial antes del primer rebalanceo es 1/N
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

    bt.to_csv(output_path)

    return bt


def print_summary(name: str, bt: pd.DataFrame):
    rebalance_turnover = bt.loc[bt["turnover"] > 0, "turnover"]

    total_turnover = float(bt["turnover"].sum())
    avg_turnover = float(rebalance_turnover.mean()) if len(rebalance_turnover) > 0 else 0.0
    max_turnover = float(rebalance_turnover.max()) if len(rebalance_turnover) > 0 else 0.0
    total_cost = float(bt["transaction_cost"].sum())
    final_equity = float(bt["equity"].iloc[-1])

    print(f"\n=== {name} con costes ===")
    print(f"Equity final neto: {final_equity:.6f}")
    print(f"Turnover total: {total_turnover:.4f}")
    print(f"Turnover medio por rebalanceo: {avg_turnover:.4f}")
    print(f"Turnover máximo en un rebalanceo: {max_turnover:.4f}")
    print(f"Coste total acumulado aproximado: {total_cost:.4f}")


if __name__ == "__main__":
    experiments = [
        {
            "name": "Markowitz CAP 10%",
            "bt_path": "data/processed/backtest_markowitz_cap10_gross.csv",
            "weights_path": "data/processed/weights_markowitz_cap10_gross.csv",
            "output_path": "data/processed/backtest_markowitz_cap10_costs.csv",
        },
        {
            "name": "Mediana+MAD CAP 10%",
            "bt_path": "data/processed/backtest_median_mad_cap10_gross.csv",
            "weights_path": "data/processed/weights_median_mad_cap10_gross.csv",
            "output_path": "data/processed/backtest_median_mad_cap10_costs.csv",
        },
    ]

    print(f"Aplicando costes de transacción: COST_RATE = {COST_RATE:.4%}")

    for exp in experiments:
        bt_net = apply_costs(
            bt_path=exp["bt_path"],
            weights_path=exp["weights_path"],
            output_path=exp["output_path"],
            cost_rate=COST_RATE,
        )

        print_summary(exp["name"], bt_net)
        print(f"Guardado: {exp['output_path']}")