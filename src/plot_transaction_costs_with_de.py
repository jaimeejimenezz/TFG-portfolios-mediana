import os
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_bt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    if "equity" not in df.columns:
        df["equity"] = (1.0 + df["portfolio_return"]).cumprod()

    return df


def max_drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def clean_cost_label(x: str) -> int:
    return int(str(x).replace(" bps", "").strip())


def plot_equity_with_costs():
    files = {
        "1/N": "data/processed/backtest_equal_weight.csv",
        "Markowitz CAP10 + costes": "data/processed/backtest_markowitz_cap10_costs_10bps.csv",
        "Mediana+MAD Random CAP10 + costes": "data/processed/backtest_median_mad_random_cap10_costs_10bps.csv",
        "Mediana+MAD DE CAP10 + costes": "data/processed/backtest_median_mad_de_cap10_costs_10bps.csv",
    }

    plt.figure(figsize=(10, 6))

    for label, path in files.items():
        bt = load_bt(path)
        plt.plot(bt.index, bt["equity"], label=label)

    plt.title("Evolución del capital acumulado con CAP10 y costes de 10 bps")
    plt.xlabel("Fecha")
    plt.ylabel("Capital acumulado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/equity_cap10_costs_10bps_with_de.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_drawdown_with_costs():
    files = {
        "1/N": "data/processed/backtest_equal_weight.csv",
        "Markowitz CAP10 + costes": "data/processed/backtest_markowitz_cap10_costs_10bps.csv",
        "Mediana+MAD Random CAP10 + costes": "data/processed/backtest_median_mad_random_cap10_costs_10bps.csv",
        "Mediana+MAD DE CAP10 + costes": "data/processed/backtest_median_mad_de_cap10_costs_10bps.csv",
    }

    plt.figure(figsize=(10, 6))

    for label, path in files.items():
        bt = load_bt(path)
        dd = max_drawdown_series(bt["equity"]) * 100
        plt.plot(bt.index, dd, label=label)

    plt.title("Drawdown con CAP10 y costes de 10 bps")
    plt.xlabel("Fecha")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/drawdown_cap10_costs_10bps_with_de.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_sensitivity_equity():
    df = pd.read_csv("data/processed/sensitivity_transaction_costs_with_de_raw.csv")

    df["Coste_num"] = df["Coste"].apply(clean_cost_label)

    plt.figure(figsize=(10, 6))

    for method in df["Método"].unique():
        sub = df[df["Método"] == method].sort_values("Coste_num")
        plt.plot(sub["Coste_num"], sub["Equity final"], marker="o", label=method)

    plt.title("Sensibilidad del equity final al coste de transacción")
    plt.xlabel("Coste de transacción (bps)")
    plt.ylabel("Equity final")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/sensitivity_equity_with_de.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_sensitivity_sharpe():
    df = pd.read_csv("data/processed/sensitivity_transaction_costs_with_de_raw.csv")

    df["Coste_num"] = df["Coste"].apply(clean_cost_label)

    plt.figure(figsize=(10, 6))

    for method in df["Método"].unique():
        sub = df[df["Método"] == method].sort_values("Coste_num")
        plt.plot(sub["Coste_num"], sub["Sharpe"], marker="o", label=method)

    plt.title("Sensibilidad del ratio Sharpe al coste de transacción")
    plt.xlabel("Coste de transacción (bps)")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/sensitivity_sharpe_with_de.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


if __name__ == "__main__":
    plot_equity_with_costs()
    plot_drawdown_with_costs()
    plot_sensitivity_equity()
    plot_sensitivity_sharpe()

    print("\nOK - Gráficas con Differential Evolution generadas.")