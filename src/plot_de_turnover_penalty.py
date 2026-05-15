import os
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_PATH = "data/processed/metrics_de_turnover_penalty_10bps_raw.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    # Nos quedamos solo con las variantes DE con TURNOVER_PENALTY definido
    df = df[df["TURNOVER_PENALTY"].notna()].copy()

    df["TURNOVER_PENALTY"] = df["TURNOVER_PENALTY"].astype(float)

    return df.sort_values("TURNOVER_PENALTY")


def plot_equity_vs_penalty(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["TURNOVER_PENALTY"],
        df["Equity final"],
        marker="o",
    )

    plt.title("Equity final neta según penalización por turnover")
    plt.xlabel("Penalización por turnover")
    plt.ylabel("Equity final neta")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/de_turnover_penalty_equity.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_sharpe_vs_penalty(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["TURNOVER_PENALTY"],
        df["Sharpe"],
        marker="o",
    )

    plt.title("Sharpe neto según penalización por turnover")
    plt.xlabel("Penalización por turnover")
    plt.ylabel("Sharpe neto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/de_turnover_penalty_sharpe.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


def plot_turnover_vs_penalty(df: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["TURNOVER_PENALTY"],
        df["Turnover total"],
        marker="o",
    )

    plt.title("Turnover total según penalización por turnover")
    plt.xlabel("Penalización por turnover")
    plt.ylabel("Turnover total")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/de_turnover_penalty_turnover.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Guardado: {output_path}")


if __name__ == "__main__":
    df = load_data()

    plot_equity_vs_penalty(df)
    plot_sharpe_vs_penalty(df)
    plot_turnover_vs_penalty(df)

    print("\nOK - Gráficas de penalización por turnover generadas.")