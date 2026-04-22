import pandas as pd
import yfinance as yf
from config import TICKERS, START, END

def download_prices(tickers, start, end):
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # ajusta por dividendos/splits
        progress=False,
        group_by="column",
    )

    # Con varios tickers suele venir MultiIndex: (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Preferimos "Close" (ya ajustado)
        if ("Close" in df.columns.get_level_values(0)):
            prices = df["Close"].copy()
        else:
            # fallback: a veces aparece como "Adj Close" o "Price"
            prices = df.xs(df.columns.levels[0][0], axis=1, level=0).copy()
    else:
        prices = df[["Close"]].copy()
        prices.columns = tickers

    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    return prices

def prices_to_returns(prices):
    # rendimientos simples diarios
    returns = prices.pct_change().dropna(how="all")
    return returns

if __name__ == "__main__":
    prices = download_prices(TICKERS, START, END)
    returns = prices_to_returns(prices)

    prices.to_csv("data/processed/prices.csv")
    returns.to_csv("data/processed/returns.csv")

    print("OK - Dataset generado")
    print("prices shape:", prices.shape, "| from:", prices.index.min().date(), "to:", prices.index.max().date())
    print("returns shape:", returns.shape)
    print("\nMissing % por ticker (top 10):")
    print((prices.isna().mean() * 100).round(2).sort_values(ascending=False).head(10))