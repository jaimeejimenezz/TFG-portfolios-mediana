import pandas as pd

data = [
    {
        "Método": "1/N",
        "Restricción": "Sin cap / cap no afecta",
        "Equity final": 2.497,
        "CAGR": 0.0946,
        "Vol anual": 0.1084,
        "Sharpe": 0.888,
        "MaxDD": -0.2218,
    },
    {
        "Método": "Markowitz",
        "Restricción": "Sin cap",
        "Equity final": 3.009,
        "CAGR": 0.1149,
        "Vol anual": 0.1097,
        "Sharpe": 1.047,
        "MaxDD": -0.1674,
    },
    {
        "Método": "Markowitz",
        "Restricción": "Cap 10%",
        "Equity final": 2.381,
        "CAGR": 0.0894,
        "Vol anual": 0.1013,
        "Sharpe": 0.896,
        "MaxDD": -0.2162,
    },
    {
        "Método": "Mediana+MAD",
        "Restricción": "Sin cap",
        "Equity final": 3.587,
        "CAGR": 0.1344,
        "Vol anual": 0.1366,
        "Sharpe": 0.992,
        "MaxDD": -0.2458,
    },
    {
        "Método": "Mediana+MAD",
        "Restricción": "Cap 10%",
        "Equity final": 2.596,
        "CAGR": 0.0988,
        "Vol anual": 0.1071,
        "Sharpe": 0.933,
        "MaxDD": -0.2103,
    },
]

df = pd.DataFrame(data)

pretty = df.copy()
pretty["Equity final"] = pretty["Equity final"].round(3)
pretty["CAGR"] = (pretty["CAGR"] * 100).round(2).astype(str) + "%"
pretty["Vol anual"] = (pretty["Vol anual"] * 100).round(2).astype(str) + "%"
pretty["MaxDD"] = (pretty["MaxDD"] * 100).round(2).astype(str) + "%"
pretty["Sharpe"] = pretty["Sharpe"].round(3)

print("\n=== Efecto de la restricción CAP 10% ===")
print(pretty)

pretty.to_csv("data/processed/metrics_cap_effect.csv", index=False)
print("\nGuardado: data/processed/metrics_cap_effect.csv")