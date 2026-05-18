import numpy as np
import pandas as pd
from scipy.optimize import minimize

RETURNS_PATH = "data/processed/returns.csv"

LOOKBACK_DAYS = 252     # ventana histórica
REBALANCE_DAYS = 21     # ~ mensual
LAMBDA = 10.0           # controla cuánto penalizas el riesgo (ajustable)
CAP = 0.10              # tope por activo - ej: 0.10 si quieres tope por activo, o None

TRADING_DAYS = 252

def load_returns(path: str) -> pd.DataFrame:
    r = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return r

def markowitz_weights(returns_window: pd.DataFrame, lam: float = 10.0, cap=None) -> np.ndarray:
    """
    Maximizamos utilidad media-varianza:  mu'w - lam * w' Σ w
    con w>=0, sum(w)=1 y opcional w<=cap
    """
    X = returns_window.dropna() #limpia filas con faltantes
    n = X.shape[1] #calcula numero de activos

    mu = X.mean().values               # media diaria
    Sigma = X.cov().values             # cov diaria

    # Regularización diagonal para mejorar la estabilidad numérica de la optimización.
    Sigma = Sigma + 1e-8 * np.eye(n) 

    def objective(w):
        # FUNCION OBJETIVO - minimizamos el negativo de la utilidad
        return -(mu @ w) + lam * (w @ Sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}] #todo el capital se reparte completo entre los activos(No sobra ni falta dinero)
    if cap is None: #Define cotas para cada peso(no ventas en corto)
        bounds = [(0.0, 1.0) for _ in range(n)]
    else:
        bounds = [(0.0, float(cap)) for _ in range(n)]

    w0 = np.ones(n) / n #INICIALIZACION - Punto inicial del optimizador: parte de pesos iguales

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons) #Llama al optimizador numérico ya que lo resuelve mejor

    if (not res.success) or np.any(np.isnan(res.x)): #por si la optimización falla 
        # fallback seguro
        return w0

    w = res.x
    # limpieza: por si hay mini errores numéricos
    w[w < 0] = 0.0
    s = w.sum()
    if s == 0:
        return w0
    return w / s


def run_backtest_markowitz(returns: pd.DataFrame, lookback_days=252, rebalance_days=21, lam=10.0, cap=None):
    tickers = list(returns.columns)
    n = len(tickers)

    port_rets = []
    dates = []

    weights_history = []

    w = np.ones(n) / n  # inicial: empezamos en 1/N porque hasta que no haya suficiente histórico no podemos optimizar con una ventana de 252 días.
    
    for i, dt in enumerate(returns.index):
        # rebalanceamos si toca y si hay ventana suficiente - cada 21 días, si ya tienes al menos 252 días previos
        if i % rebalance_days == 0 and i >= lookback_days:
            window = returns.iloc[i - lookback_days:i] #cogemos la ventana histórica de un año
            w = markowitz_weights(window, lam=lam, cap=cap) #calculamos pesos óptimos con Markowitz
            weights_history.append((dt, w.copy())) #guardamos esos pesos con la fecha del rebalanceo

        r_t = returns.iloc[i].values #toma los rendimientos de todos los activos en el día
        rp_t = float(np.dot(w, r_t)) #calcula el rendimiento de la cartera como producto escalar

        dates.append(dt)
        port_rets.append(rp_t)

    bt = pd.DataFrame({"portfolio_return": port_rets}, index=pd.DatetimeIndex(dates)) #retorno diario
    bt["equity"] = (1 + bt["portfolio_return"]).cumprod() #curva acumulada

    # Guardamos qué pesos decidió Markowitz en cada fecha de rebalanceo
    wh = pd.DataFrame(
        {"date": [d for d, _ in weights_history]}
    )
    if len(weights_history) > 0:
        W = np.vstack([w for _, w in weights_history])
        wh = pd.DataFrame(W, columns=tickers, index=[d for d, _ in weights_history])
        wh.index.name = "rebalance_date"

    return bt, wh #backtest con rentabilidades y equity - historial de pesos


if __name__ == "__main__":
    rets = load_returns(RETURNS_PATH)
    bt, w_hist = run_backtest_markowitz(
        rets,
        lookback_days=LOOKBACK_DAYS,
        rebalance_days=REBALANCE_DAYS,
        lam=LAMBDA,
        cap=CAP
    )

    bt.to_csv("data/processed/backtest_markowitz.csv")
    if len(w_hist) > 0:
        w_hist.to_csv("data/processed/weights_markowitz.csv")

    print("OK - Backtest Markowitz generado")
    print(bt.head(3))
    print(bt.tail(3))
    print("Equity final:", bt["equity"].iloc[-1])
    if len(w_hist) > 0:
        print("\nÚltimos pesos (último rebalanceo):")
        print((w_hist.iloc[-1].sort_values(ascending=False)).head(6))