# Optimización de carteras basadas en la mediana con control de riesgo

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado:

**Optimización de carteras basadas en la mediana con control de riesgo mediante metaheurísticos y proyección de restricciones**

El proyecto implementa un pipeline reproducible para la optimización y evaluación de carteras financieras bajo restricciones realistas, utilizando backtesting *walk-forward*, control de riesgo mediante MAD, metaheurísticos y análisis de costes de transacción.

## Descripción general

El objetivo principal del proyecto es implementar y evaluar una estrategia de optimización de carteras basada en la maximización de la mediana de los rendimientos de cartera, incorporando:

* control de riesgo mediante MAD respecto a la mediana;
* restricciones de inversión realistas;
* proyección al simplex con tope máximo por activo;
* resolución mediante búsqueda aleatoria y Differential Evolution;
* costes de transacción proporcionales al turnover;
* penalización explícita por turnover dentro de la función objetivo;
* comparación frente a estrategias de referencia.

La versión final considerada más equilibrada en la memoria es:

**Mediana+MAD Differential Evolution con restricción CAP10 y penalización explícita por turnover η = 0.0010.**

## Métodos comparados

El proyecto compara las siguientes estrategias:

* Cartera equiponderada 1/N.
* Modelo media-varianza de Markowitz.
* Mediana+MAD mediante búsqueda aleatoria factible.
* Mediana+MAD mediante Differential Evolution.
* Mediana+MAD Differential Evolution con penalización explícita por turnover.

## Estructura del repositorio

La estructura principal del proyecto es la siguiente:

TFG_Jaime_Jimenez_Santos_Codigo/
│
├── README.md
├── ORDEN_EJECUCION.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── make_dataset.py
│   ├── backtest_equal_weight.py
│   ├── backtest_markowitz.py
│   ├── backtest_median_mad_random.py
│   ├── backtest_median_mad_de.py
│   ├── backtest_median_mad_de_turnover.py
│   ├── compare_3_methods.py
│   ├── compare_cap_effect.py
│   ├── compare_random_vs_de.py
│   ├── compare_transaction_costs_with_de.py
│   ├── sensitivity_transaction_costs_with_de.py
│   ├── plot_transaction_costs_with_de.py
│   ├── compare_de_turnover_penalty.py
│   ├── plot_de_turnover_penalty.py
│   ├── compare_subperiod_robustness.py
│   └── run_reproduce_results.py
│
├── data/
│   └── processed/
│
└── reports/
    └── figures/

## Requisitos

El proyecto está desarrollado en Python. Se recomienda utilizar un entorno virtual para evitar conflictos con otras instalaciones.

Las dependencias principales son:

* NumPy
* Pandas
* SciPy
* Matplotlib
* yfinance

Todas las dependencias necesarias se encuentran en el archivo:

requirements.txt

## Instalación

Desde la carpeta raíz del proyecto, crear y activar un entorno virtual:

python -m venv .venv
.venv\Scripts\activate

Instalar las dependencias:

pip install -r requirements.txt

En caso de usar Linux o macOS, la activación del entorno virtual sería:

source .venv/bin/activate

## Comprobación del código

Antes de ejecutar los experimentos, puede comprobarse que los scripts compilan correctamente con:

python -m compileall src

Este comando revisa que los archivos Python de la carpeta `src/` no contienen errores básicos de sintaxis.

## Ejecución rápida recomendada

Para reproducir las tablas y figuras principales de la memoria a partir de los ficheros ya generados en `data/processed/`, ejecutar:

python src\run_reproduce_results.py

Esta opción es la recomendada para una revisión rápida del proyecto, ya que evita repetir los backtests más costosos y regenera los resultados finales a partir de los CSV incluidos.

## Ejecución completa desde cero

Para regenerar todo el pipeline desde la descarga de datos hasta las comparaciones finales, consultar el documento:

ORDEN_EJECUCION.md

El flujo completo incluye:

1. descarga y preparación de datos;
2. cálculo de rendimientos;
3. ejecución de baselines;
4. ejecución de Mediana+MAD mediante búsqueda aleatoria;
5. ejecución de Mediana+MAD mediante Differential Evolution;
6. aplicación de costes de transacción;
7. análisis de sensibilidad;
8. penalización explícita por turnover;
9. análisis por subperiodos;
10. generación de tablas y figuras finales.

Algunos scripts, especialmente los basados en Differential Evolution, pueden tardar más tiempo en ejecutarse porque realizan optimizaciones en múltiples fechas de rebalanceo.

## Principales resultados generados

Los ficheros principales de resultados se almacenan en:

data/processed/

Entre los ficheros más relevantes se encuentran:

metrics_random_vs_de.csv
metrics_transaction_costs_with_de_10bps.csv
sensitivity_transaction_costs_with_de.csv
metrics_de_turnover_penalty_10bps.csv
metrics_subperiod_robustness_10bps.csv

Estos archivos contienen las métricas utilizadas para las comparaciones principales de la memoria.

## Figuras generadas

Las figuras principales se almacenan en:

reports/figures/

Algunas de las figuras finales son:

equity_cap10_costs_10bps_with_de.png
drawdown_cap10_costs_10bps_with_de.png
sensitivity_equity_with_de.png
sensitivity_sharpe_with_de.png
de_turnover_penalty_equity.png
de_turnover_penalty_sharpe.png
de_turnover_penalty_turnover.png

## Reproducibilidad

El proyecto utiliza semillas fijas para reducir la variabilidad de los métodos estocásticos y facilitar la reproducción de los resultados.

No obstante, debe tenerse en cuenta que la descarga de datos depende de `yfinance`. Si se vuelve a ejecutar el proyecto desde cero en el futuro, podrían aparecer pequeñas diferencias si la fuente actualiza datos históricos, dividendos, ajustes o disponibilidad de activos.

Para reproducir exactamente la versión entregada, se recomienda utilizar los ficheros CSV incluidos en:

data/processed/

## Notas sobre el backtesting

El backtesting se realiza mediante una validación temporal tipo *walk-forward*. En cada fecha de rebalanceo se utiliza únicamente información pasada para calcular los pesos de la cartera, evitando sesgo de anticipación.

Los parámetros principales del experimento son:

Ventana de estimación: 252 sesiones
Frecuencia de rebalanceo: 21 sesiones
Capital inicial: 1 unidad monetaria
Tope máximo por activo: 10%
Coste principal de transacción: 10 puntos básicos
Semilla aleatoria: 42

## Autor

**Jaime Jiménez Santos**

Grado en Ingeniería Informática
Escuela Técnica Superior de Ingeniería Informática
Universidad de Málaga

## Tutor

**Francisco de Asís Fernández Navarro**

Departamento de Lenguajes y Ciencias de la Computación
Universidad de Málaga
