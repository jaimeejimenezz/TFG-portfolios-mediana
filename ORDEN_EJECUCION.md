# Orden de ejecución del proyecto

Este documento indica el orden recomendado para ejecutar el proyecto del TFG:

**Optimización de carteras basadas en la mediana con control de riesgo mediante metaheurísticos y proyección de restricciones**

El objetivo es facilitar la reproducción de los resultados principales de la memoria, incluyendo:

- preparación de datos;
- backtests de los métodos comparados;
- comparación entre baselines y métodos propuestos;
- aplicación de costes de transacción;
- análisis de sensibilidad;
- penalización explícita por turnover;
- generación de tablas y figuras finales.

---

## 1. Requisitos previos

El proyecto está desarrollado en Python y se recomienda ejecutarlo desde un entorno virtual.

### 1.1 Crear entorno virtual

Desde la carpeta raíz del proyecto:

python -m venv .venv

### 1.2 Activar entorno virtual

En Windows PowerShell:

.venv\Scripts\activate

### 1.3 Instalar dependencias

pip install -r requirements.txt

En caso de no disponer todavía de `requirements.txt`, las dependencias mínimas son:

pip install numpy pandas scipy matplotlib yfinance

---

## 2. Estructura esperada del proyecto

La estructura principal del repositorio debe ser similar a la siguiente:

tfg-portfolio/
│
├── README.md
├── requirements.txt
├── ORDEN_EJECUCION.md
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
│   └── run_reproduce_results.py
│
├── data/
│   └── processed/
│
├── reports/
    └── figures/

---

## 3. Opción recomendada: reproducir resultados finales

Esta es la opción recomendada para revisión rápida del proyecto.

Utiliza los ficheros ya generados en `data/processed/` y reproduce las tablas y figuras principales de la memoria sin volver a ejecutar todos los backtests pesados.

python src\run_reproduce_results.py

Este script ejecuta los módulos finales de comparación y generación de figuras:

1. compare_random_vs_de.py
2. compare_transaction_costs_with_de.py
3. sensitivity_transaction_costs_with_de.py
4. compare_de_turnover_penalty.py
5. plot_transaction_costs_with_de.py
6. plot_de_turnover_penalty.py

Esta opción es útil para comprobar rápidamente que los resultados finales de la memoria pueden regenerarse a partir de los CSV ya incluidos en el proyecto.

---

## 4. Opción completa: ejecución desde cero

Esta opción regenera el proyecto desde la descarga de datos hasta las tablas y figuras finales.

Puede tardar más tiempo, especialmente en los scripts que usan **Differential Evolution**, ya que ejecutan optimizaciones en múltiples fechas de rebalanceo.

Ejecutar los scripts en el siguiente orden:

python src\make_dataset.py
python src\backtest_equal_weight.py
python src\backtest_markowitz.py
python src\backtest_median_mad_random.py
python src\backtest_median_mad_de.py
python src\compare_3_methods.py
python src\compare_cap_effect.py
python src\compare_random_vs_de.py
python src\compare_transaction_costs_with_de.py
python src\sensitivity_transaction_costs_with_de.py
python src\plot_transaction_costs_with_de.py
python src\backtest_median_mad_de_turnover.py
python src\compare_de_turnover_penalty.py
python src\plot_de_turnover_penalty.py

---

## 5. Descripción de cada bloque de ejecución

### 5.1 Preparación de datos

python src\make_dataset.py

Descarga los precios ajustados de los ETFs definidos en `config.py`, calcula los rendimientos diarios y genera los ficheros base del proyecto.

Salidas principales:

data/processed/prices.csv
data/processed/returns.csv

---

### 5.2 Backtest de la cartera equiponderada 1/N

python src\backtest_equal_weight.py

Ejecuta el baseline equiponderado, asignando el mismo peso a todos los activos.

Salidas principales:

data/processed/backtest_equal_weight.csv

---

### 5.3 Backtest de Markowitz

python src\backtest_markowitz.py

Ejecuta el modelo media-varianza de Markowitz con restricciones de pesos.

Salidas principales:

data/processed/backtest_markowitz.csv
data/processed/weights_markowitz.csv

---

### 5.4 Backtest Mediana+MAD mediante búsqueda aleatoria

python src\backtest_median_mad_random.py

Ejecuta el método propuesto inicial basado en maximizar la mediana de los rendimientos de cartera bajo control de riesgo MAD, usando búsqueda aleatoria factible.

Salidas principales:

data/processed/backtest_median_mad_random.csv
data/processed/weights_median_mad_random.csv

---

### 5.5 Backtest Mediana+MAD mediante Differential Evolution

python src\backtest_median_mad_de.py

Ejecuta la versión Mediana+MAD usando Differential Evolution como metaheurística principal.

Salidas principales:

data/processed/backtest_median_mad_de.csv
data/processed/weights_median_mad_de.csv
data/processed/optimization_stats_median_mad_de.csv

---

### 5.6 Comparación inicial de métodos

python src\compare_3_methods.py

Compara los métodos principales iniciales:

1/N
Markowitz
Mediana+MAD Random

Salidas principales:

data/processed/metrics_3_methods.csv

---

### 5.7 Comparación del efecto de CAP10

python src\compare_cap_effect.py

Resume el efecto de introducir una restricción de peso máximo del 10% por activo.

Salidas principales:

data/processed/metrics_cap_effect.csv

---

### 5.8 Comparación Random Search vs Differential Evolution

python src\compare_random_vs_de.py

Compara los métodos principales con restricción CAP10:

1/N
Markowitz CAP10
Mediana+MAD Random CAP10
Mediana+MAD DE CAP10

Salidas principales:

data/processed/metrics_random_vs_de.csv

Esta salida se corresponde con la comparación bruta entre búsqueda aleatoria y Differential Evolution.

---

### 5.9 Comparación con costes de transacción incluyendo Differential Evolution

python src\compare_transaction_costs_with_de.py

Aplica costes de transacción de 10 puntos básicos a los métodos optimizados y compara los resultados netos.

Métodos comparados:

1/N
Markowitz CAP10 + costes
Mediana+MAD Random CAP10 + costes
Mediana+MAD DE CAP10 + costes

Salidas principales:

data/processed/metrics_transaction_costs_with_de_10bps.csv

---

### 5.10 Sensibilidad a costes de transacción

python src\sensitivity_transaction_costs_with_de.py

Evalúa la sensibilidad de los métodos ante distintos niveles de costes de transacción:

0 bps
5 bps
10 bps
20 bps
50 bps

Salidas principales:

data/processed/sensitivity_transaction_costs_with_de.csv

---

### 5.11 Figuras de costes y sensibilidad

python src\plot_transaction_costs_with_de.py

Genera las figuras asociadas al análisis de costes y sensibilidad.

Salidas principales:

reports/figures/equity_cap10_costs_10bps_with_de.png
reports/figures/drawdown_cap10_costs_10bps_with_de.png
reports/figures/sensitivity_equity_with_de.png
reports/figures/sensitivity_sharpe_with_de.png

---

### 5.12 Backtest Mediana+MAD DE con penalización explícita por turnover

python src\backtest_median_mad_de_turnover.py

Ejecuta la extensión final del método Mediana+MAD DE, incorporando una penalización explícita por turnover dentro de la función objetivo.

Se evalúan distintos valores de penalización:

η = 0.0005
η = 0.0010
η = 0.0020

Salidas principales:

data/processed/backtest_median_mad_de_turnover_tp_0p0005.csv
data/processed/backtest_median_mad_de_turnover_tp_0p001.csv
data/processed/backtest_median_mad_de_turnover_tp_0p002.csv

data/processed/weights_median_mad_de_turnover_tp_0p0005.csv
data/processed/weights_median_mad_de_turnover_tp_0p001.csv
data/processed/weights_median_mad_de_turnover_tp_0p002.csv

data/processed/optimization_stats_median_mad_de_turnover_tp_0p0005.csv
data/processed/optimization_stats_median_mad_de_turnover_tp_0p001.csv
data/processed/optimization_stats_median_mad_de_turnover_tp_0p002.csv

---

### 5.13 Comparación final con penalización por turnover

python src\compare_de_turnover_penalty.py

Compara la versión Mediana+MAD DE base con las variantes que incorporan penalización explícita por turnover, aplicando costes de transacción de 10 puntos básicos.

Métodos comparados:

1/N
Markowitz CAP10 + costes
Mediana+MAD Random CAP10 + costes
Mediana+MAD DE CAP10 + costes
Mediana+MAD DE + η = 0.0005
Mediana+MAD DE + η = 0.0010
Mediana+MAD DE + η = 0.0020

Salidas principales:

data/processed/metrics_de_turnover_penalty_10bps.csv

La configuración final seleccionada en la memoria es:

Mediana+MAD DE con penalización explícita por turnover η = 0.0010

---

### 5.14 Figuras de penalización por turnover

python src\plot_de_turnover_penalty.py

Genera las figuras finales del análisis de penalización por turnover.

Salidas principales:

reports/figures/de_turnover_penalty_equity.png
reports/figures/de_turnover_penalty_sharpe.png
reports/figures/de_turnover_penalty_turnover.png

---

## 6. Ficheros principales de resultados

Los ficheros más importantes para comprobar las tablas de la memoria son:

data/processed/metrics_random_vs_de.csv
data/processed/metrics_transaction_costs_with_de_10bps.csv
data/processed/sensitivity_transaction_costs_with_de.csv
data/processed/metrics_de_turnover_penalty_10bps.csv

Estos ficheros contienen las métricas utilizadas en las comparativas principales del trabajo.

---

## 7. Figuras principales generadas

Las figuras finales utilizadas en la memoria se encuentran en:

reports/figures/equity_cap10_costs_10bps_with_de.png
reports/figures/drawdown_cap10_costs_10bps_with_de.png
reports/figures/sensitivity_equity_with_de.png
reports/figures/sensitivity_sharpe_with_de.png
reports/figures/de_turnover_penalty_equity.png
reports/figures/de_turnover_penalty_sharpe.png
reports/figures/de_turnover_penalty_turnover.png

---

## 8. Comprobación rápida de que el código compila

Antes de entregar el proyecto, se recomienda ejecutar:

python -m compileall src

Este comando comprueba que todos los scripts de la carpeta `src/` pueden compilarse correctamente.

---

## 9. Comprobación rápida de resultados finales

Después de comprobar que el código compila, ejecutar:

python src\run_reproduce_results.py

Si todo está correcto, deben regenerarse las tablas y figuras principales sin errores.

---

## 10. Notas sobre tiempos de ejecución

La ejecución completa desde cero puede tardar más tiempo por los scripts basados en Differential Evolution:

backtest_median_mad_de.py
backtest_median_mad_de_turnover.py

Por ese motivo, para una revisión rápida se recomienda usar:

python src\run_reproduce_results.py

Esta opción parte de los CSV ya generados y reproduce los resultados finales de la memoria.

---

## 11. Nota sobre reproducibilidad

Los experimentos utilizan una semilla fija para reducir la variabilidad de los métodos estocásticos. Además, los resultados intermedios se almacenan en formato CSV dentro de `data/processed/`, lo que permite auditar las tablas y figuras generadas.

La descarga de datos depende de `yfinance`, por lo que, si se vuelve a ejecutar el proyecto desde cero en el futuro, podrían existir ligeras diferencias si la fuente actualiza datos históricos, dividendos, ajustes o disponibilidad de activos.

Para reproducir exactamente la versión entregada, se recomienda utilizar los CSV incluidos en `data/processed/`.

---

## 12. Orden recomendado para revisión del tutor

Para revisar el proyecto de forma eficiente:

1. Leer `README.md`.
2. Consultar este documento, `ORDEN_EJECUCION.md`.
3. Ejecutar la reproducción rápida:

python src\run_reproduce_results.py

4. Revisar los resultados principales en:

data/processed/metrics_de_turnover_penalty_10bps.csv

5. Revisar las figuras finales en:

reports/figures/

---

## 13. Resumen final del pipeline

El flujo completo del proyecto es:

Datos de precios
      ↓
Cálculo de rendimientos
      ↓
Backtests de baselines
      ↓
Backtest Mediana+MAD Random
      ↓
Backtest Mediana+MAD Differential Evolution
      ↓
Comparación bruta con CAP10
      ↓
Costes de transacción
      ↓
Sensibilidad a costes
      ↓
Penalización explícita por turnover
      ↓
Comparación final
      ↓
Tablas y figuras de la memoria

La versión final considerada más equilibrada es:

Mediana+MAD Differential Evolution
+ restricción CAP10
+ control de riesgo MAD
+ penalización explícita por turnover η = 0.0010
