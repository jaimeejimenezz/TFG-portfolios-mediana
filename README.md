# Optimización de carteras basadas en la mediana con control de riesgo

Este repositorio contiene el código desarrollado para el TFG "Optimización de carteras basadas en la mediana con control de riesgo mediante metaheurísticos y proyección de restricciones".

## Objetivo

El objetivo es implementar y evaluar un método de optimización de carteras basado en la maximización de la mediana de los rendimientos, incorporando control de riesgo mediante MAD, restricciones realistas de pesos y resolución mediante Differential Evolution.

## Métodos comparados

- Cartera equiponderada 1/N.
- Markowitz media-varianza.
- Mediana+MAD mediante búsqueda aleatoria.
- Mediana+MAD mediante Differential Evolution.
- Mediana+MAD Differential Evolution con penalización explícita por turnover.

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt