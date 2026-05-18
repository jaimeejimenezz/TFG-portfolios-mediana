import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "src/compare_random_vs_de.py",
    "src/compare_transaction_costs_with_de.py",
    "src/sensitivity_transaction_costs_with_de.py",
    "src/compare_de_turnover_penalty.py",
    "src/plot_transaction_costs_with_de.py",
    "src/plot_de_turnover_penalty.py",
]

def run_script(script_path: str):
    print("\n" + "=" * 80)
    print(f"Ejecutando: {script_path}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Error ejecutando {script_path}")

def main():
    for script in SCRIPTS:
        run_script(script)

    print("\n" + "=" * 80)
    print("OK - Resultados finales reproducidos correctamente")
    print("=" * 80)

if __name__ == "__main__":
    main()