"""Rutas, constantes y utilidades compartidas por los cuadernos de la Tarea 1 (Lending Club)."""
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[1]
CSV_CRUDO = RAIZ / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "data" / "resultados"
# Carpeta terminada en .nosync: iCloud no la sincroniza (modelos serializados pesados)
MODELOS = RAIZ / "data" / "modelos.nosync"
FIGURAS = RAIZ / "book" / "figuras"

# Prueba de humo del código (nunca en el libro): con LC_FRACCION_PRUEBA=0.02 los cuadernos de
# modelado trabajan con una fracción de cada conjunto y escriben en una carpeta aparte.
FRACCION_PRUEBA = float(os.environ.get("LC_FRACCION_PRUEBA", "0") or 0)
if FRACCION_PRUEBA:
    RESULTADOS = RAIZ / "data" / "prueba.nosync" / "resultados"
    MODELOS = RAIZ / "data" / "prueba.nosync" / "modelos"
    FIGURAS = RAIZ / "data" / "prueba.nosync" / "figuras"
for _d in (INTERIM, PROCESSED, RESULTADOS, MODELOS, FIGURAS):
    _d.mkdir(parents=True, exist_ok=True)

PRESTAMOS = INTERIM / "prestamos_cerrados.parquet"   # 1.348.059 préstamos cerrados x 154 columnas
MODELO = PROCESSED / "base_modelo.parquet"           # id, default y variables del modelo (sin codificar)
PARTICION = PROCESSED / "particion.parquet"          # id, split (train/test)

SEMILLA = 42
OBJETIVO = "default"
PYTHON_VENV = str(Path.home() / ".venvs" / "ml-lending" / "bin" / "python")

# Variables del EDA (sección 9.10.4.1): originación del préstamo, sin información posterior
NUM_EDA = [
    "loan_amnt", "installment", "int_rate", "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "emp_length", "term", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "acc_open_past_24mths", "bc_util",
    "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "mths_since_recent_inq",
    "percent_bc_gt_75", "mths_since_last_delinq", "antig_credito_meses",
]
CAT_EDA = [
    "grade", "purpose", "home_ownership", "addr_state", "verification_status",
    "application_type", "initial_list_status",
]

# Variables que existen solo después de la originación (fuga de información): nunca se modelan
FUGA = [
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "pymnt_plan", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term", "hardship_flag",
] + [c for c in ["hardship_type", "hardship_reason", "hardship_status", "deferral_term",
                 "hardship_amount", "hardship_start_date", "hardship_end_date",
                 "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
                 "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
                 "hardship_last_payment_amount"]]

PALETA = {0: "#4C72B0", 1: "#C44E52"}
ETIQUETAS = {0: "Fully Paid (0)", 1: "Charged Off (1)"}


def estilo():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
                         "axes.titleweight": "bold", "axes.titlesize": 11})


def emp_length_a_numero(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '1 year' -> 1, ..., '10+ years' -> 10; NaN se conserva."""
    return (s.str.replace("< 1", "0", regex=False).str.extract(r"(\d+)")[0].astype("float64"))


def cargar_prestamos(columnas=None) -> pd.DataFrame:
    """Lee el parquet intermedio (préstamos cerrados, todas las columnas originales + default)."""
    return pd.read_parquet(PRESTAMOS, columns=columnas)


def hardware() -> dict:
    """Describe el equipo para que los tiempos sean interpretables."""
    info = {"sistema": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "nucleos_logicos": os.cpu_count()}
    if platform.system() == "Darwin":
        q = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        info["cpu"] = q("machdep.cpu.brand_string")
        info["nucleos_rendimiento"] = q("hw.perflevel0.physicalcpu")
        info["nucleos_eficiencia"] = q("hw.perflevel1.physicalcpu")
        info["ram_gb"] = round(int(q("hw.memsize")) / 2**30, 1)
    return info


@contextmanager
def cronometro(registro: dict, clave: str):
    t0 = time.time()
    yield
    registro[clave] = time.time() - t0


def guardar_json(obj, ruta: Path):
    ruta.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def cargar_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def guardar_fig(fig, nombre: str):
    fig.savefig(FIGURAS / f"{nombre}.png")
