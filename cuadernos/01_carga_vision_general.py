# %% [markdown]
# # 1. Carga inicial y visión general
#
# Este capítulo corresponde a la sección 9.10.4.1.1 del enunciado. Se lee el archivo
# `accepted_2007_to_2018Q4.csv.gz` completo (Kaggle, *wordsforthewise/lending-club*), se verifica
# la lectura, se describe su estructura y se define la población de préstamos sobre la que se
# construye la variable objetivo `default`. La lectura se hace con pandas y, para comparar tiempos,
# también con Spark.

# %%
import os
import sys
import time
import warnings

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as u

u.estilo()
pd.set_option("display.max_columns", 40, "display.width", 180, "display.max_colwidth", 40)
print("Equipo utilizado:")
for k, v in u.hardware().items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 1.1 Lectura del CSV completo con pandas
#
# Se lee el archivo comprimido completo, sin muestreo ni `nrows`.

# %%
t0 = time.time()
crudo = pd.read_csv(u.CSV_CRUDO, low_memory=False)
t_pandas = time.time() - t0
print(f"Tamaño del archivo comprimido: {u.CSV_CRUDO.stat().st_size / 2**20:.0f} MB")
print(f"Tiempo de lectura con pandas: {t_pandas:.1f} s")
print(f"Dimensión del archivo: {crudo.shape[0]:,} filas x {crudo.shape[1]} columnas")
print(f"Memoria en pandas: {crudo.memory_usage(deep=True).sum() / 2**30:.2f} GB")

# %% [markdown]
# ## 1.2 Primeras y últimas filas

# %%
crudo.head()

# %%
crudo.tail()

# %% [markdown]
# Las primeras filas se leen correctamente. Las dos últimas no son préstamos: son líneas de
# totales que Lending Club añadió al final del archivo ("Total amount funded in policy code 1/2"),
# con todas las demás columnas vacías. A ellas se suman otras filas con `id` no numérico dentro del
# archivo. Se identifican por tener `loan_status` vacío.

# %%
id_num = pd.to_numeric(crudo["id"], errors="coerce")
print("Filas con id no numérico:", id_num.isna().sum())
print("Filas con loan_status vacío:", crudo["loan_status"].isna().sum())
print("Coinciden:", (id_num.isna() == crudo["loan_status"].isna()).all())

# %% [markdown]
# ## 1.3 Estado del préstamo y definición de la población
#
# La variable objetivo se construye a partir de `loan_status`. El enunciado fija
# 0 = *Fully Paid* y 1 = *Charged Off*.

# %%
estados = crudo["loan_status"].value_counts(dropna=False).rename("n").to_frame()
estados["%"] = (100 * estados["n"] / len(crudo)).round(2)
estados

# %% [markdown]
# Solo los préstamos cerrados tienen un desenlace observado. Los estados *Current*, *In Grace
# Period*, *Late* y *Default* corresponden a préstamos que seguían vigentes en la fecha de
# extracción de los datos (primer trimestre de 2019, según las fechas más recientes de
# `last_credit_pull_d`; 2018Q4 es el último trimestre de emisión incluido):
# asignarles 0, como haría literalmente `1 if x == "Charged Off" else 0`, los contaría como pagados
# sin saber si lo serán. Por eso la población de estudio son los préstamos con desenlace final:
#
# | Estado original | `default` |
# |---|---|
# | Fully Paid | 0 |
# | Does not meet the credit policy. Status:Fully Paid | 0 |
# | Charged Off | 1 |
# | Does not meet the credit policy. Status:Charged Off | 1 |
#
# Las dos categorías *Does not meet the credit policy* (2.749 préstamos) son préstamos
# cerrados con el mismo desenlace y se conservan. Esta definición no es un muestreo: se usan
# **todas** las filas con desenlace conocido, sin submuestrear ninguna clase.

# %%
mapa = {"Fully Paid": 0, "Does not meet the credit policy. Status:Fully Paid": 0,
        "Charged Off": 1, "Does not meet the credit policy. Status:Charged Off": 1}
df = crudo[crudo["loan_status"].isin(mapa)].copy()
df["default"] = df["loan_status"].map(mapa).astype("int8")
df["id"] = df["id"].astype("int64")
del crudo
print(f"Préstamos cerrados: {len(df):,} filas x {df.shape[1]} columnas")
print("id único:", df["id"].is_unique)

# %% [markdown]
# ## 1.4 Tipos de datos y valores nulos: `info()`

# %%
df.info(verbose=True, show_counts=True)

# %% [markdown]
# De las 152 columnas (151 originales y `default`), 115 son numéricas (incluidas `id` y `default`)
# y 37 de texto. Varias
# columnas de texto son en realidad numéricas u ordinales (`term`, `emp_length`), que se convierten
# a número en la sección 1.6, o fechas (`issue_d`, `earliest_cr_line`), a partir de las cuales esa
# sección deriva `antig_credito_meses` e `issue_year`. Hay tres grupos de columnas
# que no deben entrar al modelo:
#
# * **Identificadores y texto libre**: `id`, `member_id` (100 % vacío), `url`, `desc`, `title`,
#   `emp_title`, `zip_code`.
# * **Información posterior a la originación (fuga de información, 40 columnas listadas abajo)**:
#   montos efectivamente financiados (`funded_amnt`, `funded_amnt_inv`), pagos recibidos
#   (`total_pymnt`, `total_rec_prncp`, `recoveries`...), saldo pendiente, último y próximo pago,
#   última consulta y FICO más recientes, `pymnt_plan` y variables de *hardship* y *settlement*.
#   Solo se conocen durante la vida del
#   préstamo; un modelo que las use "predice" el default con información del propio default.
# * **Columnas de solicitudes conjuntas o de coprestatario**, vacías en más del 98 % de las filas.

# %%
print(f"Columnas con posible fuga de información ({len(u.FUGA)}):")
print(", ".join(u.FUGA))

# %% [markdown]
# ## 1.5 Estadísticas descriptivas: `describe()`

# %%
df.describe().T.round(2)

# %% [markdown]
# La tabla completa tiene 115 filas y pandas solo muestra las primeras y las últimas. Las variables
# que más interesan para el modelo se resumen aparte:

# %%
df[["loan_amnt", "annual_inc", "revol_bal", "tot_cur_bal", "dti", "out_prncp"]].describe().T.round(2)

# %% [markdown]
# El resumen confirma escalas muy distintas (de proporciones a saldos de millones de dólares) y
# colas largas en ingresos y saldos: la media de `annual_inc` (76.238 dólares) supera a su mediana
# (65.000) y el máximo llega a 11 millones; lo mismo ocurre con `revol_bal` (máximo de 2,9 millones)
# y `tot_cur_bal` (8 millones). `dti` tiene un mínimo de −1 y un máximo de 999, códigos que no son
# razones deuda/ingreso válidas. `out_prncp` vale 0 en todos los préstamos cerrados, como es de
# esperar. El capítulo 2 examina cada variable.

# %%
df.describe(include="object").T

# %% [markdown]
# ## 1.6 Variables derivadas
#
# * `term`: texto " 36 months" / " 60 months" a número de meses.
# * `emp_length`: "< 1 year" = 0, "1 year" = 1, ..., "10+ years" = 10; se conserva el faltante.
# * `antig_credito_meses`: meses entre la primera línea de crédito (`earliest_cr_line`) y la
#   emisión del préstamo (`issue_d`). Ambas fechas se conocen al originar el préstamo.
# * `issue_year`: año de emisión, solo para el análisis descriptivo.

# %%
df["term"] = df["term"].str.extract(r"(\d+)")[0].astype("float64")
df["emp_length"] = u.emp_length_a_numero(df["emp_length"])
emision = pd.to_datetime(df["issue_d"], format="%b-%Y")
primera = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
df["antig_credito_meses"] = ((emision.dt.year - primera.dt.year) * 12
                             + (emision.dt.month - primera.dt.month)).astype("float64")
df["issue_year"] = emision.dt.year.astype("int16")
df[["term", "emp_length", "antig_credito_meses", "issue_year"]].describe().T.round(2)

# %% [markdown]
# ## 1.7 Préstamos por año de emisión
#
# El conjunto cubre préstamos emitidos entre 2007 y 2018. Como solo se conservan préstamos cerrados,
# las cohortes recientes están incompletas: de un préstamo emitido en 2016-2018 solo se observa el
# desenlace si terminó antes de la extracción de los datos (primer trimestre de 2019), es decir,
# si entró en default pronto o si se prepagó. La
# tabla lo refleja: la tasa de default sube a 23 % en las cohortes de 2016 y 2017, donde pesan los
# defaults tempranos, y cae a 15,8 % en 2018, donde casi solo han terminado los préstamos
# prepagados. Este sesgo de selección por cohorte no afecta la comparación entre modelos, que se
# evalúan sobre la misma población, pero sí limita la extrapolación a préstamos futuros.

# %%
cohortes = df.groupby("issue_year").agg(prestamos=("default", "size"), tasa_default=("default", "mean"))
fig, ax1 = plt.subplots(figsize=(9, 3.6))
ax1.bar(cohortes.index, cohortes["prestamos"], color="#8DA0CB")
ax1.set_ylabel("Préstamos cerrados")
ax2 = ax1.twinx()
ax2.plot(cohortes.index, 100 * cohortes["tasa_default"], color="#C44E52", marker="o")
ax2.set_ylabel("Tasa de default (%)", color="#C44E52")
ax2.grid(False)
ax1.set_title("Préstamos cerrados y tasa de default por año de emisión")
u.guardar_fig(fig, "01_cohortes")
plt.show()
cohortes.assign(tasa_default=lambda d: (100 * d["tasa_default"]).round(2)).T

# %% [markdown]
# ## 1.8 Lectura con Spark (comparación de tiempos)
#
# Se lee el mismo CSV con Spark, con la configuración de sesión exigida en la sección 9.10.4.5.
# Todas las columnas se leen como texto (sin `inferSchema`, que obligaría a recorrer el archivo dos
# veces); el conteo de filas fuerza la lectura completa.

# %%
import spark_utils as su

spark = su.sesion_spark()
t0 = time.time()
sdf = su.leer_csv(spark)
n_spark = sdf.count()
t_spark = time.time() - t0
print(f"Tiempo de lectura + conteo con Spark: {t_spark:.1f} s ({n_spark:,} filas)")
print(f"Tiempo de lectura con pandas: {t_pandas:.1f} s")
print("Particiones del DataFrame leído:", sdf.rdd.getNumPartitions())

# %% [markdown]
# Spark obtiene el mismo número de filas que pandas (2.260.701). Su tiempo es menor porque el
# conteo no necesita convertir los 151 campos de cada fila en objetos de Python: el lector de CSV
# de Spark solo analiza los delimitadores. pandas, en cambio, construye un DataFrame de 6,3 GB en
# memoria. El archivo `.gz` no es divisible, así que Spark lo lee en **una sola partición** con un
# único núcleo: en esta etapa no aprovecha el paralelismo. La ventaja de Spark aparece cuando los
# datos ya están particionados y en caché (capítulo 7).

# %%
u.guardar_json({"pandas_lectura_s": t_pandas, "spark_lectura_conteo_s": t_spark,
                "filas_csv": int(n_spark)}, u.RESULTADOS / "tiempos_carga.json")
spark.stop()

# %% [markdown]
# ## 1.9 Almacenamiento intermedio
#
# Se guarda el conjunto de préstamos cerrados, con todas sus columnas, en formato Parquet para los
# capítulos de EDA. El modelado con Spark no usa este archivo: vuelve a leer el CSV original.

# %%
df.to_parquet(u.PRESTAMOS, index=False)
print(f"Guardado {u.PRESTAMOS.name}: {len(df):,} filas x {df.shape[1]} columnas, "
      f"{u.PRESTAMOS.stat().st_size / 2**20:.0f} MB")
print(f"Tasa global de default: {100 * df['default'].mean():.2f} %")
