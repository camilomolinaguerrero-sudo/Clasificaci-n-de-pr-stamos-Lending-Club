# %% [markdown]
# # 5. Preprocesamiento
#
# Sección 9.10.4.2 del enunciado. Este capítulo construye la **partición común** que comparten los
# dos entornos y aplica el mismo preprocesamiento con scikit-learn y con PySpark. Al final se
# comprueba, con estadísticos agregados, que ambos entornos producen la misma matriz de diseño.
#
# Las reglas de construcción de variables están en `src/features.py`, escritas dos veces (pandas y
# PySpark) una junto a la otra. Se distinguen dos tipos de transformación:
#
# | Tipo | Transformaciones | Cuándo se aplican |
# |---|---|---|
# | Deterministas, fila a fila | conversión de tipos, `dti` fuera de [0, 100] a faltante, `log1p` de `annual_inc`, `total_rev_hi_lim` y `avg_cur_bal`, indicadores de faltante, unificación de `ANY`/`NONE`/`OTHER` | antes de la partición: no aprenden nada de los datos |
# | Aprendidas | categorías con menos de 1 % agrupadas, mediana de imputación, media y desviación del escalado, categorías del one-hot | se ajustan **solo con entrenamiento** y se aplican a prueba |

# %%
import json
import sys
import time
import warnings

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import features as fe
import utils as u

pd.set_option("display.max_columns", 30, "display.width", 180, "display.max_rows", 100)
tiempos = {}

# %% [markdown]
# ## 5.1 Base de modelado
#
# Se parte del archivo intermedio del capítulo 1 (1.348.059 préstamos cerrados) y se construyen las
# 30 variables seleccionadas en el EDA: 20 numéricas, 4 indicadores binarios y 6 categóricas.

# %%
t0 = time.time()
base = fe.base_pandas(u.cargar_prestamos())
base.to_parquet(u.MODELO, index=False)
tiempos["base_pandas_s"] = time.time() - t0
print(f"Base de modelado: {base.shape[0]:,} filas x {base.shape[1]} columnas "
      f"({tiempos['base_pandas_s']:.1f} s)")
print("Numéricas:", fe.NUM)
print("Indicadores:", fe.IND)
print("Categóricas:", fe.CAT)
base.head()

# %% [markdown]
# ## 5.2 Partición común (80/20 estratificada)
#
# La prueba de DeLong compara AUC calculados sobre **las mismas observaciones**, así que la
# partición se define una sola vez, con semilla 42 y estratificada por `default`, y se guarda en
# `data/processed/particion.parquet` con las columnas `id` y `split`. scikit-learn y PySpark leen
# ese archivo; PySpark no usa `randomSplit`, que no reproduciría la partición.

# %%
id_tr, id_te = train_test_split(base["id"], test_size=0.2, stratify=base["default"],
                                random_state=u.SEMILLA)
particion = pd.concat([pd.DataFrame({"id": id_tr, "split": "train"}),
                       pd.DataFrame({"id": id_te, "split": "test"})]).sort_values("id")
particion.to_parquet(u.PARTICION, index=False)
resumen = (base.merge(particion, on="id").groupby("split")["default"]
           .agg(filas="size", defaults="sum", tasa_default="mean"))
resumen["tasa_default"] = (100 * resumen["tasa_default"]).round(3)
print("id único en la partición:", particion["id"].is_unique, "| filas:", len(particion))
resumen

# %% [markdown]
# La tasa de default es la misma en ambos conjuntos hasta el tercer decimal, como garantiza la
# estratificación. El conjunto de prueba tiene 269.612 préstamos, de los cuales 53.864 son
# defaults: sobre él se calculan todas las métricas y pruebas de los capítulos 6 a 9.

# %% [markdown]
# ## 5.3 Preprocesamiento con scikit-learn
#
# 1. Lectura de la base y de la partición; separación en entrenamiento y prueba.
# 2. Agrupación de categorías raras aprendida en entrenamiento: en `purpose` las categorías con
#    menos de 1 % pasan a `other` y en `addr_state` a `OTROS`.
# 3. `ColumnTransformer` ajustado solo con entrenamiento:
#    * bloque numérico (20 variables + 4 indicadores): `SimpleImputer(strategy="median")` seguido
#      de `StandardScaler`. El escalado es necesario para la regresión logística y el SVM lineal,
#      que penalizan coeficientes y son sensibles a la escala; a los árboles no les afecta, y se usa
#      la misma matriz para los seis modelos para que la comparación sea limpia.
#    * bloque categórico: `OneHotEncoder(handle_unknown="ignore")`, que codifica como ceros una
#      categoría nueva en prueba.
#
# El enunciado pide `GridSearchCV` **sin** `Pipeline`, así que el preprocesamiento se ajusta una vez
# con todo el conjunto de entrenamiento y la búsqueda trabaja sobre la matriz ya transformada. Esto
# introduce una fuga mínima dentro de la validación cruzada (la mediana y la escala de cada pliegue
# de validación se calcularon incluyendo sus filas); con 1,08 millones de filas el efecto sobre
# medianas, medias y desviaciones es despreciable, y PySpark sigue exactamente el mismo esquema.

# %%
t0 = time.time()
sk = fe.preparar_sklearn()
tiempos["preproc_sklearn_s"] = time.time() - t0
print(f"Tiempo de preprocesamiento (lectura + ajuste + transformación): {tiempos['preproc_sklearn_s']:.1f} s")
print(f"X_train: {sk['X_train'].shape}, X_test: {sk['X_test'].shape}")
print(f"Tasa de default: entrenamiento {sk['y_train'].mean():.4f}, prueba {sk['y_test'].mean():.4f}")
print("\nCategorías que se conservan (>= 1 % en entrenamiento):")
for c, v in sk["frecuentes"].items():
    print(f"  {c} ({len(v)}): {v}")

# %%
num = sk["ct"].named_transformers_["num"]
param_sk = pd.DataFrame({"mediana": num.named_steps["simpleimputer"].statistics_,
                         "media": num.named_steps["standardscaler"].mean_,
                         "desv_est": num.named_steps["standardscaler"].scale_}, index=fe.NUM_IND)
param_sk.round(4)

# %% [markdown]
# ## 5.4 Preprocesamiento con PySpark
#
# El flujo en Spark repite los mismos pasos sin sacar los datos de la JVM:
#
# 1. Lectura del CSV completo (`spark.read.csv`, todas las columnas como texto) y construcción de las
#    variables con `fe.base_spark`, que replica las reglas de pandas con funciones de
#    `pyspark.sql.functions`.
# 2. Lectura de `particion.parquet` y unión por `id`; entrenamiento y prueba se separan con
#    `filter(split == ...)`.
# 3. Agrupación de categorías raras con conteos de entrenamiento (`groupBy().count()`).
# 4. `Pipeline` con `Imputer` (mediana exacta, `relativeError=0`), `VectorAssembler` +
#    `StandardScaler(withMean=True, withStd=True)` para el bloque numérico, `StringIndexer` +
#    `OneHotEncoder` para las categóricas y un `VectorAssembler` final. El `Pipeline` se ajusta solo
#    con entrenamiento.
# 5. El resultado (`id`, `label`, `features`) se reparte en 40 particiones (4 por cada uno de los 10
#    núcleos del equipo; las 400 particiones configuradas rigen las uniones y agregaciones; la
#    justificación está en el capítulo 7) y se **cachea** con `persist(StorageLevel.MEMORY_AND_DISK)`
#    antes de cualquier modelo, como exige el enunciado.
#
# No se usa `.toPandas()`, y `.collect()` y `.first()` solo se aplican a resultados ya agregados
# (conteos por categoría y estadísticos): al driver nunca llegan filas de datos.

# %%
import spark_utils as su
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.stat import Summarizer
from pyspark.sql import functions as F

spark = su.sesion_spark()
print("Spark", spark.version, "| configuración:")
for k in su.CONFIG_OBLIGATORIA:
    print(f"  {k} = {spark.conf.get(k)}")

t0 = time.time()
datos = su.preparar_spark(spark)
tiempos["preproc_spark_s"] = time.time() - t0
train_s, test_s = datos["train"], datos["test"]
print(f"\nTiempo de preprocesamiento en Spark (lectura del CSV + variables + unión + Pipeline + caché): "
      f"{tiempos['preproc_spark_s']:.1f} s")
print(f"Entrenamiento: {datos['n_train']:,} filas | prueba: {datos['n_test']:,} filas | "
      f"particiones: {train_s.rdd.getNumPartitions()}")
print("Dimensión del vector de características:", len(datos["nombres"]))
print("¿Mismas categorías frecuentes que en pandas?", datos["frecuentes"] == sk["frecuentes"])

# %% [markdown]
# ## 5.5 Equivalencia entre entornos
#
# Se comparan tres niveles, siempre con agregados:
#
# 1. **Base sin codificar**: número de valores no nulos y suma de cada variable numérica, y conteo
#    por categoría, calculados en pandas y en Spark sobre las 1.348.059 filas.
# 2. **Parámetros aprendidos**: medianas, medias y desviaciones del escalado.
# 3. **Matriz final de entrenamiento**: media de cada una de las 72 columnas.

# %%
base_s = datos["base"]
agg_s = base_s.agg(*[F.count(c).alias(f"n_{c}") for c in fe.NUM_IND],
                   *[F.sum(c).alias(f"s_{c}") for c in fe.NUM_IND]).first().asDict()
comp = pd.DataFrame({
    "no_nulos_pandas": base[fe.NUM_IND].notna().sum(),
    "no_nulos_spark": [agg_s[f"n_{c}"] for c in fe.NUM_IND],
    "suma_pandas": base[fe.NUM_IND].sum(),
    "suma_spark": [agg_s[f"s_{c}"] for c in fe.NUM_IND]})
comp["dif_relativa_suma"] = (comp["suma_spark"] - comp["suma_pandas"]).abs() / comp["suma_pandas"].abs().clip(lower=1)
print("Filas: pandas", len(base), "| Spark", datos["n_base"])
print("Máxima diferencia en no nulos:", int((comp["no_nulos_pandas"] - comp["no_nulos_spark"]).abs().max()))
print("Máxima diferencia relativa en sumas:", f"{comp['dif_relativa_suma'].max():.2e}")
cat_ok = all(
    base[c].value_counts().sort_index().to_dict()
    == {r[c]: r["count"] for r in base_s.groupBy(c).count().collect()}
    for c in fe.CAT)
print("Conteos por categoría idénticos:", cat_ok)
comp.round(4)

# %%
param_spark = pd.DataFrame({"mediana": datos["medianas"], "media": datos["medias"],
                            "desv_est": datos["desv"]}, index=fe.NUM_IND)
dif = (param_spark - param_sk).abs()
print("Máxima diferencia absoluta en medianas:", f"{dif['mediana'].max():.2e}")
print("Máxima diferencia absoluta en medias:  ", f"{dif['media'].max():.2e}")
print("Máxima diferencia relativa en desviaciones:", f"{(dif['desv_est'] / param_sk['desv_est']).max():.2e}")
param_spark.join(param_sk, lsuffix="_spark", rsuffix="_sklearn").round(4)

# %% [markdown]
# Las desviaciones estándar difieren en menos de media parte por millón (diferencia relativa máxima
# de 4,6 × 10⁻⁷): `StandardScaler` de scikit-learn usa
# la desviación poblacional (divide entre $n$) y el de Spark la muestral (divide entre $n-1$); con
# $n$ = 1.078.447 el cociente es $\sqrt{n/(n-1)} \approx 1 + 4{,}6 \times 10^{-7}$.

# %%
medias_spark = np.asarray(train_s.select(Summarizer.mean(F.col("features")).alias("m")).first()["m"])
medias_sk = pd.Series(sk["X_train"].mean(axis=0), index=sk["nombres"])
# Spark nombra las columnas one-hot como "<var>_ohe_<categoría>"; se alinean con las de sklearn
mapa = {n: n.replace("_ohe_", "_") if "_ohe_" in n else fe.NUM_IND[i] for i, n in enumerate(datos["nombres"])}
medias_spark = pd.Series(medias_spark, index=[mapa[n] for n in datos["nombres"]])
print(f"Columnas: scikit-learn {len(medias_sk)} | Spark {len(medias_spark)} | "
      f"mismos nombres: {set(medias_sk.index) == set(medias_spark.index)}")
alineado = pd.DataFrame({"sklearn": medias_sk, "spark": medias_spark.reindex(medias_sk.index)})
print("Máxima diferencia absoluta entre medias de columnas:",
      f"{(alineado['sklearn'] - alineado['spark']).abs().max():.2e}")

# %% [markdown]
# Las dos matrices de entrenamiento coinciden columna a columna: mismo número de filas, mismas 72
# columnas y medias iguales salvo por errores de redondeo (diferencia máxima de 10⁻¹²). Las únicas
# diferencias entre entornos son el nombre de las columnas one-hot (Spark las llama
# `<variable>_ohe_<categoría>` y se alinean por nombre antes de comparar) y la desviación muestral o
# poblacional del escalado, sin efecto práctico.

# %%
u.guardar_json(tiempos, u.RESULTADOS / "tiempos_preprocesamiento.json")
datos["train"].unpersist()
datos["test"].unpersist()
spark.stop()
pd.Series(tiempos).round(1)
