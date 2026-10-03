"""Construcción de variables idéntica en pandas y en PySpark (capítulo 5).

Cada regla aparece dos veces, una por entorno, una al lado de la otra, para que la equivalencia
pueda revisarse línea a línea. Hay dos tipos de transformaciones:

* Deterministas, fila a fila, sin parámetros aprendidos (conversión de tipos, `log1p`, códigos
  inválidos de `dti`, indicadores de faltante, unificación de `home_ownership`). Se aplican antes
  de la partición: no pueden filtrar información del conjunto de prueba.
* Aprendidas de los datos (agrupación de categorías raras, mediana de imputación, media y
  desviación del escalado, categorías del one-hot). Se ajustan solo con el conjunto de
  entrenamiento.
"""
import numpy as np
import pandas as pd

import utils as u

_V = u.cargar_json(u.PROCESSED / "variables_modelo.json")
NUM = _V["numericas"]            # 20 variables numéricas originales
LOG = _V["log1p"]                # 3 montos con log(1 + x)
IND = _V["indicadores"]          # 4 indicadores binarios
CAT = _V["categoricas"]          # 6 categóricas
NUM_IND = NUM + IND              # bloque numérico que se imputa y escala
RARAS = {"purpose": "other", "addr_state": "OTROS"}   # variable -> categoría de agrupación
UMBRAL_RARA = 0.01
ESTADOS = {"Fully Paid": 0, "Does not meet the credit policy. Status:Fully Paid": 0,
           "Charged Off": 1, "Does not meet the credit policy. Status:Charged Off": 1}
HOME_OTRAS = ["ANY", "NONE", "OTHER"]


# ----------------------------------------------------------------------------------------------
# pandas
# ----------------------------------------------------------------------------------------------
def base_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe el parquet intermedio (capítulo 1: `term`, `emp_length` y `antig_credito_meses` ya
    convertidos) y devuelve id, default y las variables del modelo sin codificar."""
    b = pd.DataFrame({"id": df["id"].astype("int64"), "default": df["default"].astype("int8")})
    for c in NUM:
        b[c] = df[c].astype("float64")
    b.loc[(b["dti"] < 0) | (b["dti"] > 100), "dti"] = np.nan
    b["hist_morosidad"] = df["mths_since_last_delinq"].notna().astype("float64")
    b["emp_length_faltante"] = df["emp_length"].isna().astype("float64")
    b["consulta_faltante"] = df["mths_since_recent_inq"].isna().astype("float64")
    b["buro_faltante"] = df["avg_cur_bal"].isna().astype("float64")
    for c in LOG:
        b[c] = np.log1p(b[c])
    for c in CAT:
        b[c] = df[c].astype("object")
    b["home_ownership"] = b["home_ownership"].replace({h: "OTHER" for h in HOME_OTRAS})
    return b


def base_pandas_desde_csv() -> pd.DataFrame:
    """Mismo recorrido que hace Spark: lee el CSV comprimido (solo las columnas necesarias), filtra
    los préstamos cerrados, aplica las conversiones del capítulo 1 y llama a `base_pandas`."""
    cols = sorted({"id", "loan_status", "issue_d", "earliest_cr_line", "mths_since_last_delinq",
                   "avg_cur_bal", "mths_since_recent_inq", *NUM, *CAT} - {"antig_credito_meses"})
    df = pd.read_csv(u.CSV_CRUDO, usecols=cols, low_memory=False)
    df = df[df["loan_status"].isin(ESTADOS)].copy()
    df["default"] = df["loan_status"].map(ESTADOS)
    df["id"] = df["id"].astype("int64")
    df["term"] = df["term"].str.extract(r"(\d+)")[0].astype("float64")
    df["emp_length"] = u.emp_length_a_numero(df["emp_length"])
    emision = pd.to_datetime(df["issue_d"], format="%b-%Y")
    primera = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["antig_credito_meses"] = ((emision.dt.year - primera.dt.year) * 12
                                 + (emision.dt.month - primera.dt.month)).astype("float64")
    return base_pandas(df)


def categorias_frecuentes_pandas(train: pd.DataFrame) -> dict:
    """Categorías con al menos 1 % del conjunto de entrenamiento, por variable a agrupar."""
    return {c: sorted(train[c].value_counts(normalize=True).loc[lambda s: s >= UMBRAL_RARA].index)
            for c in RARAS}


def agrupar_raras_pandas(df: pd.DataFrame, frecuentes: dict) -> pd.DataFrame:
    df = df.copy()
    for c, otra in RARAS.items():
        df[c] = df[c].where(df[c].isin(frecuentes[c]), otra)
    return df


def preprocesador_sklearn():
    """ColumnTransformer: mediana + estandarización en el bloque numérico y one-hot en las
    categóricas. Se ajusta solo con entrenamiento."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_IND),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64), CAT)],
        verbose_feature_names_out=False)


def preparar_sklearn(base=None):
    """Lee la base (o usa la recibida) y la partición, ajusta el preprocesamiento con entrenamiento y
    transforma ambos conjuntos. Devuelve matrices, etiquetas, ids y objetos ajustados."""
    if base is None:
        base = pd.read_parquet(u.MODELO)
    split = pd.read_parquet(u.PARTICION)
    base = base.merge(split, on="id", how="inner", validate="one_to_one")
    train = base[base["split"] == "train"].reset_index(drop=True)
    test = base[base["split"] == "test"].reset_index(drop=True)
    if u.FRACCION_PRUEBA:   # misma regla por id que en Spark, para que las pruebas emparejen
        k = round(1 / u.FRACCION_PRUEBA)
        train = train[train["id"] % k == 0].reset_index(drop=True)
        test = test[test["id"] % k == 0].reset_index(drop=True)
    frecuentes = categorias_frecuentes_pandas(train)
    train, test = agrupar_raras_pandas(train, frecuentes), agrupar_raras_pandas(test, frecuentes)
    ct = preprocesador_sklearn().fit(train[NUM_IND + CAT])
    return {"X_train": ct.transform(train[NUM_IND + CAT]), "X_test": ct.transform(test[NUM_IND + CAT]),
            "y_train": train["default"].to_numpy(), "y_test": test["default"].to_numpy(),
            "id_train": train["id"].to_numpy(), "id_test": test["id"].to_numpy(),
            "nombres": list(ct.get_feature_names_out()), "ct": ct, "frecuentes": frecuentes}


# ----------------------------------------------------------------------------------------------
# PySpark
# ----------------------------------------------------------------------------------------------
def base_spark(crudo):
    """Recibe el CSV leído por Spark (todas las columnas como texto) y devuelve id, default y las
    variables del modelo sin codificar, con las mismas reglas que `base_pandas` + capítulo 1."""
    from pyspark.sql import functions as F

    def num(c):
        return F.col(c).try_cast("double")

    mapa = F.create_map(*[x for k, v in ESTADOS.items() for x in (F.lit(k), F.lit(v))])
    df = crudo.filter(F.col("loan_status").isin(list(ESTADOS)))
    emision = F.to_date(F.concat(F.lit("01-"), F.col("issue_d")), "dd-MMM-yyyy")
    primera = F.to_date(F.concat(F.lit("01-"), F.col("earliest_cr_line")), "dd-MMM-yyyy")
    emp = F.regexp_extract(F.regexp_replace(F.col("emp_length"), "< 1", "0"), r"(\d+)", 1)
    cols = {
        "id": F.col("id").try_cast("long"),
        "default": mapa[F.col("loan_status")].cast("int"),
        "term": F.regexp_extract(F.col("term"), r"(\d+)", 1).try_cast("double"),
        "emp_length": F.when(emp == "", None).otherwise(emp).try_cast("double"),
        "antig_credito_meses": ((F.year(emision) - F.year(primera)) * 12
                                + (F.month(emision) - F.month(primera))).cast("double"),
    }
    for c in NUM:
        cols.setdefault(c, num(c))
    dti = cols["dti"]
    cols["dti"] = F.when((dti < 0) | (dti > 100), None).otherwise(dti)
    cols["hist_morosidad"] = num("mths_since_last_delinq").isNotNull().cast("double")
    cols["emp_length_faltante"] = cols["emp_length"].isNull().cast("double")
    cols["consulta_faltante"] = num("mths_since_recent_inq").isNull().cast("double")
    cols["buro_faltante"] = num("avg_cur_bal").isNull().cast("double")
    for c in LOG:
        cols[c] = F.log1p(cols[c])
    for c in CAT:
        cols[c] = F.col(c)
    cols["home_ownership"] = F.when(F.col("home_ownership").isin(HOME_OTRAS), "OTHER") \
        .otherwise(F.col("home_ownership"))
    return df.select(*[cols[c].alias(c) for c in ["id", "default"] + NUM_IND + CAT])


def categorias_frecuentes_spark(train) -> dict:
    """Mismo criterio que en pandas, calculado con una agregación distribuida (solo viajan al
    driver los conteos por categoría, no los datos)."""
    from pyspark.sql import functions as F
    n = train.count()
    res = {}
    for c in RARAS:
        filas = train.groupBy(c).count().filter(F.col("count") / n >= UMBRAL_RARA).collect()
        res[c] = sorted(r[c] for r in filas)
    return res


def agrupar_raras_spark(df, frecuentes: dict):
    from pyspark.sql import functions as F
    for c, otra in RARAS.items():
        df = df.withColumn(c, F.when(F.col(c).isin(frecuentes[c]), F.col(c)).otherwise(F.lit(otra)))
    return df


def etapas_spark():
    """Etapas del Pipeline de Spark equivalentes al ColumnTransformer de scikit-learn."""
    from pyspark.ml.feature import (Imputer, OneHotEncoder, StandardScaler, StringIndexer,
                                    VectorAssembler)
    imp = [f"{c}_imp" for c in NUM_IND]
    idx = [f"{c}_idx" for c in CAT]
    ohe = [f"{c}_ohe" for c in CAT]
    return [
        Imputer(inputCols=NUM_IND, outputCols=imp, strategy="median", relativeError=0.0),
        VectorAssembler(inputCols=imp, outputCol="num_vec"),
        StandardScaler(inputCol="num_vec", outputCol="num_esc", withMean=True, withStd=True),
        StringIndexer(inputCols=CAT, outputCols=idx, handleInvalid="keep",
                      stringOrderType="alphabetAsc"),
        # StringIndexer(keep) asigna a una categoría nueva el último índice ("__unknown") y
        # OneHotEncoder(dropLast=True) elimina justo esa columna: la categoría nueva queda como un
        # vector de ceros, igual que OneHotEncoder(handle_unknown="ignore") de scikit-learn.
        OneHotEncoder(inputCols=idx, outputCols=ohe, dropLast=True, handleInvalid="error"),
        VectorAssembler(inputCols=["num_esc"] + ohe, outputCol="features"),
    ]


def nombres_spark(df, col="features") -> list:
    """Nombres de las columnas del vector ensamblado, leídos de los metadatos de Spark."""
    attrs = df.schema[col].metadata["ml_attr"]["attrs"]
    pares = [(a["idx"], a["name"]) for grupo in attrs.values() for a in grupo]
    return [n for _, n in sorted(pares)]
