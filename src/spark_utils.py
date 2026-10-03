"""Sesión de Spark y lectura del CSV de Lending Club (configuración de la sección 9.10.4.5)."""
import os
import subprocess

import utils as u

# PySpark usa el Python del entorno virtual tanto en el driver como en los workers
os.environ.setdefault("PYSPARK_PYTHON", u.PYTHON_VENV)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", u.PYTHON_VENV)
if "JAVA_HOME" not in os.environ:
    for version in ("21", "17"):
        r = subprocess.run(["/usr/libexec/java_home", "-v", version], capture_output=True, text=True)
        if r.returncode == 0:
            os.environ["JAVA_HOME"] = r.stdout.strip()
            break

from pyspark.sql import SparkSession  # noqa: E402

# Particiones del conjunto de entrenamiento y prueba cacheado (4 por núcleo en un equipo de 10
# núcleos). La sesión conserva la configuración obligatoria (400 particiones de shuffle); el
# capítulo 11 compara 400, 40 y 10 particiones de datos.
PARTICIONES_DATOS = 40

CONFIG_OBLIGATORIA = {
    "spark.sql.shuffle.partitions": "400",
    "spark.default.parallelism": "400",
    "spark.executor.memory": "8g",
    "spark.driver.memory": "8g",
    "spark.memory.fraction": 0.8,
    "spark.memory.storageFraction": 0.3,
}


def sesion_spark(nombre="LendingClub_Optimized", extra=None) -> SparkSession:
    """Crea la sesión con la configuración obligatoria del enunciado.

    En modo local el driver y el ejecutor comparten la misma JVM, así que la memoria efectiva es
    `spark.driver.memory`; `spark.executor.memory` se fija por fidelidad al enunciado.
    """
    b = SparkSession.builder.appName(nombre).master("local[*]")
    prueba = {"spark.sql.shuffle.partitions": "8"} if u.FRACCION_PRUEBA else {}   # solo pruebas de humo
    for k, v in {**CONFIG_OBLIGATORIA, **prueba, **(extra or {})}.items():
        b = b.config(k, v)
    # Dirección local fija: si la red cambia (p. ej., al despertar el equipo) el driver no pierde
    # su propia dirección y la aplicación no se cae
    b = (b.config("spark.driver.host", "127.0.0.1").config("spark.driver.bindAddress", "127.0.0.1")
          .config("spark.driver.maxResultSize", "4g")
          .config("spark.ui.showConsoleProgress", "false")
          .config("spark.local.dir", str(u.MODELOS / "spark_tmp")))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def leer_csv(spark):
    """Lee el CSV completo con todas las columnas como texto (sin inferSchema: una sola pasada).

    `multiLine` y `escape` son necesarios porque algunas descripciones (`desc`) contienen saltos de
    línea y comillas dentro de campos entrecomillados.
    """
    return (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
            .csv(str(u.CSV_CRUDO)))


def preparar_spark(spark, particiones=None, cachear=True):
    """Preprocesamiento completo en Spark (capítulo 5), sin traer datos al driver.

    1. Lee el CSV y construye las variables (`features.base_spark`).
    2. Une la partición común por `id` y separa entrenamiento y prueba con `filter`.
    3. Agrupa categorías raras y ajusta el `Pipeline` solo con entrenamiento.
    4. Cachea (`MEMORY_AND_DISK`) el resultado del `VectorAssembler`: `id`, `label`, `features`.

    El CSV comprimido llega en una sola partición (gzip no es divisible); se reparticiona en
    `PARTICIONES_DATOS` = 40 particiones (o en `particiones` si se indica). La sesión mantiene
    `spark.sql.shuffle.partitions` = 400 para las agregaciones y uniones. La base unida también se cachea mientras se ajusta el `Pipeline`,
    para no releer y descomprimir el CSV en cada pasada; quien llama la libera con
    `datos["base"].unpersist()` cuando ya no la necesita.
    """
    from pyspark import StorageLevel
    from pyspark.ml import Pipeline
    from pyspark.sql import functions as F

    import features as fe

    nivel = StorageLevel.MEMORY_AND_DISK
    n_part = particiones or PARTICIONES_DATOS
    split = spark.read.parquet(str(u.PARTICION))
    base = fe.base_spark(leer_csv(spark)).join(split, on="id", how="inner").repartition(n_part)
    if cachear:
        base = base.persist(nivel)
    n_base = base.count()
    train = base.filter(F.col("split") == "train").drop("split")
    test = base.filter(F.col("split") == "test").drop("split")
    if u.FRACCION_PRUEBA:
        k = round(1 / u.FRACCION_PRUEBA)
        train, test = train.filter(F.col("id") % k == 0), test.filter(F.col("id") % k == 0)
    frecuentes = fe.categorias_frecuentes_spark(train)
    train, test = fe.agrupar_raras_spark(train, frecuentes), fe.agrupar_raras_spark(test, frecuentes)
    modelo = Pipeline(stages=fe.etapas_spark()).fit(train)

    def sel(d):
        return modelo.transform(d).select("id", F.col("default").cast("double").alias("label"), "features")

    tr, te = sel(train), sel(test)
    if cachear:
        tr, te = tr.persist(nivel), te.persist(nivel)
    n_train, n_test = tr.count(), te.count()
    imp, esc = modelo.stages[0], modelo.stages[2]
    medianas = imp.surrogateDF.first().asDict()
    return {"train": tr, "test": te, "base": base, "pipeline": modelo, "frecuentes": frecuentes,
            "n_base": n_base, "n_train": n_train, "n_test": n_test, "nombres": fe.nombres_spark(tr),
            "medianas": [medianas[c] for c in fe.NUM_IND], "medias": list(esc.mean.toArray()),
            "desv": list(esc.std.toArray())}
