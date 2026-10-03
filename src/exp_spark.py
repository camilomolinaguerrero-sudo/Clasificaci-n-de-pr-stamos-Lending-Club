"""Experimentos de rendimiento con Spark (capítulo 11). Cada llamada corre en un proceso nuevo,
porque la memoria del driver y otras opciones solo se aplican al arrancar la JVM.

Uso: python exp_spark.py '<json>'   ->  imprime una línea "RESULTADO <json>"

Claves del json de entrada:
  tarea:       "preparar" | "lr_cv" | "dt_cv" | "lr_fit" | "escala" | "maxbins" | "svc"
  particiones: número de particiones del DataFrame de entrenamiento (por defecto 40, como en el cap. 7)
  config:      dict con opciones de SparkSession que reemplazan las obligatorias
  cache:       true/false (persistir o no los datos de entrenamiento)
  desde_csv:   true para reconstruir desde el CSV (en lugar del parquet de experimentos)
  ...          parámetros propios de cada tarea
"""
import json
import sys
import time

import utils as u

EXP = u.MODELOS / "experimentos"
TRAIN_PQ, TEST_PQ = EXP / "train.parquet", EXP / "test.parquet"


def main(cfg):
    import spark_utils as su
    from pyspark import StorageLevel
    from pyspark.ml.classification import (DecisionTreeClassifier, LinearSVC, LogisticRegression,
                                           RandomForestClassifier)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.sql import functions as F

    spark = su.sesion_spark(extra=cfg.get("config"))
    spark.sparkContext.setCheckpointDir(str(u.MODELOS / "checkpoints"))
    out = {"entrada": cfg}
    tarea = cfg["tarea"]
    n_part = cfg.get("particiones", su.PARTICIONES_DATOS)

    if tarea == "preparar":
        datos = su.preparar_spark(spark)
        datos["train"].write.mode("overwrite").parquet(str(TRAIN_PQ))
        datos["test"].write.mode("overwrite").parquet(str(TEST_PQ))
        out["n_train"] = datos["n_train"]
        print("RESULTADO " + json.dumps(out))
        return

    t0 = time.time()
    if cfg.get("desde_csv"):
        datos = su.preparar_spark(spark, particiones=n_part, cachear=cfg.get("cache", True))
        if cfg.get("cache", True):
            datos["base"].unpersist()
        train, test = datos["train"], datos["test"]
    else:
        train = spark.read.parquet(str(TRAIN_PQ))
        test = spark.read.parquet(str(TEST_PQ))
        if cfg.get("fraccion", 1.0) < 1.0:
            train = train.sample(fraction=cfg["fraccion"], seed=u.SEMILLA)
        for _ in range(int(cfg.get("replicas", 1)) - 1):
            train = train.unionByName(spark.read.parquet(str(TRAIN_PQ)))
        train = train.repartition(n_part)
        test = test.repartition(n_part)
        if cfg.get("cache", True):
            train = train.persist(StorageLevel.MEMORY_AND_DISK)
            test = test.persist(StorageLevel.MEMORY_AND_DISK)
    out["n_train"] = train.count()
    test.count()
    out["t_preparacion_s"] = time.time() - t0
    out["particiones_efectivas"] = train.rdd.getNumPartitions()
    if cfg.get("cache", True):
        info = spark.sparkContext._jsc.sc().getRDDStorageInfo()
        out["cache_memoria_mb"] = sum(r.memSize() for r in info) / 2**20
        out["cache_disco_mb"] = sum(r.diskSize() for r in info) / 2**20

    ev_prob = BinaryClassificationEvaluator(rawPredictionCol="probability", labelCol="label", numBins=0)
    ev_raw = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", numBins=0)

    def ajustar(est, evaluador=ev_prob):
        t0 = time.time()
        m = est.fit(train)
        out["t_ajuste_s"] = time.time() - t0
        t0 = time.time()
        out["auc_test"] = evaluador.evaluate(m.transform(test))
        out["t_evaluacion_s"] = time.time() - t0
        return m

    def validar(est, grid):
        cv = CrossValidator(estimator=est, estimatorParamMaps=grid, evaluator=ev_prob, numFolds=3,
                            seed=u.SEMILLA, parallelism=cfg.get("paralelismo", 1))
        t0 = time.time()
        m = cv.fit(train)
        out["t_cv_s"] = time.time() - t0
        out["auc_cv"] = max(m.avgMetrics)
        out["auc_test"] = ev_prob.evaluate(m.bestModel.transform(test))

    if tarea == "lr_cv":
        lr = LogisticRegression(elasticNetParam=0.0, standardization=False, maxIter=100)
        validar(lr, ParamGridBuilder().addGrid(lr.regParam, [1e-6, 1e-5, 1e-4]).build())
    elif tarea == "dt_cv":
        dt = DecisionTreeClassifier(maxBins=32, seed=u.SEMILLA)
        validar(dt, ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]).build())
    elif tarea == "lr_fit":
        ajustar(LogisticRegression(elasticNetParam=0.0, standardization=False, regParam=1e-5,
                                   maxIter=cfg.get("max_iter", 100), tol=0.0 if cfg.get("iter_fijas") else 1e-6))
    elif tarea == "escala":
        modelo = cfg["modelo"]
        if modelo == "lr":
            ajustar(LogisticRegression(elasticNetParam=0.0, standardization=False, regParam=1e-5, maxIter=100))
        elif modelo == "dt":
            ajustar(DecisionTreeClassifier(maxDepth=10, maxBins=32, seed=u.SEMILLA))
        elif modelo == "rf":
            ajustar(RandomForestClassifier(numTrees=50, maxDepth=10, featureSubsetStrategy="8",
                                           maxBins=32, seed=u.SEMILLA, cacheNodeIds=True))
    elif tarea == "maxbins":
        ajustar(DecisionTreeClassifier(maxDepth=cfg.get("profundidad", 10), maxBins=cfg["maxbins"],
                                       seed=u.SEMILLA))
    elif tarea == "poda":
        m = ajustar(DecisionTreeClassifier(maxDepth=cfg["profundidad"], maxBins=32, seed=u.SEMILLA))
        out["nodos"] = m.numNodes
        out["hojas"] = m.toDebugString.count("Predict")
    elif tarea == "svc":
        if cfg.get("balanceado"):
            p1 = train.agg(F.mean("label")).first()[0]
            peso = F.when(F.col("label") == 1, 0.5 / p1).otherwise(0.5 / (1 - p1))
            train = train.withColumn("peso", peso)
        extra = {"weightCol": "peso"} if cfg.get("balanceado") else {}
        svc = LinearSVC(standardization=False, regParam=cfg.get("reg", 1e-5), maxIter=100, **extra)
        m = ajustar(svc, ev_raw)
        w = m.coefficients.toArray()
        out["norma_w"] = float((w ** 2).sum() ** 0.5)
        out["intercepto"] = float(m.intercept)
    print("RESULTADO " + json.dumps(out, default=float))
    spark.stop()


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
