"""Ajuste, evaluación y registro de resultados de los modelos (capítulos 6 y 7).

Cada modelo deja en `data/resultados/<entorno>/`:
* `<modelo>.json`: hiperparámetros, AUC de validación cruzada, tiempos y métricas;
* `<modelo>_cv.csv`: AUC medio por combinación de hiperparámetros;
y sus puntuaciones de prueba en `data/resultados/puntuaciones_<entorno>.parquet` (id, default y
una columna por modelo). Si los archivos ya existen, el cuaderno los lee en lugar de reentrenar
(salvo `LC_FORZAR=1`), de modo que el libro puede reconstruirse sin repetir horas de cómputo.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

import evaluacion as ev
import utils as u

FORZAR = os.environ.get("LC_FORZAR", "0") == "1"
REG_PARAMS = [1e-6, 1e-5, 1e-4]
ORDEN = ["LogisticRegression", "DecisionTree", "RandomForest", "GradientBoosting", "LinearSVC",
         "NaiveBayes"]


def carpeta(entorno):
    d = u.RESULTADOS / entorno
    d.mkdir(parents=True, exist_ok=True)
    return d


def ya_entrenado(entorno, nombre):
    return (not FORZAR) and (carpeta(entorno) / f"{nombre}.json").exists()


def cargar_resultado(entorno, nombre):
    return u.cargar_json(carpeta(entorno) / f"{nombre}.json")


def guardar_puntuaciones(entorno, nombre, ids, y, s):
    """Añade (o reemplaza) la columna del modelo. Exige que las observaciones sean exactamente las
    mismas que las ya guardadas y escribe primero en un archivo temporal (escritura atómica)."""
    ruta = u.RESULTADOS / f"puntuaciones_{entorno}.parquet"
    nuevo = pd.DataFrame({"id": np.asarray(ids, dtype="int64"), "default": np.asarray(y, dtype="int8"),
                          nombre: np.asarray(s, dtype="float64")}).sort_values("id")
    if not np.isfinite(nuevo[nombre]).all():
        raise ValueError(f"{nombre}: puntuaciones no finitas")
    if ruta.exists():
        prev = pd.read_parquet(ruta).drop(columns=[nombre], errors="ignore").sort_values("id")
        if not (np.array_equal(prev["id"].to_numpy(), nuevo["id"].to_numpy())
                and np.array_equal(prev["default"].to_numpy(), nuevo["default"].to_numpy())):
            raise ValueError(f"{nombre}: las observaciones de prueba no coinciden con {ruta.name}")
        nuevo = prev.merge(nuevo, on=["id", "default"], how="inner", validate="one_to_one")
    tmp = ruta.with_suffix(".tmp")
    nuevo.to_parquet(tmp, index=False)
    tmp.replace(ruta)


def leer_puntuaciones(entorno):
    return pd.read_parquet(u.RESULTADOS / f"puntuaciones_{entorno}.parquet")


# ----------------------------------------------------------------------------------------------
# scikit-learn
# ----------------------------------------------------------------------------------------------
def ajustar_sklearn(nombre, estimador, grid, d, usa_decision=False, n_jobs=-1, n_jobs_reajuste=None):
    """GridSearchCV (3 pliegues estratificados, AUC ROC) sobre la matriz ya preprocesada.

    Con `n_jobs_reajuste` (bosque aleatorio) la búsqueda reparte los ajustes entre núcleos con
    estimadores de un hilo y el reajuste final se hace aparte con `n_jobs_reajuste` hilos; si no,
    `GridSearchCV(refit=True)` reajusta con el mismo estimador. En ambos casos el tiempo de
    entrenamiento incluye validación cruzada + reajuste, como `CrossValidator.fit` en Spark.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if ya_entrenado("sklearn", nombre):
        r = cargar_resultado("sklearn", nombre)
        print(f"{nombre}: resultados leídos de disco (entrenado el {r['fecha']})")
        return r
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=u.SEMILLA)
    manual = n_jobs_reajuste is not None
    gs = GridSearchCV(estimador, grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, refit=not manual,
                      error_score="raise")
    t0 = time.time()
    gs.fit(d["X_train"], d["y_train"])
    if manual:
        t_cv = time.time() - t0
        t1 = time.time()
        mejor = clone(estimador).set_params(**gs.best_params_, n_jobs=n_jobs_reajuste)
        mejor.fit(d["X_train"], d["y_train"])
        t_reajuste = time.time() - t1
    else:
        mejor, t_reajuste = gs.best_estimator_, gs.refit_time_
        t_cv = time.time() - t0 - t_reajuste
    t_fit = time.time() - t0
    puntuar = mejor.decision_function if usa_decision else (lambda X: mejor.predict_proba(X)[:, 1])
    t0 = time.time()
    s_test = puntuar(d["X_test"])
    t_pred = time.time() - t0
    t0 = time.time()
    s_train = puntuar(d["X_train"])
    umbral, j = ev.umbral_youden(d["y_train"], s_train)
    t_umbral = time.time() - t0
    defecto = 0.0 if usa_decision else 0.5
    t0 = time.time()
    m_youden = ev.metricas(d["y_test"], s_test, umbral)
    m_defecto = ev.metricas(d["y_test"], s_test, defecto)
    t_eval = time.time() - t0
    cvres = pd.DataFrame(gs.cv_results_)
    cols = [c for c in cvres if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                           "mean_fit_time", "rank_test_score"]
    cvres[cols].to_csv(carpeta("sklearn") / f"{nombre}_cv.csv", index=False)
    guardar_puntuaciones("sklearn", nombre, d["id_test"], d["y_test"], s_test)
    joblib.dump(mejor, u.MODELOS / f"sklearn_{nombre}.joblib", compress=3)
    n_iter = getattr(mejor, "n_iter_", None)
    r = {"modelo": nombre, "entorno": "sklearn", "mejores_parametros": gs.best_params_,
         "t_cv_s": t_cv, "t_reajuste_s": t_reajuste,
         "iteraciones": None if n_iter is None else int(np.max(n_iter)),
         "max_iter": mejor.get_params().get("max_iter"),
         "auc_cv": gs.best_score_, "auc_train": ev.metricas(d["y_train"], s_train, umbral)["auc_roc"],
         "umbral_youden": umbral, "youden_train": j, "umbral_defecto": defecto,
         "t_entrenamiento_cv_s": t_fit, "t_prediccion_s": t_pred, "t_umbral_s": t_umbral,
         "t_evaluacion_s": t_eval, "n_combinaciones": len(cvres), "n_ajustes": 3 * len(cvres) + 1,
         "metricas_youden": m_youden, "metricas_defecto": m_defecto,
         "fecha": time.strftime("%Y-%m-%d %H:%M")}
    u.guardar_json(r, carpeta("sklearn") / f"{nombre}.json")
    return r


def resumen(r) -> pd.Series:
    m = r["metricas_youden"]
    return pd.Series({"AUC CV": r["auc_cv"], "AUC prueba": m["auc_roc"], "AUC-PR": m.get("auc_pr"),
                      "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"],
                      "F1": m["f1"], "umbral": r["umbral_youden"],
                      "t_cv (s)": r["t_entrenamiento_cv_s"], "t_pred (s)": r["t_prediccion_s"]},
                     name=r["modelo"])


# ----------------------------------------------------------------------------------------------
# PySpark
# ----------------------------------------------------------------------------------------------
def ajustar_spark(nombre, estimador, grid, datos, usa_raw=False, paralelismo=1):
    """CrossValidator (3 pliegues, AUC ROC) con ParamGridBuilder; sin bucles sobre hiperparámetros.

    La puntuación continua es el segundo elemento de `probability` o, en LinearSVC, de
    `rawPrediction`, extraído con `vector_to_array`. El evaluador del CrossValidator usa esa misma
    columna: en el árbol de decisión `rawPrediction` contiene conteos de la hoja y no ordena igual
    que la probabilidad, y en Naive Bayes contiene log-verosimilitudes no normalizadas.
    """
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.functions import vector_to_array
    from pyspark.ml.tuning import CrossValidator
    from pyspark.sql import functions as F

    if ya_entrenado("spark", nombre):
        r = cargar_resultado("spark", nombre)
        print(f"{nombre}: resultados leídos de disco (entrenado el {r['fecha']})")
        return r
    train, test = datos["train"], datos["test"]
    col = "rawPrediction" if usa_raw else "probability"
    evaluador = BinaryClassificationEvaluator(rawPredictionCol=col, labelCol="label",
                                              metricName="areaUnderROC", numBins=0)
    cv = CrossValidator(estimator=estimador, estimatorParamMaps=grid, evaluator=evaluador,
                        numFolds=3, seed=u.SEMILLA, parallelism=paralelismo)
    t0 = time.time()
    modelo = cv.fit(train)
    t_fit = time.time() - t0
    mejor = modelo.bestModel
    puntuar = lambda df: mejor.transform(df).select(
        "id", "label", vector_to_array(F.col(col))[1].alias("score"))
    t0 = time.time()
    pred_test = puntuar(test).persist()
    pred_test.count()
    t_pred = time.time() - t0
    t0 = time.time()
    pred_train = puntuar(train)
    umbral, j = ev.umbral_youden_spark(pred_train)
    auc_train = BinaryClassificationEvaluator(rawPredictionCol="score", labelCol="label",
                                              numBins=0).evaluate(pred_train)
    t_umbral = time.time() - t0
    defecto = 0.0 if usa_raw else 0.5
    t0 = time.time()
    m_youden = ev.metricas_spark(pred_test, umbral)
    m_defecto = ev.metricas_spark(pred_test, defecto)
    t_eval = time.time() - t0
    # Transferencia permitida tras el entrenamiento: solo id, default y puntuación de prueba
    t0 = time.time()
    local = pred_test.select("id", "label", "score").toPandas()
    t_transfer = time.time() - t0
    pred_test.unpersist()
    for m, umb in ((m_youden, umbral), (m_defecto, defecto)):
        m["auc_pr"] = ev.metricas(local["label"], local["score"], umb)["auc_pr"]
    i = int(np.argmax(modelo.avgMetrics))
    mejores = {p.name: v for p, v in grid[i].items()}
    try:   # iteraciones del optimizador (modelos lineales) para comprobar la convergencia
        iteraciones = int(mejor.summary.totalIterations) or None
    except Exception:
        iteraciones = None
    filas = [{**{p.name: v for p, v in g.items()}, "mean_test_score": a, "std_test_score": s}
             for g, a, s in zip(grid, modelo.avgMetrics, modelo.stdMetrics)]
    pd.DataFrame(filas).to_csv(carpeta("spark") / f"{nombre}_cv.csv", index=False)
    guardar_puntuaciones("spark", nombre, local["id"], local["label"].astype(int), local["score"])
    mejor.write().overwrite().save(str(u.MODELOS / f"spark_{nombre}"))
    r = {"modelo": nombre, "entorno": "spark", "mejores_parametros": mejores,
         "auc_cv": float(modelo.avgMetrics[i]), "auc_train": auc_train, "umbral_youden": umbral,
         "youden_train": j, "umbral_defecto": defecto, "t_entrenamiento_cv_s": t_fit,
         "t_prediccion_s": t_pred, "t_umbral_s": t_umbral, "t_evaluacion_s": t_eval,
         "t_transferencia_s": t_transfer, "n_combinaciones": len(grid), "iteraciones": iteraciones,
         "max_iter": mejor.getOrDefault("maxIter") if mejor.hasParam("maxIter") else None,
         "n_ajustes": 3 * len(grid) + 1, "paralelismo_cv": paralelismo,
         "particiones": train.rdd.getNumPartitions(),
         "metricas_youden": m_youden, "metricas_defecto": m_defecto,
         "fecha": time.strftime("%Y-%m-%d %H:%M")}
    u.guardar_json(r, carpeta("spark") / f"{nombre}.json")
    return r
