"""Umbral de decisión y métricas, con la misma definición en scikit-learn y en PySpark.

Umbral: se maximiza el índice de Youden (J = TPR - FPR) con las puntuaciones del conjunto de
ENTRENAMIENTO, evaluando como candidatos los cuantiles 0,1 %, 0,2 %, ..., 99,9 % de esas
puntuaciones (cuantiles que son valores observados). La misma rejilla se usa en los dos entornos;
en Spark se calcula con agregaciones distribuidas, sin traer las puntuaciones al driver.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

PROBS = [i / 1000 for i in range(1, 1000)]


def _youden_desde_conteos(umbrales, pos_desde, neg_desde, P, N):
    """pos_desde[k] y neg_desde[k]: positivos y negativos con puntuación >= umbrales[k]."""
    j = pos_desde / P - neg_desde / N
    k = int(np.argmax(j))
    return float(umbrales[k]), float(j[k])


def umbral_youden(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    umbrales = np.unique(np.quantile(s, PROBS, method="inverted_cdf"))
    s_pos, s_neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    pos_desde = len(s_pos) - np.searchsorted(s_pos, umbrales, side="left")
    neg_desde = len(s_neg) - np.searchsorted(s_neg, umbrales, side="left")
    return _youden_desde_conteos(umbrales, pos_desde, neg_desde, len(s_pos), len(s_neg))


def metricas(y, s, umbral) -> dict:
    """Métricas de clasificación en el umbral dado y de ordenamiento (AUC ROC y AUC-PR)."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"umbral": float(umbral), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc_roc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# ----------------------------------------------------------------------------------------------
# PySpark (todo con agregaciones: al driver solo llegan conteos)
# ----------------------------------------------------------------------------------------------
def umbral_youden_spark(df, col="score", label="label"):
    from pyspark.ml.feature import Bucketizer
    from pyspark.sql import functions as F
    umbrales = np.unique(df.approxQuantile(col, PROBS, 0.0))
    cortes = [-float("inf")] + list(umbrales) + [float("inf")]
    b = Bucketizer(splits=cortes, inputCol=col, outputCol="_cubeta").transform(df)
    filas = b.groupBy("_cubeta").agg(F.sum(F.col(label)).alias("pos"), F.count("*").alias("n")).collect()
    k = len(cortes) - 1
    pos, n = np.zeros(k), np.zeros(k)
    for r in filas:
        pos[int(r["_cubeta"])] = r["pos"]
        n[int(r["_cubeta"])] = r["n"]
    neg = n - pos
    # cubeta i = [cortes[i], cortes[i+1]); el umbral umbrales[j] = cortes[j+1] deja como positivas
    # las cubetas j+1, j+2, ...
    pos_desde = np.cumsum(pos[::-1])[::-1][1:]
    neg_desde = np.cumsum(neg[::-1])[::-1][1:]
    return _youden_desde_conteos(umbrales, pos_desde, neg_desde, pos.sum(), neg.sum())


def metricas_spark(df, umbral, col="score", label="label") -> dict:
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import functions as F
    pred = (F.col(col) >= umbral).cast("int")
    y = F.col(label).cast("int")
    r = df.agg(F.sum(((pred == 1) & (y == 1)).cast("int")).alias("tp"),
               F.sum(((pred == 1) & (y == 0)).cast("int")).alias("fp"),
               F.sum(((pred == 0) & (y == 1)).cast("int")).alias("fn"),
               F.sum(((pred == 0) & (y == 0)).cast("int")).alias("tn")).first()
    tp, fp, fn, tn = (int(r[k]) for k in ("tp", "fp", "fn", "tn"))
    ev = BinaryClassificationEvaluator(rawPredictionCol=col, labelCol=label, numBins=0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"umbral": float(umbral), "accuracy": (tp + tn) / (tp + tn + fp + fn), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "auc_roc": ev.evaluate(df, {ev.metricName: "areaUnderROC"}),
            "auc_pr_spark": ev.evaluate(df, {ev.metricName: "areaUnderPR"}),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}
