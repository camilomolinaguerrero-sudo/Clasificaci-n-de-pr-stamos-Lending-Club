# %% [markdown]
# # 10. Comparación de resultados
#
# Sección 9.10.4.8 del enunciado. Este capítulo reúne en tablas y gráficos los resultados de los
# capítulos 6 a 8: métricas de los seis modelos en los dos entornos, tiempos de cómputo por etapa,
# curvas ROC y el resumen de las pruebas de DeLong, McNemar y bootstrap pareado.

# %%
import sys
import warnings

sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

import modelado as mo
import utils as u

u.estilo()
pd.set_option("display.max_columns", 30, "display.width", 200)
ENT = ("sklearn", "spark")
NOMBRE = {"sklearn": "scikit-learn", "spark": "PySpark"}
res = {e: {m: mo.cargar_resultado(e, m) for m in mo.ORDEN} for e in ENT}
pre = {e: u.cargar_json(u.RESULTADOS / f"{e}_preprocesamiento.json") for e in ENT}
punt = {e: mo.leer_puntuaciones(e) for e in ENT}

# %% [markdown]
# ## 10.1 Métricas en el conjunto de prueba
#
# Umbral de Youden elegido con entrenamiento; AUC ROC y AUC-PR no dependen del umbral.

# %%
filas = []
for m in mo.ORDEN:
    for e in ENT:
        r = res[e][m]
        mt = r["metricas_youden"]
        filas.append({"modelo": m, "entorno": NOMBRE[e], "parámetros": str(r["mejores_parametros"]),
                      "AUC CV": r["auc_cv"], "AUC train": r["auc_train"], "AUC prueba": mt["auc_roc"],
                      "AUC-PR": mt["auc_pr"], "Accuracy": mt["accuracy"], "Precision": mt["precision"],
                      "Recall": mt["recall"], "F1": mt["f1"], "umbral": r["umbral_youden"]})
metr = pd.DataFrame(filas).set_index(["modelo", "entorno"])
metr.to_csv(u.RESULTADOS / "tabla_metricas.csv")
metr.round(4)

# %% [markdown]
# Los dos entornos producen modelos casi idénticos en desempeño: en los cinco modelos comparables
# la diferencia de AUC de prueba entre entornos es de 0,003 o menos, y el F1 difiere en menos de
# 0,004. La columna "AUC train" muestra el sobreajuste de los modelos de alta capacidad:
# el bosque de profundidad 15 alcanza 0,81 (scikit-learn) y 0,80 (PySpark) en entrenamiento frente
# a 0,72 en prueba, y aun así es el segundo mejor en prueba; el gradient boosting, más regularizado
# por su tasa de aprendizaje y su poca profundidad, se mantiene en 0,73 frente a 0,72. El de mejor
# desempeño discriminativo en ambos entornos es el gradient boosting (AUC 0,7235 en scikit-learn y
# 0,7228 en PySpark; AUC-PR 0,395 y 0,393), aunque en el umbral de Youden la mayor precisión (0,339
# y 0,338) y la mayor exactitud (0,686 y 0,683) corresponden al bosque aleatorio, que marca menos
# préstamos como default.

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
for ax, k in zip(axes, ["AUC prueba", "AUC-PR", "F1", "Recall"]):
    t = metr[k].unstack("entorno").loc[mo.ORDEN]
    t.plot.barh(ax=ax, color=["#4C72B0", "#DD8452"], width=0.75, legend=(k == "AUC prueba"))
    ax.set_title(k)
    ax.set_xlim(max(0, t.values.min() - 0.05), min(1, t.values.max() + 0.03))
    ax.invert_yaxis()
fig.suptitle("Métricas en prueba por modelo y entorno", fontsize=13, weight="bold")
fig.tight_layout()
u.guardar_fig(fig, "10_metricas")
plt.show()

# %% [markdown]
# ## 10.2 Curvas ROC (una figura por entorno)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
colores = dict(zip(mo.ORDEN, sns.color_palette("tab10", 6)))
for ax, e in zip(axes, ENT):
    y = punt[e]["default"].to_numpy()
    for m in mo.ORDEN:
        fpr, tpr, _ = roc_curve(y, punt[e][m])
        ax.plot(fpr, tpr, color=colores[m], lw=1.6,
                label=f"{m} (AUC {res[e][m]['metricas_youden']['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"Curvas ROC · {NOMBRE[e]}")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_aspect("equal")
fig.tight_layout()
u.guardar_fig(fig, "10_roc")
plt.show()

# %% [markdown]
# Las curvas ROC de los dos entornos son prácticamente superponibles para los cuatro mejores
# modelos, que forman un haz estrecho; Naive Bayes queda por debajo en todo el rango, y el SVM de
# scikit-learn sigue la diagonal, mientras que el de PySpark se separa de ella.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, e in zip(axes, ENT):
    y = punt[e]["default"].to_numpy()
    for m in mo.ORDEN:
        pr, rc, _ = precision_recall_curve(y, punt[e][m])
        ax.plot(rc, pr, color=colores[m], lw=1.4, label=f"{m} (AP {res[e][m]['metricas_youden']['auc_pr']:.4f})")
    ax.axhline(y.mean(), color="k", ls="--", lw=0.8)
    ax.set_title(f"Curvas precisión-recall · {NOMBRE[e]}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.legend(fontsize=8.5)
fig.tight_layout()
u.guardar_fig(fig, "10_pr")
plt.show()

# %%
filas = {}
for e in ENT:
    for m in ["GradientBoosting", "RandomForest", "LogisticRegression"]:
        pr, rc, _ = precision_recall_curve(punt[e]["default"], punt[e][m])
        filas[(m, NOMBRE[e])] = pr[np.argmin(np.abs(rc - 0.5))]
pd.Series(filas, name="precisión con recall de 50 %").unstack().round(3)

# %% [markdown]
# En las curvas precisión-recall, la línea punteada es la precisión de un clasificador aleatorio
# (0,20). A un recall de 50 % los tres mejores modelos alcanzan una precisión de 0,37 a 0,38 en
# ambos entornos: menos de dos de cada cinco préstamos señalados como riesgosos terminan en
# default.

# %% [markdown]
# ## 10.3 Tiempos de cómputo
#
# * **Preprocesamiento**: en ambos entornos, desde el CSV comprimido hasta la matriz de diseño de
#   entrenamiento y prueba: lectura, filtro de préstamos cerrados, construcción de variables, unión
#   con la partición común y ajuste/transformación del preprocesamiento (en PySpark, además, la
#   materialización de la caché).
# * **Entrenamiento con validación**: `GridSearchCV.fit` / `CrossValidator.fit`, incluido el
#   reajuste del mejor modelo con todo el entrenamiento.
# * **Predicción**: puntuaciones de las 269.612 observaciones de prueba (en Spark, `transform` +
#   materialización).
# * **Transferencia**: solo PySpark, `toPandas()` de `id`, `default` y la puntuación.

# %%
filas = []
for m in mo.ORDEN:
    for e in ENT:
        r = res[e][m]
        filas.append({"modelo": m, "entorno": NOMBRE[e], "ajustes": r["n_ajustes"],
                      "entrenamiento + CV (s)": r["t_entrenamiento_cv_s"], "predicción (s)": r["t_prediccion_s"],
                      "umbral en train (s)": r["t_umbral_s"], "evaluación (s)": r["t_evaluacion_s"],
                      "transferencia (s)": r.get("t_transferencia_s", np.nan)})
tiem = pd.DataFrame(filas).set_index(["modelo", "entorno"])
cociente = (tiem["entrenamiento + CV (s)"].xs("PySpark", level="entorno")
            / tiem["entrenamiento + CV (s)"].xs("scikit-learn", level="entorno"))
tiem.to_csv(u.RESULTADOS / "tabla_tiempos.csv")
print("Preprocesamiento: scikit-learn "
      f"{pre['sklearn']['t_preprocesamiento_s']:.1f} s | PySpark {pre['spark']['t_preprocesamiento_s']:.1f} s")
tot = tiem.groupby("entorno")[["entrenamiento + CV (s)", "predicción (s)", "transferencia (s)"]].sum()
print("\nTotales por entorno (s):")
print(tot.round(1).to_string())
print("\nCociente de tiempo de entrenamiento + CV (PySpark / scikit-learn):")
print(cociente.round(2).to_string())
tiem.round(2)

# %% [markdown]
# **scikit-learn fue más rápido en cinco de los seis modelos**: el entrenamiento con validación
# cruzada de los seis tomó 1.967 s (33 min) en scikit-learn y 11.495 s (3,2 h) en PySpark, 5,8
# veces más. En los cinco modelos donde PySpark fue más lento, el cociente va de 3,0 (árbol) a 19,5
# (Naive Bayes). La única excepción es el SVM lineal, donde PySpark fue 2,7 veces más rápido (225 s
# frente a 616 s): en scikit-learn los ajustes con $C$ = 0,927 tardaron unos 600 s cada uno en
# converger cerca de la solución degenerada (capítulos 6 y 11), mientras que Spark limita su
# optimizador a 100 iteraciones. No es una ventaja del entorno sino del criterio de parada. El
# preprocesamiento también fue más rápido en scikit-learn (17,6 s frente a 34,5 s), y la predicción
# de las 269.612 observaciones tardó menos de 0,4 s en scikit-learn frente a 0,2-61 s en PySpark.
# Las etapas posteriores (umbral en entrenamiento y métricas distribuidas) suman en PySpark otros
# entre 5 y 533 s por modelo, frente a 0,3-1,8 s en scikit-learn. El capítulo 12 explica por
# qué.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
t = tiem["entrenamiento + CV (s)"].unstack("entorno").loc[mo.ORDEN]
t.plot.barh(ax=axes[0], color=["#DD8452", "#4C72B0"][::-1], width=0.75, logx=True)
axes[0].set_title("Entrenamiento con validación cruzada (s, escala log)")
axes[0].invert_yaxis()
t = tiem["predicción (s)"].unstack("entorno").loc[mo.ORDEN]
t.plot.barh(ax=axes[1], color=["#DD8452", "#4C72B0"][::-1], width=0.75, logx=True, legend=False)
axes[1].set_title("Predicción sobre 269.612 observaciones (s, escala log)")
axes[1].invert_yaxis()
fig.tight_layout()
u.guardar_fig(fig, "10_tiempos")
plt.show()

# %% [markdown]
# ## 10.4 Resumen de las pruebas estadísticas
#
# ### DeLong entre entornos

# %%
de = pd.read_csv(u.RESULTADOS / "delong_entre_entornos.csv")
de["IC 95 % ΔAUC"] = de.apply(lambda r: f"[{r['delta_inf']:+.4f}, {r['delta_sup']:+.4f}]", axis=1)
de["modelo"] = de["modelo_1"].str.split(" ").str[0]
de.set_index("modelo")[["auc1", "auc2", "delta", "IC 95 % ΔAUC", "z", "p_holm", "significativa",
                        "relevante (|Δ| ≥ 0,005)"]].rename(
    columns={"auc1": "AUC scikit-learn", "auc2": "AUC PySpark"}).round(5)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
d = de.set_index("modelo").loc[mo.ORDEN]
ax.errorbar(d["delta"], range(len(d)), xerr=[d["delta"] - d["delta_inf"], d["delta_sup"] - d["delta"]],
            fmt="o", color="#4C72B0", capsize=4)
for x in (-0.005, 0.005):
    ax.axvline(x, color="#C44E52", ls="--", lw=1)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(d)))
ax.set_yticklabels(d.index)
ax.invert_yaxis()
ax.set_xlabel("ΔAUC = AUC scikit-learn − AUC PySpark (IC 95 % DeLong)")
ax.set_title("Diferencias entre entornos (líneas rojas: margen de relevancia ±0,005)")
u.guardar_fig(fig, "10_delong_entornos")
plt.show()

# %% [markdown]
# Todos los intervalos, salvo el del SVM, caen dentro de la franja de ±0,005: las diferencias entre
# entornos son estadísticamente detectables en tres modelos, pero irrelevantes en la práctica.

# %% [markdown]
# ### Mapas de calor de DeLong entre modelos
#
# ![Mapas de calor de DeLong](figuras/08_delong_heatmaps.png)
#
# ### DeLong, McNemar y bootstrap pareado

# %%
pd.read_csv(u.RESULTADOS / "resumen_tres_pruebas.csv")

# %% [markdown]
# La tabla resume el capítulo 8. DeLong y el bootstrap coinciden en que el SVM es la única
# diferencia relevante entre entornos; McNemar, que no evalúa relevancia, rechaza la igualdad de
# errores en cinco de las seis comparaciones (todas salvo Naive Bayes). DeLong y el bootstrap
# coinciden también en que el gradient boosting supera al bosque dentro de cada entorno, mientras
# que McNemar es significativo en sentido contrario (el bosque comete menos errores en su umbral).
# En la regresión logística entre entornos, DeLong y el bootstrap del AUC no detectan diferencia,
# pero McNemar y el bootstrap de AUC-PR y F1 sí, aunque irrelevante (ΔAUC-PR = −0,0001; ΔF1 =
# 0,0006), por la pequeña diferencia de umbrales.
