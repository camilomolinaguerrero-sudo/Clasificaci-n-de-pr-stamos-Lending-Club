# Default en préstamos de Lending Club: scikit-learn frente a PySpark

**Tarea 1 · Proyecto Integrador de Aprendizaje Automático** (sección 9.10 de las notas del curso)
· Doctorado en Ingeniería · Camilo Molina Guerrero

---

## Objetivo

Construir modelos de clasificación supervisada que predigan si un préstamo emitido por Lending Club
termina en **default** (*Charged Off*, 1) o se paga por completo (*Fully Paid*, 0), y comparar seis
modelos implementados en **scikit-learn** y en **PySpark** sobre el conjunto de datos completo,
sin muestreo. La comparación tiene tres dimensiones:

1. **Desempeño**: AUC ROC (criterio de selección), AUC-PR, accuracy, precisión, recall, F1 y matriz
   de confusión en un conjunto de prueba común.
2. **Significancia**: prueba de DeLong para las diferencias de AUC, con corrección de Holm, y
   pruebas complementarias de McNemar y de bootstrap pareado.
3. **Costo computacional**: tiempos de preprocesamiento, entrenamiento con validación cruzada,
   predicción y transferencia de resultados, con la configuración de Spark exigida.

Las predicciones se interpretan con LIME.

## Datos

*Lending Club Loan Data*, archivo `accepted_2007_to_2018Q4.csv.gz` (Kaggle,
[wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)):
2.260.701 préstamos emitidos entre 2007 y 2018, con 151 columnas. Se modelan los **1.348.059
préstamos cerrados** (todos los que tienen desenlace conocido), de los cuales 19,98 % terminaron en
default. La partición común reserva 269.612 préstamos (20 %, estratificado, semilla 42) como
conjunto de prueba para todos los modelos de ambos entornos.

## Resumen de resultados

| Aspecto | scikit-learn | PySpark |
|---|---|---|
| Mejor modelo (AUC de prueba, IC 95 % DeLong) | Gradient boosting: 0,7235 [0,7212; 0,7258] | Gradient boosting: 0,7228 [0,7205; 0,7252] |
| Bosque aleatorio · regresión logística · árbol | 0,7197 · 0,7171 · 0,7041 | 0,7181 · 0,7171 · 0,7014 |
| Naive Bayes · SVM lineal (*hinge*) | 0,6525 · 0,5391 | 0,6525 · 0,6436 |
| F1 del mejor modelo (umbral de Youden en entrenamiento) | 0,437 | 0,437 |
| Entrenamiento + validación cruzada de los 6 modelos | 1.967 s (33 min) | 11.495 s (3,2 h) |
| Preprocesamiento desde el CSV | 17,6 s | 34,5 s |

* **Precisión.** Los dos entornos producen modelos equivalentes: las diferencias de AUC entre
  entornos son significativas en tres modelos de árboles (prueba de DeLong, Holm), pero menores que
  el margen de relevancia de 0,005. La única diferencia relevante es la del SVM lineal, cuya pérdida
  *hinge* tiene, en estos datos, un óptimo casi degenerado.
* **Velocidad.** scikit-learn fue 5,8 veces más rápido. PySpark solo lo supera en el árbol de
  decisión y a partir de unos 2 millones de filas; el número de particiones fue la condición que más
  pesó en su rendimiento.
* **Implementación.** La principal diferencia entre los árboles de ambos entornos no es la
  discretización (`maxBins`), sino la poda de hojas con la misma clase que aplica Spark, que en este
  problema desbalanceado borra cortes útiles en los árboles poco profundos.
* **Interpretabilidad.** LIME muestra que ambos gradient boosting se apoyan en la tasa de interés, el
  plazo, la capacidad de pago y el historial; en Spark cada explicación es unas 60 veces más lenta.

## Estructura del libro

| Parte | Capítulos |
|---|---|
| Análisis exploratorio | 1. Carga y visión general · 2. Análisis unidimensional · 3. Análisis bidimensional · 4. Valores faltantes · Resumen ejecutivo del EDA |
| Preprocesamiento y modelado | 5. Partición común y preprocesamiento en ambos entornos · 6. Modelado con scikit-learn · 7. Modelado con PySpark |
| Evaluación y comparación | 8. DeLong, McNemar y bootstrap pareado · 9. LIME · 10. Comparación de resultados · 11. Experimentos complementarios |
| Discusión | 12. Reflexión crítica · 13. Referencias |

## Entorno y reproducibilidad

Todos los tiempos se midieron en un portátil **Apple M5** (10 núcleos: 4 de rendimiento y 6 de
eficiencia) con **24 GB** de memoria unificada, macOS, Python 3.11.16, scikit-learn 1.9.1, PySpark
4.2.0 en modo local (`local[*]`) con Java 25 y la configuración de sesión del enunciado
(`spark.sql.shuffle.partitions` y `spark.default.parallelism` 400, 8 GB de memoria,
`memory.fraction` 0,8 y `storageFraction` 0,3); los datos de entrenamiento cacheados se reparten en
40 particiones (justificación en el capítulo 7 y medición en el capítulo 11). Semilla global: 42. El código está en `src/` y los capítulos se generan desde `cuadernos/`; el
[README del repositorio](https://github.com/camilomolinaguerrero-sudo/Clasificaci-n-de-pr-stamos-Lending-Club)
explica cómo reproducir todo desde el CSV de Kaggle.
