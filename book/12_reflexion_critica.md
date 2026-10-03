# 12. Reflexión crítica

Este capítulo responde las ocho preguntas de la sección 9.10.5 del enunciado con la evidencia de
los capítulos 5 a 11. Las cifras remiten a las tablas de esos capítulos.

## 12.1 ¿Qué entorno fue más rápido y por qué?

**scikit-learn, con claridad.** El entrenamiento con validación cruzada de los seis modelos tomó
1.967 s (33 min) en scikit-learn y 11.495 s (3,2 h) en PySpark, 5,8 veces más. scikit-learn fue más
rápido en cinco de los seis modelos, con cocientes entre 3,0 (árbol) y 19,5 (Naive Bayes). También
lo fue en el preprocesamiento desde el CSV (17,6 s frente a 34,5 s), en la predicción (menos de
0,4 s frente a 0,2-61 s) y en el cálculo del umbral y de las métricas (0,3-1,8 s frente a 5-533 s
por modelo). La única excepción, el SVM lineal (225 s en Spark frente a 616 s), no es una ventaja de
Spark: `liblinear` tarda en converger cerca de la solución degenerada de la pérdida *hinge*,
mientras que Spark corta su optimizador a las 100 iteraciones (capítulo 11).

Las razones se pueden separar con los experimentos del capítulo 11:

1. **El tamaño completo del conjunto cabe en memoria.** La matriz de entrenamiento ocupa 620 MB
   (1.078.447 × 72 en `float64`) y los datos cacheados en Spark, 151 MB. Con ese volumen,
   scikit-learn trabaja sobre arreglos contiguos de NumPy con código compilado y multihilo, sin
   serializar nada. Spark paga en cada pasada un costo fijo: programar tareas, mover datos entre la
   JVM y Python y combinar resultados parciales. El capítulo 11 lo muestra con nitidez: con 10.000
   filas, un ajuste de Spark tarda de 2 a 7 s, cuando a scikit-learn le basta una fracción de
   segundo. Ese costo fijo solo se diluye con muchos más datos.
2. **El caché hace viable a Spark, pero no lo vuelve más rápido que scikit-learn.** Sin
   `persist`, cada iteración de L-BFGS cuesta 4,9 s en lugar de 0,52 s (9,4 veces más), porque
   vuelve a leer y descomprimir el CSV. Todos los tiempos de Spark ya incluyen este beneficio.
3. **`CrossValidator` es secuencial.** Entrena una combinación tras otra, y cada ajuste de los
   modelos iterativos exige muchas pasadas distribuidas: el gradient boosting de Spark, con 100
   iteraciones dependientes y varias pasadas por iteración, consumió 8.805 s (77 % del total). En
   scikit-learn, `GridSearchCV(n_jobs=-1)` reparte hasta 10 ajustes completos entre los núcleos.
   Subir `parallelism` a 3 no ayudó (135 s frente a 140 s en la regresión logística), porque cada
   ajuste de Spark ya ocupa los 10 núcleos.
4. **La configuración de Spark está pensada para un clúster.** Con los datos repartidos en 400
   particiones, como el `spark.sql.shuffle.partitions` del enunciado, la validación cruzada fue 2,3
   veces más lenta en la regresión logística y 11,5 veces en el árbol que con 40 particiones. La
   memoria configurada (8 GB y fracciones 0,8/0,3) no tuvo efecto medible, porque los datos
   cacheados ocupan 151 MB.

La expectativa del enunciado, que PySpark superara en velocidad a scikit-learn, no se cumplió en
estas condiciones: un solo portátil y un conjunto que cabe holgadamente en memoria. Spark está
diseñado para repartir datos que no caben en una máquina entre muchas máquinas, y aquí no tuvo ni
lo uno ni lo otro.

## 12.2 ¿Cuál fue más preciso?

En los dos entornos el modelo con mejor desempeño discriminativo es el **gradient boosting**: AUC
de 0,7235 [0,7212; 0,7258] en scikit-learn y 0,7228 [0,7205; 0,7252] en PySpark, AUC-PR de 0,395 y
0,393 y F1 de 0,437 en el umbral de Youden. Le siguen el bosque aleatorio (0,7197 y 0,7181) y la
regresión logística (0,7171 en ambos). En el umbral de Youden, la mayor precisión (0,339 y 0,338) y
la mayor exactitud (0,686 y 0,683) corresponden al bosque, que marca menos préstamos como default.
La regresión logística, con 72 coeficientes y menos de 10 s de entrenamiento en scikit-learn, queda
a solo 0,006 del mejor modelo: la mayor parte de la señal predictiva es aproximadamente lineal en
las variables transformadas (tasa de interés, plazo, FICO, `dti`).

Entre entornos, scikit-learn es ligeramente más preciso en los tres modelos de árboles (ventajas de
0,0007 a 0,0027 de AUC), iguala a PySpark en la regresión logística y Naive Bayes y queda muy por
debajo en el SVM lineal (0,539 frente a 0,644). Ninguna de estas diferencias, salvo la del SVM (a
favor de PySpark), alcanza el margen de relevancia práctica de 0,005.

Un AUC de 0,72 es un resultado modesto pero realista para el riesgo de crédito con información de
originación: el capítulo 9 muestra préstamos de perfil impecable que incumplieron y préstamos de
alto riesgo que se pagaron. Lending Club ya fijaba su tasa de interés con un modelo de riesgo, de
modo que buena parte de la información útil está resumida en `int_rate`, que por sí sola alcanza un
AUC de 0,683 (capítulo 3).

## 12.3 ¿Qué diferencias de AUC son estadísticamente significativas y cuáles son relevantes?

El margen de relevancia práctica se fijó antes de ver los resultados en |ΔAUC| ≥ 0,005, el
equivalente a 0,01 en el coeficiente de Gini (capítulo 8).

**Entre entornos** (familia de 6 comparaciones, Holm):

| Modelo | ΔAUC (sklearn − Spark) | IC 95 % | p ajustado | Significativa | Relevante |
|---|---|---|---|---|---|
| Regresión logística | −0,00001 | [−0,00003; 0,00001] | 0,94 | no | no |
| Árbol | 0,0027 | [0,0017; 0,0037] | < 10⁻⁶ | sí | no |
| Bosque | 0,0015 | [0,0011; 0,0020] | < 10⁻⁶ | sí | no |
| Gradient boosting | 0,0007 | [0,0002; 0,0011] | 0,009 | sí | no |
| SVM lineal | −0,1045 | [−0,1083; −0,1007] | < 10⁻⁶ | sí | **sí** |
| Naive Bayes | 0,0000 | [0,0000; 0,0000] | 0,94 | no | no |

Las tres diferencias significativas de los modelos de árboles son reales en el sentido estadístico
(no se deben al azar del conjunto de prueba), pero de una magnitud que no cambiaría ninguna
decisión. Se detectan porque los AUC estimados de ambos entornos están muy correlacionados (0,91 a
0,98), lo que reduce el error estándar de la diferencia a valores entre 0,0002 y 0,0005. Solo el
SVM difiere de manera relevante.

**Entre modelos** (dos familias de 15 pares): las 30 diferencias son significativas tras Holm. Son
relevantes todas las que involucran al árbol, a Naive Bayes o al SVM, y la del gradient boosting
frente a la regresión logística (0,0064 en scikit-learn y 0,0057 en PySpark). No lo son la del
bosque frente a la regresión logística (0,0026 y 0,0011) ni la del gradient boosting frente al
bosque (0,0038 y 0,0047). En la práctica hay un grupo de tres modelos cercanos, encabezado por el
gradient boosting, y tres modelos claramente inferiores.

## 12.4 ¿Qué diferencias de implementación explican las discrepancias?

Los dos entornos recibieron la misma matriz de 72 columnas (verificado en el capítulo 5: medias de
columna iguales hasta 10⁻¹²), la misma partición y los mismos espacios de búsqueda. Lo que difiere
es cómo cada biblioteca construye el modelo:

| Fuente de discrepancia | scikit-learn | PySpark | Modelos afectados |
|---|---|---|---|
| Poda posterior del árbol | no poda | une hojas hermanas que predicen la misma clase | árbol, bosque |
| Cortes de los árboles | todos los valores distintos de cada variable | a lo sumo `maxBins − 1` = 31 cortes por variable, elegidos con cuantiles de una muestra | árbol, bosque, gradient boosting |
| Variables candidatas por nodo | ⌊√72⌋ = 8 | `"sqrt"` usaría ⌈√72⌉ = 9; se fijó `"8"` | bosque |
| Bootstrap | muestra multinomial exacta | pesos Poisson(1) por fila | bosque |
| Hojas del boosting | paso de Newton; predicción inicial = log-odds | media de los gradientes; primer árbol ajustado a las etiquetas | gradient boosting |
| Regularización | $C\sum\ell_i + \frac12\lVert w\rVert^2$ | $\frac1n\sum\ell_i + \frac\lambda2\lVert w\rVert^2$; con `standardization=False` | regresión logística, SVM |
| Intercepto | penalizado en `liblinear` (SVM), no en `lbfgs` | nunca penalizado | SVM |
| $\lambda$ efectivo en la validación cruzada | $1{,}5\lambda$ ($C$ fijado con $n$, pliegues de $2n/3$) | $\lambda$ | regresión logística, SVM |
| Optimizador | `lbfgs` (≤ 1.000 iteraciones), `liblinear` dual (≤ 5.000) | L-BFGS / OWL-QN (≤ 100 iteraciones) | regresión logística, SVM |
| Pliegues de validación | `StratifiedKFold` barajado | asignación aleatoria por fila, sin estratificar | todos (solo en la selección) |
| Escalado | desviación poblacional ($n$) | desviación muestral ($n-1$) | despreciable |

La evidencia de los capítulos 7, 8 y 11 permite atribuir cada discrepancia observada:

* **Regresión logística y Naive Bayes** no muestran discrepancias: ambos tienen un óptimo único
  (problema convexo y bien condicionado, o solución cerrada) y los dos entornos lo alcanzan. La
  diferencia entre el $\lambda$ de la validación cruzada (1,5λ frente a λ) y los distintos
  optimizadores no se notan porque la regularización es despreciable con un millón de filas.
* **Árbol y bosque**: la causa principal no es la que se esperaba. Se suponía que la
  discretización (`maxBins` = 32) explicaría la brecha, pero subir `maxBins` a 512 apenas cambia el
  AUC (de 0,649 a 0,655 a profundidad 5). La causa es la **poda**: al terminar de construir un
  árbol, Spark une las hojas hermanas que predicen la misma clase, sin parámetro público para
  desactivarlo. Con 80 % de préstamos pagados casi todas las hojas predicen "pagado", así que la
  poda borra cortes que no cambian la clase pero sí la probabilidad. A profundidad 2 y 3, el árbol
  de Spark queda en una sola hoja (AUC 0,5 frente a 0,671 y 0,683); a profundidad 5, en 6 hojas
  frente a 32 (0,649 frente a 0,695); a profundidad 10, en 558 frente a 958 (0,7015 frente a
  0,7041); y a profundidad 15 el número de hojas es el mismo y Spark incluso supera a scikit-learn
  (0,680 frente a 0,676). Como la búsqueda eligió profundidades de 10 y 15, en los modelos finales la
  diferencia es pequeña: 0,0027 de AUC en el árbol y 0,0015 en el bosque, donde se suma el
  bootstrap aproximado por Poisson.
* **Gradient boosting**: sus árboles son de regresión, la poda no le afecta y las demás
  diferencias (discretización y forma de calcular las hojas) dejan una diferencia final de 0,0007.
* **SVM lineal**: aquí la discrepancia es enorme (0,10 de AUC) y la causa es la **función de
  pérdida combinada con el optimizador**. Con la pérdida *hinge*, fijada en scikit-learn para
  igualar la de Spark (sección 6.7), y clases tan solapadas y desbalanceadas, el óptimo está
  pegado a la solución trivial $w = 0$, $b = -1$ (todos los préstamos "pagados"). En el capítulo 11,
  con $\lambda = 10^{-5}$, la función objetivo de la solución de `liblinear` (39.956,8) es
  prácticamente la de la solución trivial (39.956,7), aunque sus parámetros no son los triviales
  ($\lVert w \rVert$ = 0,48, $b$ = −0,365, porque `liblinear` además penaliza el intercepto): cerca
  del óptimo la función objetivo es casi plana y la dirección de $w$, lo único que ordena las
  puntuaciones, queda mal determinada (AUC de prueba entre 0,46 y 0,54). Spark limita OWL-QN a 100
  iteraciones; su solución queda cerca de la trivial en escala ($\lVert w \rVert$ = 0,046,
  $b$ = −1,026) pero conserva una dirección útil (AUC 0,644 en el capítulo 7). Paradójicamente, el
  entorno que resuelve el problema de optimización de forma menos exacta obtiene el mejor
  clasificador. Con pesos que equilibran las clases, el SVM *hinge* alcanza un AUC de 0,717 en ambos
  entornos, y con la pérdida *squared hinge* también (capítulo 11). La pérdida *hinge* sin pesos,
  elegida para igualar los entornos, es una mala elección para este problema, aunque garantiza que
  ambos minimicen la misma función.
* **Escalado**: la desviación muestral frente a la poblacional no tiene efecto medible (diferencia
  relativa de 4,6 × 10⁻⁷).

## 12.5 Limitaciones de DeLong y aporte de McNemar y del bootstrap

La prueba de DeLong es la adecuada para la pregunta "¿ordenan igual de bien dos modelos las mismas
observaciones?", pero en este diseño tiene cuatro limitaciones:

1. **Solo mide la incertidumbre del muestreo de prueba.** El error estándar de DeLong trata los
   modelos como fijos y pregunta cuánto variaría el AUC con otro conjunto de prueba de la misma
   población. No incorpora la variabilidad del entrenamiento: otra semilla del bosque, otros
   pliegues de validación cruzada u otra partición entrenamiento/prueba darían modelos algo
   distintos. Con 269.612 observaciones el error estándar de una diferencia pareada es muy pequeño
   (0,0002 a 0,0005) y puede ser menor que la variabilidad entre semillas de un modelo estocástico,
   así que una diferencia "significativa" entre dos bosques no prueba que un algoritmo sea mejor que
   el otro en general.
2. **Supone observaciones independientes.** Los préstamos de un mismo estado o cohorte comparten
   factores (el ciclo económico de 2016-2017, por ejemplo); la dependencia hace que el error
   estándar real sea mayor que el que calcula la prueba.
3. **El AUC resume todos los umbrales.** Dos modelos con el mismo AUC pueden diferir mucho en el
   umbral operativo (aprobar o rechazar un crédito), y una diferencia de AUC puede venir de
   regiones de la curva ROC (tasas de falsos positivos muy altas) que no interesan al negocio.
4. **Es asintótica.** Con este tamaño de muestra la aproximación normal es excelente; no es una
   limitación práctica aquí, pero sí lo sería con pocas decenas de defaults.

Las pruebas complementarias atacan las limitaciones 3 y 4 y añaden otra mirada:

* **McNemar** compara decisiones concretas en el umbral elegido con entrenamiento: cuenta en
  cuántos préstamos acierta uno y falla el otro. Trata igual un falso positivo (rechazar a un buen
  pagador) que un falso negativo (aprobar un default), aunque en crédito sus costos son muy
  distintos, y con 80 % de préstamos pagados sus conclusiones están dominadas por la clase
  mayoritaria.
* **El bootstrap pareado** no necesita una fórmula de varianza: da intervalos para cualquier
  métrica, incluidas AUC-PR y F1, que se concentran en la clase minoritaria y no tienen una prueba
  analítica estándar. Para el AUC sirve además como verificación independiente de DeLong.

Ninguna de las tres pruebas resuelve la limitación 1; para eso habría que repetir el
entrenamiento con varias semillas y particiones (por ejemplo, la validación cruzada 5×2 de
Dietterich, 1998), lo que multiplicaría el costo de cómputo.

**¿Coinciden las tres pruebas?** En lo esencial sí, con discrepancias que se explican por lo que mide
cada una (tabla de la sección 8.6):

* DeLong y el bootstrap del AUC coinciden en las ocho comparaciones complementarias (las 6 entre
  entornos y las 2 entre los dos mejores modelos), con intervalos que difieren a lo sumo en 0,0001.
  Es la comprobación esperada: estiman la misma cantidad, y con 269.612 observaciones la
  aproximación asintótica de DeLong es excelente.
* McNemar, que no evalúa relevancia, rechaza la igualdad de errores en cinco de las seis
  comparaciones entre entornos, y sus discrepancias con DeLong son instructivas. Entre los dos
  entornos de la regresión logística encuentra una diferencia muy significativa (1.334 frente a 336
  préstamos discordantes) aunque los modelos son idénticos en AUC: los umbrales de Youden difieren
  en 0,002 y eso basta para cambiar la decisión de 1.670 préstamos cercanos al umbral (el bootstrap
  de AUC-PR y F1 también detecta esa diferencia, irrelevante en magnitud). Entre el gradient
  boosting y el bosque, DeLong favorece al primero (mejor ordenamiento) y McNemar al segundo (menos
  errores: 31,4 % frente a 34,7 % en scikit-learn y 31,7 % frente a 34,5 % en PySpark), porque el
  bosque, con su umbral, marca menos préstamos como default, y McNemar, al contar igual todos los
  errores, premia acertar en la clase mayoritaria.
* El bootstrap de AUC-PR y F1, las métricas de la clase minoritaria, confirma la ventaja del
  gradient boosting sobre el bosque (ΔAUC-PR de 0,007, por encima del margen) y muestra que en F1
  los entornos son indistinguibles para el bosque y el gradient boosting.

La conclusión de fondo es la misma con DeLong y el bootstrap: salvo el SVM, los dos entornos
producen modelos equivalentes, y el gradient boosting es el mejor en capacidad de ordenamiento por un
margen pequeño pero real. McNemar recuerda que "mejor ordenamiento" no implica "menos errores" en un
umbral dado.

## 12.6 ¿A partir de qué volumen de datos PySpark comienza a superar a scikit-learn?

El capítulo 11 midió un ajuste de tres modelos con 1 % a 100 % del entrenamiento y con el
entrenamiento replicado 2 y 4 veces (hasta 4,3 millones de filas):

| Modelo | Cociente PySpark / scikit-learn con 1,08 M filas | PySpark más rápido desde |
|---|---|---|
| Árbol (profundidad 10) | 1,19 | **≈ 2,2 millones de filas** (11,7 s frente a 14,0 s) |
| Regresión logística | 5,76 (3,6 con 4,3 M) | no ocurre hasta 4,3 M |
| Bosque (50 árboles, profundidad 10) | 3,20 (2,3 con 4,3 M) | no ocurre hasta 4,3 M |

La respuesta depende de cómo paraleliza scikit-learn cada algoritmo. El árbol de scikit-learn usa un
solo núcleo; Spark reparte cada pasada entre los 10, y su costo marginal por millón de filas
(3,3 s) es la mitad del de scikit-learn (6,7 s). En cuanto el volumen diluye el costo fijo de Spark,
lo supera: desde unos 2 millones de filas en este equipo. La regresión logística (`lbfgs` con
operaciones vectorizadas) y el bosque (`n_jobs=-1`) ya aprovechan los 10 núcleos en scikit-learn, y
su costo marginal (0,9 y 10,4 s por millón de filas) es menor que el de Spark (2,7 y 20,9 s): en
una sola máquina, Spark no los alcanzaría con más datos.

El cruce que importa es otro: **la memoria**. scikit-learn necesita la matriz completa en RAM (2,5 GB
para 4,3 millones de filas), y hacia los 100 millones de filas (unos 60 GB) ya no cabría en los 24 GB
del equipo, mientras que Spark puede derramar a disco o repartir los datos entre varias máquinas. Con
el conjunto de este proyecto (1,35 millones de préstamos), scikit-learn es la herramienta adecuada;
PySpark se justifica cuando los datos no caben en una máquina o cuando ya viven en un clúster.

## 12.7 ¿Qué aporta LIME y cuáles son sus limitaciones en entornos distribuidos?

**Aporte.** LIME mostró que los dos gradient boosting, dos implementaciones del mismo algoritmo,
se apoyan en razones parecidas: en el falso negativo coinciden 8 de las 10 condiciones más
influyentes y en el falso positivo 6, con pesos similares. Las condiciones que dominan son tasa de
interés, plazo, propósito, `dti`, monto, ingreso y que la antigüedad laboral esté informada. Son
las que el EDA había señalado, incluido el faltante informativo de `emp_length` del capítulo 4. Los
dos errores analizados no provienen de reglas espurias, sino de préstamos cuyo desenlace contradijo
un perfil típico: un préstamo de 10.000 dólares a 5,3 % con FICO de 774 e ingreso de 350.000
dólares que incumplió, y otro a 24,5 % y 60 meses que se pagó. Con dos instancias no se puede
generalizar a todas las decisiones, pero para un analista de crédito este tipo de verificación vale
más que una métrica global.

**Limitaciones generales.** El modelo lineal local explica solo la mitad de la variación de la
probabilidad en el vecindario ($R^2$ de 0,45 a 0,54). Con cinco semillas, solo 5 de las 10
condiciones se repiten. Las perturbaciones ignoran las correlaciones entre variables y tratan cada
columna one-hot por separado, y por eso aparecen condiciones poco informativas como
`addr_state_OR=0`.

**Limitaciones en entornos distribuidos.** LIME es un algoritmo de memoria local. Con PySpark hubo
que hacer dos cosas. Primero, traer al driver una muestra de 50.000 filas como datos de referencia,
porque el millón completo no debe salir del clúster; eso cambió ligeramente los cuartiles y, por
tanto, algunas condiciones (`int_rate > 16.01` frente a `> 15.99`). Segundo, envolver el modelo en
una función que convierte cada lote de 5.000 perturbaciones en un DataFrame de Spark, lanza un
trabajo distribuido y devuelve las probabilidades. Cada explicación tardó entre 1,1 y 1,9 s, entre
55 y 65 veces más que con scikit-learn. Para explicar miles de préstamos habría que distribuir el
explicador a los ejecutores, algo que LIME no ofrece.

## 12.8 ¿Qué efecto tuvo cada condición obligatoria en el rendimiento?

| Condición | Efecto observado |
|---|---|
| **Conjunto completo, sin muestreo** | En scikit-learn, el DataFrame crudo ocupó 6,3 GB y la matriz de diseño 620 MB: caben en 24 GB y los 33 min de entrenamiento son manejables. En Spark multiplicó el costo de los modelos iterativos (gradient boosting: 2,4 h). En lo estadístico, las 269.612 observaciones de prueba dan errores estándar de 0,0002 a 0,0005 para las diferencias de AUC, de modo que diferencias triviales resultan significativas; de ahí la necesidad del margen de relevancia. |
| **Sin `.toPandas()` ni `.collect()` antes de entrenar** | Obligó a hacer en Spark todo el preprocesamiento, la verificación de equivalencia (con agregados), el umbral de Youden (con cuantiles y conteos) y las métricas. Esas etapas posteriores costaron entre 5 y 533 s por modelo, frente a 0,3-1,8 s en scikit-learn. La transferencia final de `id`, `default` y puntuación tomó 0,1-0,2 s (55 s en el bosque, probablemente por un recálculo). |
| **Partición común en archivo (sin `randomSplit`)** | Costo despreciable (una unión por `id`) y beneficio decisivo: permitió emparejar las 269.612 observaciones y aplicar DeLong, McNemar y el bootstrap entre entornos. |
| **Caché tras el `VectorAssembler`** | Indispensable: sin caché cada iteración cuesta 9,4 veces más (4,9 s frente a 0,52 s), y la validación cruzada de la regresión logística habría pasado de menos de 4 min a más de media hora. |
| **`ParamGridBuilder` + `CrossValidator` (sin bucles)** | Garantiza la misma selección por AUC con 3 pliegues, pero entrena las combinaciones en serie; `parallelism=3` no aportó ganancia. Es la principal razón de que el gradient boosting tarde 2,4 h. |
| **Sesión: 400 particiones de shuffle** | Con los datos en 400 particiones la validación cruzada fue 2,3 veces (regresión logística) y 11,5 veces (árbol) más lenta que con 40; por eso los datos se repartieron en 40, manteniendo la configuración de la sesión. |
| **Sesión: memoria (8 GB, fracciones 0,8/0,3)** | Sin efecto medible: con 2 GB o con las fracciones por defecto los tiempos cambian unos segundos, porque los datos cacheados ocupan 151 MB. |
| **`GridSearchCV` sin `Pipeline`** | El preprocesamiento se ajusta una vez (no en cada pliegue), lo que ahorra tiempo e introduce una fuga mínima en la validación cruzada (mediana y escala calculadas con todo el entrenamiento), despreciable con un millón de filas. |
| **`loss="hinge"` y $C = 1/(\lambda n)$** | Igualan formalmente las funciones objetivo, pero la pérdida *hinge* sin pesos lleva al SVM a una solución casi degenerada (AUC 0,539 en scikit-learn); con clases balanceadas el mismo SVM alcanza 0,717. |

## 12.9 Conclusión

Con 1,35 millones de préstamos y la información disponible al originarlos, el mejor modelo de los
seis es el **gradient boosting**, con un AUC de 0,72 en ambos entornos. La regresión logística queda
a 0,006 y el bosque aleatorio entre ambos. Las dos bibliotecas producen modelos equivalentes en la
práctica: las diferencias de AUC entre entornos son menores que 0,003 salvo en el SVM lineal, cuya
discrepancia se debe a la pérdida *hinge* y no al entorno. Las diferencias de implementación más
interesantes resultaron ser la poda de hojas de los árboles de Spark y la degeneración del SVM
*hinge*; ninguna de las dos estaba anticipada en el enunciado, y ambas se identificaron con
experimentos dirigidos.

En velocidad, scikit-learn fue 5,8 veces más rápido en el entrenamiento con validación cruzada.
PySpark solo superó a scikit-learn en el árbol de decisión y a partir de unos 2 millones de filas.
La condición que más afectó el rendimiento de Spark fue el número de particiones, seguida del caché,
sin el cual el entrenamiento iterativo sería inviable.

Las conclusiones tienen tres limitaciones. Primero, todo se midió en una sola máquina; en un clúster
los tiempos de Spark serían otros. Segundo, cada modelo se entrenó una vez, con una semilla, de modo
que las pruebas estadísticas no incorporan la variabilidad del entrenamiento. Tercero, la población
de préstamos cerrados tiene un sesgo de selección por cohorte (las cohortes recientes solo aportan
sus defaults tempranos y sus prepagos), que limita la extrapolación a préstamos futuros, aunque no
afecta la comparación entre modelos y entornos.
