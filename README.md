# Clasificación de préstamos de Lending Club: scikit-learn frente a PySpark

Tarea 1 (Proyecto Integrador de Aprendizaje Automático, sección 9.10 de las notas del curso de
Machine Learning) · Doctorado en Ingeniería · Camilo Molina Guerrero

Seis clasificadores (regresión logística, árbol de decisión, bosque aleatorio, gradient boosting,
SVM lineal y Naive Bayes) predicen si un préstamo de Lending Club termina en default. Cada modelo
se implementa con scikit-learn y con PySpark sobre los 1.348.059 préstamos cerrados, sin muestreo,
con la misma partición de prueba. Las diferencias de AUC se contrastan con la prueba de DeLong
(Holm), McNemar y bootstrap pareado, y las predicciones se interpretan con LIME.

**Libro publicado:** <https://camilomolinaguerrero-sudo.github.io/Clasificaci-n-de-pr-stamos-Lending-Club/>

## Estructura

```
├── README.md, requirements.txt
├── src/
│   ├── utils.py          rutas, semilla (42), hardware, utilidades
│   ├── features.py       variables idénticas en pandas y PySpark; preprocesamiento de cada entorno
│   ├── spark_utils.py    SparkSession con la configuración obligatoria; preprocesamiento distribuido
│   ├── modelado.py       GridSearchCV y CrossValidator, métricas, tiempos y puntos de control
│   ├── evaluacion.py     umbral de Youden en entrenamiento; métricas locales y distribuidas
│   ├── estadistica.py    DeLong rápido (Sun y Xu), McNemar, bootstrap pareado, Holm, Little, Cramér
│   ├── exp_spark.py      experimentos de rendimiento de Spark (un proceso por configuración)
│   ├── py2nb.py          convierte cuadernos/*.py (formato percent) en .ipynb y los ejecuta
│   └── ver_salidas.py    imprime las salidas de texto de un cuaderno
├── cuadernos/            fuente de los capítulos (formato # %%)
├── book/                 Jupyter Book: _config.yml, _toc.yml, capítulos ejecutados y figuras
└── data/
    ├── processed/        partición común (id, split) y lista de variables
    └── resultados/       métricas, tiempos, puntuaciones de prueba, pruebas estadísticas
```

## Reproducir

1. Descargar `accepted_2007_to_2018Q4.csv.gz` de
   [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en `data/raw/`.
2. Crear el entorno (Python 3.11, Java 17 o superior; R con `pROC` para validar DeLong):

```bash
python3.11 -m venv ~/.venvs/ml-lending
~/.venvs/ml-lending/bin/pip install -r requirements.txt
~/.venvs/ml-lending/bin/python -m ipykernel install --user --name ml-lending
```

3. Ejecutar los capítulos en orden (los de modelado tardan horas y guardan puntos de control en
   `data/resultados/`):

```bash
for c in cuadernos/*.py; do ~/.venvs/ml-lending/bin/python src/py2nb.py "$c" "book/$(basename "${c%.py}").ipynb" --ejecutar; done
```

4. Construir y publicar el libro:

```bash
~/.venvs/ml-lending/bin/jupyter-book build book/
~/.venvs/ml-lending/bin/ghp-import -n -p -f book/_build/html
```
