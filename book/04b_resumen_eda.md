# Resumen ejecutivo del EDA

Esta sección sintetiza los capítulos 1 a 4 (sección 9.10.4.1.5 del enunciado) y fija las decisiones
que guían el preprocesamiento.

## Calidad de los datos

| Aspecto | Hallazgo |
|---|---|
| Tamaño | 2.260.701 filas y 151 columnas en el CSV; 33 filas de totales o sin `loan_status` (dos de ellas al final del archivo) que no son préstamos. |
| Población | 1.348.059 préstamos cerrados (Fully Paid o Charged Off, incluidos los 2.749 *Does not meet the credit policy*). Los 912.609 préstamos vigentes al corte se excluyen porque su desenlace no se conoce. |
| Fuga de información | 40 columnas solo existen después de la originación (pagos, recuperaciones, último pago, FICO reciente, *hardship*, *settlement*) y no pueden usarse. |
| Faltantes | 42 columnas superan 70 % de faltantes; un bloque de unas 20 variables del buró falta en todos los préstamos de 2007-2011 y en el 52 % de los de 2012 (faltante estructural por fecha); la prueba de Little rechaza MCAR (p ≈ 10⁻¹⁷⁵ incluso con una muestra de 5.000 préstamos). |
| Valores inválidos | `dti` < 0 o > 100 (535 casos, con un máximo de 999) se tratan como faltantes. |
| Atípicos | Ingresos y saldos con colas muy largas (asimetría de hasta 52); son prestatarios reales, no errores, y se tratan con log(1 + x), no eliminándolos. |
| Sesgo de cohorte | Solo se observan los préstamos ya cerrados: las cohortes de 2016-2017 tienen más default (23 %) y la de 2018 menos (15,8 %) por la censura del desenlace. |

## Variables más prometedoras

| Variable | Evidencia | Dirección |
|---|---|---|
| `int_rate` | AUC univariado 0,683; r punto-biserial 0,26 | más tasa, más default |
| `term` | AUC 0,594; 39 % de los defaults a 60 meses frente a 20 % de los pagados | plazo largo, más default |
| `fico_range_high` | AUC 0,407 (0,593 invertido) | más FICO, menos default |
| `dti` | AUC 0,577 | más endeudamiento, más default |
| `acc_open_past_24mths` | AUC 0,571 | más cuentas recientes, más default |
| `avg_cur_bal`, `mort_acc`, `annual_inc` | AUC 0,44-0,45 | más patrimonio, menos default |
| `grade` | V de Cramér 0,26; default de 6,0 % (A) a 49,7 % (G) | redundante con `int_rate` |
| `verification_status`, `home_ownership`, `purpose`, `addr_state` | V de Cramér entre 0,05 y 0,09 | asociación débil |
| `emp_length` faltante | default de 26,9 % frente a 19,5 % con dato | faltante informativo |

## Problemas detectados

1. **Desbalance moderado**: 19,98 % de defaults (4 pagados por cada default). La exactitud es
   engañosa y, con 20 % de defaults, es de esperar que el umbral de 0,5 deje casi todos los
   préstamos como "pagados" (se confirma en los capítulos 6 y 7).
2. **Multicolinealidad**: ocho pares con |r| > 0,7, todos con explicación contable (FICO bajo y
   alto, monto y cuota, utilizaciones, saldos y límites); `grade` explica el 90,7 % de la varianza
   de `int_rate`.
3. **Faltantes sistemáticos**: dependen del año de emisión (MAR) y, en `emp_length` y
   `mths_since_recent_inq`, se asocian con el desenlace.
4. **Poder discriminante individual moderado**: ninguna variable separa las clases; el mejor AUC
   univariado es 0,68. Se espera un AUC multivariado moderado, del orden de 0,70.

## Decisiones para el preprocesamiento

| Decisión | Detalle |
|---|---|
| Variables | 20 numéricas, 4 indicadores binarios y 6 categóricas (lista en el capítulo 4, sección 4.5) |
| Eliminadas | fuga de información, identificadores y texto libre, columnas con más de 70 % de faltantes, variables disponibles solo desde 2015, y `installment`, `fico_range_low`, `grade`, `sub_grade`, `bc_util`, `percent_bc_gt_75`, `tot_cur_bal`, `revol_bal` y `total_acc` por redundancia; además, otras 31 columnas del buró y de la solicitud (`num_*`, `mo_sin_*`, `tot_hi_cred_lim`, `bc_open_to_buy`...) no se seleccionan por redundantes o poco informativas |
| Transformaciones | log(1 + x) en `annual_inc`, `total_rev_hi_lim` y `avg_cur_bal`; estandarización del bloque numérico |
| Faltantes | mediana de entrenamiento; indicadores `hist_morosidad`, `emp_length_faltante`, `consulta_faltante` y `buro_faltante` |
| Categóricas | `ANY`/`NONE`/`OTHER` unificadas; categorías con menos de 1 % en entrenamiento agrupadas (`purpose` → `other`, `addr_state` → `OTROS`); one-hot |
| Desbalance | partición y validación cruzada estratificadas; selección por AUC ROC; umbral de decisión optimizado con entrenamiento (índice de Youden, capítulos 6 y 7); sin sobremuestreo ni submuestreo, porque el enunciado exige el conjunto completo y el AUC no depende de la proporción de clases |
