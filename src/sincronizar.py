"""Actualiza el texto (celdas markdown) de un cuaderno ya ejecutado sin volver a ejecutarlo.

Regenera el cuaderno desde la fuente `cuadernos/*.py` y copia las salidas de la versión ejecutada
en cada celda de código cuyo contenido no cambió. Si alguna celda de código cambió o no tiene
pareja, se detiene sin escribir nada (hay que volver a ejecutar el cuaderno).

Uso: python src/sincronizar.py cuadernos/06_modelado_sklearn.py book/06_modelado_sklearn.ipynb
"""
import sys
from pathlib import Path

import nbformat

from py2nb import convertir

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
nuevo = convertir(src)
viejo = nbformat.read(dst, as_version=4)
codigo_viejo = [c for c in viejo.cells if c.cell_type == "code"]
codigo_nuevo = [c for c in nuevo.cells if c.cell_type == "code"]
if len(codigo_viejo) != len(codigo_nuevo):
    sys.exit(f"ERROR: {len(codigo_nuevo)} celdas de código en la fuente y {len(codigo_viejo)} en el cuaderno")
for v, n in zip(codigo_viejo, codigo_nuevo):
    if v.source.strip() != n.source.strip():
        sys.exit("ERROR: cambió una celda de código; hay que ejecutar de nuevo:\n" + n.source[:300])
    n.outputs, n.execution_count = v.outputs, v.execution_count
nuevo.metadata = viejo.metadata
nbformat.write(nuevo, dst)
print("OK", dst, f"({sum(c.cell_type == 'markdown' for c in nuevo.cells)} celdas de texto)")
