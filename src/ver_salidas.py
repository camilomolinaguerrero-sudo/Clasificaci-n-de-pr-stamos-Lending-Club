"""Imprime las salidas de texto de un cuaderno ejecutado (para revisión)."""
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        txt = ""
        if o.output_type == "stream":
            txt = o.text
        elif o.output_type in ("execute_result", "display_data"):
            txt = o.data.get("text/plain", "") if "image/png" not in o.data else "[figura]"
        elif o.output_type == "error":
            txt = "ERROR " + o.ename + ": " + o.evalue
        if txt:
            print(f"--- celda {i}\n{txt[:15000]}")
