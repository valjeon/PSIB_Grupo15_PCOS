"""Muestra los casos donde el conteo de folículos más se equivoca (correr después de
comparar_segmentacion.py).

  fila 1: las PCOS con MENOS folículos  -> ¿se pierden folículos o la imagen no los muestra?
  fila 2: las sanas con MÁS folículos   -> ¿qué está tomando como folículo?

Salida: reports/seg_peores.png (rojo = folículo, amarillo = descartado, celeste = abanico)
y la lista de archivos en la terminal, para abrirlos y mirarlos en tamaño original.

Uso:
    python3 ver_casos.py                         # combinación mediana+estirar
    python3 ver_casos.py --filtro lee --contraste clahe --n 8
    python3 ver_casos.py --contraste-min 20      # mismos parámetros que usaste al comparar
"""
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

from comparar_preprocesamiento import hoja
from segmentacion import segment_detalle, superponer

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="mediana")
    ap.add_argument("--contraste", default="estirar")
    ap.add_argument("--metodo", default="multiotsu")
    ap.add_argument("--n", type=int, default=6, help="imágenes por fila")
    ap.add_argument("--area-min", type=int, default=None)
    ap.add_argument("--contraste-min", type=float, default=None)
    ap.add_argument("--circ-min", type=float, default=None)
    ap.add_argument("--sigma-umbral", type=float, default=None)
    ap.add_argument("--reumbral", action="store_true",
                    help="(ya es el comportamiento por defecto; se acepta por compatibilidad)")
    ap.add_argument("--sin-reumbral", action="store_true",
                    help="desactivar el re-umbral con Otsu local de las regiones grandes o alargadas")
    args = ap.parse_args()
    kw = {k: v for k, v in dict(area_min=args.area_min, contraste_min=args.contraste_min,
                                circ_min=args.circ_min, sigma_umbral=args.sigma_umbral,
                                reumbral=False if args.sin_reumbral else None).items()
          if v is not None}

    with open(REPORTES / "seg_conteos.csv", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f)
                 if r["filtro"] == args.filtro and r["contraste"] == args.contraste]
    if not filas:
        raise SystemExit(f"No hay resultados de {args.filtro}+{args.contraste} en "
                         "seg_conteos.csv. Corré antes comparar_segmentacion.py con esa combinación.")
    pcos = sorted((r for r in filas if r["label"] == "1"), key=lambda r: int(r["n_foliculos"]))
    sanas = sorted((r for r in filas if r["label"] == "0"), key=lambda r: -int(r["n_foliculos"]))
    elegidas = [pcos[:args.n], sanas[:args.n]]

    grilla = []
    for titulo, grupo in zip(["PCOS con menos folículos", "Sanas con más folículos"], elegidas):
        print(f"\n{titulo}:")
        celdas = []
        for r in grupo:
            d = segment_detalle(ROOT / r["path"], filtro=args.filtro, contraste=args.contraste,
                                metodo=args.metodo, **kw)
            print(f"  {d['n_foliculos']:3d}  {r['path']}")
            img = superponer(d, ver_rechazos=True)
            cv2.putText(img, str(d["n_foliculos"]), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2, cv2.LINE_AA)
            celdas.append(img)
        while len(celdas) < args.n:
            celdas.append(np.zeros_like(celdas[0]))
        grilla.append(celdas)

    titulos = [f"{i + 1}" for i in range(args.n)]
    cv2.imwrite(str(REPORTES / "seg_peores.png"), hoja(grilla, titulos, lado=300))
    print("\nFigura: reports/seg_peores.png (fila 1 = PCOS, fila 2 = sanas)")


if __name__ == "__main__":
    main()