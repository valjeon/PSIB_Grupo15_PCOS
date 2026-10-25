"""Compara filtros de speckle y métodos de contraste sobre una muestra de data/clean.

Salidas en reports/:
  preproc_marcas.png      original | máscara de marcas | sin marcas, recortada, tamaño común
  preproc_filtros.png     la imagen base con cada filtro de speckle
  preproc_contraste.png   la imagen filtrada con cada método de contraste
  preproc_metricas.csv    SSI y β por filtro; entropía y desvío por método de contraste

Métricas (sin referencia limpia, como es habitual en ecografía):
  SSI  = índice de supresión de speckle. < 1 suaviza; más bajo = más suavizado.
  beta = correlación de laplacianos con la original. 1 = bordes intactos.
  Un buen filtro baja el SSI sin hundir beta. Complementar siempre mirando las figuras.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import random
import time

import cv2
import numpy as np

from preprocess import (CONTRASTES, FILTROS, a_gris, base, contrastar, entropia, filtrar,
                        leer, quitar_marcas, beta, ssi)

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def hoja(filas, titulos, lado=160, cab=26):
    """Arma una grilla (BGR) con una imagen por celda."""
    out = np.zeros((cab + len(filas) * lado, len(titulos) * lado, 3), np.uint8)
    for j, t in enumerate(titulos):
        cv2.putText(out, t, (j * lado + 4, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, fila in enumerate(filas):
        for j, im in enumerate(fila):
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR if im.ndim == 2 else cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA)
            out[cab + i * lado:cab + (i + 1) * lado, j * lado:(j + 1) * lado] = im
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fig", type=int, default=8, help="imágenes en las figuras (mitad por clase)")
    ap.add_argument("--n-metricas", type=int, default=20, help="imágenes por clase para las métricas")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--filtro-ref", default="mediana", help="filtro previo a la figura de contraste")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    por_clase = defaultdict(list)
    for p in sorted(CLEAN.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXT:
            por_clase[p.parent.name].append(p)
    if not por_clase:
        raise SystemExit(f"No hay imágenes en {CLEAN}.")
    rng = random.Random(args.seed)
    REPORTES.mkdir(exist_ok=True)

    # ---------------- figuras
    k = max(1, args.n_fig // max(len(por_clase), 1))
    muestra = [p for cl in sorted(por_clase) for p in rng.sample(por_clase[cl], min(k, len(por_clase[cl])))]
    f_marcas, f_filtros, f_contraste = [], [], []
    nombres_f = list(FILTROS)
    nombres_c = list(CONTRASTES)
    for p in muestra:
        rgb = leer(p)
        _, mask = quitar_marcas(rgb)
        b = base(rgb, args.tam)
        f_marcas.append([a_gris(rgb), mask if mask.any() else np.zeros_like(a_gris(rgb)), b])
        f_filtros.append([filtrar(b, m) for m in nombres_f])
        ref = filtrar(b, args.filtro_ref)
        f_contraste.append([contrastar(ref, m) for m in nombres_c])
    cv2.imwrite(str(REPORTES / "preproc_marcas.png"),
                hoja(f_marcas, ["original (gris)", "mascara de marcas", "base: sin marcas+recorte"]))
    cv2.imwrite(str(REPORTES / "preproc_filtros.png"), hoja(f_filtros, nombres_f))
    cv2.imwrite(str(REPORTES / "preproc_contraste.png"),
                hoja(f_contraste, [f"{args.filtro_ref}+{m}" for m in nombres_c]))

    # ---------------- métricas
    acum_f = defaultdict(lambda: defaultdict(list))   # (filtro, clase) -> métrica -> valores
    tiempos = defaultdict(list)
    acum_c = defaultdict(lambda: defaultdict(list))
    for cl, rutas in sorted(por_clase.items()):
        for p in rng.sample(rutas, min(args.n_metricas, len(rutas))):
            b = base(p, args.tam)
            for m in nombres_f:
                if m == "ninguno":
                    continue
                t0 = time.perf_counter()
                g = filtrar(b, m)
                tiempos[m].append(time.perf_counter() - t0)
                acum_f[(m, cl)]["ssi"].append(ssi(b, g))
                acum_f[(m, cl)]["beta"].append(beta(b, g))
            ref = filtrar(b, args.filtro_ref)
            for m in nombres_c:
                c = contrastar(ref, m)
                acum_c[(m, cl)]["entropia"].append(entropia(c))
                acum_c[(m, cl)]["desvio"].append(float(c.std()))

    filas = []
    L = [f"Muestra de métricas: hasta {args.n_metricas} imágenes por clase | tamaño {args.tam}", "",
         "FILTROS DE SPECKLE (SSI más bajo = suaviza más; beta más alto = conserva bordes)",
         f"{'filtro':10s}{'clase':14s}{'SSI':>8s}{'beta':>8s}{'ms/img':>9s}"]
    for (m, cl), d in sorted(acum_f.items()):
        s, bt = float(np.mean(d["ssi"])), float(np.mean(d["beta"]))
        ms = 1000 * float(np.mean(tiempos[m]))
        filas.append(["filtro", m, cl, round(s, 4), round(bt, 4), "", "", len(d["ssi"])])
        L.append(f"{m:10s}{cl:14s}{s:8.3f}{bt:8.3f}{ms:9.1f}")
    L += ["", f"CONTRASTE (sobre filtro '{args.filtro_ref}')", f"{'metodo':10s}{'clase':14s}{'entropia':>10s}{'desvio':>9s}"]
    for (m, cl), d in sorted(acum_c.items()):
        e, ds = float(np.mean(d["entropia"])), float(np.mean(d["desvio"]))
        filas.append(["contraste", m, cl, "", "", round(e, 3), round(ds, 2), len(d["entropia"])])
        L.append(f"{m:10s}{cl:14s}{e:10.3f}{ds:9.2f}")
    with open(REPORTES / "preproc_metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "metodo", "clase", "ssi", "beta", "entropia", "desvio", "n"])
        w.writerows(filas)
    L += ["", "Figuras: reports/preproc_marcas.png, preproc_filtros.png, preproc_contraste.png",
          "Elegir mirando las figuras: ¿se ven los folículos nítidos y sin halos? "
          "Anotar la elección y su justificación en el informe."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_comparacion_preproc.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
