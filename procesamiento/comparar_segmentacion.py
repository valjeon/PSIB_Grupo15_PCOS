"""Elige filtro + contraste según cómo sale la SEGMENTACIÓN (no según cómo se ve la imagen).

Usa solo imágenes con split == "cv" de splits.csv (el hold-out no se toca) y, por
defecto, UNA imagen por grupo de near-duplicados, para que un grupo grande no pese
más que el resto.

Salidas en reports/:
  seg_overlays.png          contornos de folículos (rojo) y del abanico (celeste) por combinación
  seg_conteos.csv           n de folículos por imagen y combinación
  resumen_segmentacion.txt  por combinación: conteo medio por clase, AUC del conteo
                            (qué tan bien separa PCOS de sanos) y su desvío entre folds

Criterio sugerido: AUC alto y estable entre folds, pocos folículos en sanos, y
contornos que en la figura sigan bien a los folículos. Anotar la elección en el informe.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import random
import time

import cv2
import numpy as np

from comparar_preprocesamiento import hoja
from segmentacion import segment_detalle, superponer

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"


def auc(pos, neg):
    """AUC de Mann-Whitney: P(conteo de una PCOS > conteo de una sana)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    mayor = (pos[:, None] > neg[None]).sum()
    empate = (pos[:, None] == neg[None]).sum()
    return float((mayor + 0.5 * empate) / (len(pos) * len(neg)))


def leer_splits(ruta, uno_por_grupo, seed):
    with open(ruta, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["split"] == "cv"]
    if uno_por_grupo:
        rng = random.Random(seed)
        por_grupo = defaultdict(list)
        for r in filas:
            por_grupo[r["grupo"]].append(r)
        filas = [rng.choice(v) for _, v in sorted(por_grupo.items(), key=lambda kv: int(kv[0]))]
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtros", nargs="+", default=["mediana", "lee", "kuan", "bilateral"])
    ap.add_argument("--contrastes", nargs="+", default=["estirar", "clahe"])
    ap.add_argument("--metodo", default="multiotsu", choices=["multiotsu", "otsu", "chanvese"])
    ap.add_argument("--n-por-clase", type=int, default=40, help="0 = todas")
    ap.add_argument("--n-fig", type=int, default=8)
    ap.add_argument("--todas-del-grupo", action="store_true",
                    help="usar todas las imágenes de cada grupo (por defecto, una por grupo)")
    ap.add_argument("--splits", default="splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ver-rechazos", action="store_true",
                    help="dibujar en amarillo las regiones oscuras descartadas")
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

    filas = leer_splits(ROOT / args.splits, not args.todas_del_grupo, args.seed)
    rng = random.Random(args.seed)
    por_clase = defaultdict(list)
    for r in filas:
        por_clase[int(r["label"])].append(r)
    muestra = []
    for y in sorted(por_clase):
        v = por_clase[y]
        k = len(v) if args.n_por_clase <= 0 else min(args.n_por_clase, len(v))
        muestra += rng.sample(v, k)
    combos = [(f, c) for f in args.filtros for c in args.contrastes]
    print(f"{len(muestra)} imágenes x {len(combos)} combinaciones (método: {args.metodo})")

    REPORTES.mkdir(exist_ok=True)
    conteos = []
    tiempos = defaultdict(list)
    fig_rutas = [r for y in sorted(por_clase) for r in
                 [m for m in muestra if int(m["label"]) == y][:max(1, args.n_fig // 2)]]
    fig_ids = {r["path"] for r in fig_rutas}
    fig = defaultdict(dict)

    for r in muestra:
        for f, c in combos:
            t0 = time.perf_counter()
            d = segment_detalle(ROOT / r["path"], filtro=f, contraste=c, metodo=args.metodo, **kw)
            tiempos[(f, c)].append(time.perf_counter() - t0)
            conteos.append(dict(path=r["path"], label=int(r["label"]), fold=int(r["fold"]),
                                grupo=r["grupo"], filtro=f, contraste=c,
                                n_foliculos=d["n_foliculos"]))
            if r["path"] in fig_ids:
                fig[r["path"]][(f, c)] = superponer(d, ver_rechazos=args.ver_rechazos)

    with open(REPORTES / "seg_conteos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conteos[0]))
        w.writeheader()
        w.writerows(conteos)

    filas_fig = [[fig[r["path"]][fc] for fc in combos] for r in fig_rutas]
    titulos = [f"{f}+{c}" for f, c in combos]
    lado = 200 if len(combos) > 3 else 400   # con pocas combinaciones, figura más grande
    cv2.imwrite(str(REPORTES / "seg_overlays.png"), hoja(filas_fig, titulos, lado=lado))

    L = [f"Imágenes: {len(muestra)} (split cv, "
         f"{'una por grupo' if not args.todas_del_grupo else 'todas las del grupo'}) | "
         f"método: {args.metodo} | parámetros: {kw or 'por defecto'}", "",
         f"{'combinación':20s}{'n PCOS':>8s}{'n sana':>8s}{'AUC':>7s}{'±folds':>8s}"
         f"{'sanas=0':>9s}{'s/img':>7s}"]
    resultados = []
    for f, c in combos:
        sel = [x for x in conteos if x["filtro"] == f and x["contraste"] == c]
        pos = [x["n_foliculos"] for x in sel if x["label"] == 1]
        neg = [x["n_foliculos"] for x in sel if x["label"] == 0]
        por_fold = []
        for k in sorted({x["fold"] for x in sel}):
            pk = [x["n_foliculos"] for x in sel if x["fold"] == k and x["label"] == 1]
            nk = [x["n_foliculos"] for x in sel if x["fold"] == k and x["label"] == 0]
            if pk and nk:
                por_fold.append(auc(pk, nk))
        a = auc(pos, neg)
        resultados.append((a, f, c))
        L.append(f"{f + '+' + c:20s}{np.mean(pos):8.1f}{np.mean(neg):8.1f}{a:7.3f}"
                 f"{np.std(por_fold) if por_fold else float('nan'):8.3f}"
                 f"{100 * np.mean(np.array(neg) == 0):8.0f}%"
                 f"{np.mean(tiempos[(f, c)]):7.2f}")
    mejor = max(resultados)
    L += ["", f"Mayor AUC del conteo: {mejor[1]}+{mejor[2]} ({mejor[0]:.3f})",
          "No elegir solo por AUC: revisar reports/seg_overlays.png. Un conteo alto en sanas o",
          "contornos que se salen del folículo indican que el umbral está agarrando estroma."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_segmentacion.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()