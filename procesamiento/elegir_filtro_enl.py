"""Elige el filtro de speckle usando ENL como único criterio, sobre los bloques que marcaste a mano.
 
ENL = media^2 / varianza, medido en bloques de tejido homogéneo. Más ENL = más suavizado del speckle.
 
Entrada: rois.csv, generado por anotar_rois.py (imagen, clase, posición y tamaño de cada bloque).
Los bloques son los MISMOS píxeles para la imagen original y para todos los filtros, así que la
comparación es pareada. Se marcan antes de ver ningún filtro.
 
Salidas en reports/:
  enl_resultados_manual.csv   una fila por imagen y filtro (ENL, cociente vs. original, beta)
  enl_resumen_manual.txt      ranking por ENL + tests estadísticos pareados
  enl_rois_manual.png         bloques sobre algunas imágenes (para verificar)
 
Uso:
    python elegir_filtro_enl.py                 # lee rois.csv de la raíz del repo
    python elegir_filtro_enl.py --rois otro.csv
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import time

import cv2
import numpy as np
from scipy import stats

from preprocess import FILTROS, base, beta

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"


# ---------------------------------------------------------------- ENL y bloques marcados
def enl(img, bloques, lado=16, eps=1e-3):
    """Mediana del ENL = media^2 / varianza sobre los bloques dados."""
    f = img.astype(np.float64)
    vals = []
    for y, x in bloques:
        b = f[y:y + lado, x:x + lado]
        vals.append(b.mean() ** 2 / (b.var() + eps))
    return float(np.median(vals))


def leer_rois(archivo):
    """rois.csv -> ({ruta: (clase, [(y, x), ...])}, lado, tam)."""
    por_img, lados, tams = {}, set(), set()
    with open(archivo, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ruta = ROOT / r["path"]
            por_img.setdefault(ruta, (r["clase"], []))[1].append((int(r["y"]), int(r["x"])))
            lados.add(int(r["lado"]))
            tams.add(int(r["tam"]))
    if not por_img:
        raise SystemExit(f"{archivo} está vacío.")
    if len(lados) != 1 or len(tams) != 1:
        raise SystemExit("rois.csv mezcla distintos lado/tam: anotá todo con la misma configuración.")
    return por_img, lados.pop(), tams.pop()


def figura_rois(items, lado_bloque, destino, cols=4, n=12):
    tiles = []
    for g, bloques in items[:n]:
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for y, x in bloques:
            cv2.rectangle(im, (x, y), (x + lado_bloque, y + lado_bloque), (0, 255, 0), 1)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    filas = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(destino), np.vstack(filas))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", default="rois.csv", help="CSV generado por anotar_rois.py")
    args = ap.parse_args()

    manual, lado, tam = leer_rois(ROOT / args.rois)
    muestra = [(p, cl) for p, (cl, _) in manual.items()]
    origen = f"{args.rois} (bloques marcados a mano)"
    filtros = {k: v for k, v in FILTROS.items() if k != "ninguno"}
    nombres = list(filtros)
    REPORTES.mkdir(exist_ok=True)

    filas, enl_o, enl_f, betas, tiempos = [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    para_fig = []
    for p, cl in muestra:
        g = base(p, tam)
        bloques = manual[p][1]
        para_fig.append((g, bloques))
        e0 = enl(g, bloques, lado)
        enl_o.append(e0)
        for m in nombres:
            t0 = time.perf_counter()
            gf = filtros[m](g)
            tiempos[m].append(time.perf_counter() - t0)
            e, b = enl(gf, bloques, lado), beta(g, gf)
            enl_f[m].append(e)
            betas[m].append(b)
            filas.append([p.name, cl, m, round(e0, 3), round(e, 3), round(e / e0, 3), round(b, 4)])

    n = len(enl_o)
    if n < 10:
        raise SystemExit(f"Solo {n} imágenes marcadas: marcá más con anotar_rois.py (mínimo 10).")
    ratio = {m: np.array(enl_f[m]) / np.array(enl_o) for m in nombres}

    with open(REPORTES / "enl_resultados_manual.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "clase", "filtro", "enl_original", "enl_filtrada", "cociente", "beta"])
        w.writerows(filas)
    figura_rois(para_fig, lado, REPORTES / "enl_rois_manual.png")

    orden = sorted(nombres, key=lambda m: -np.median(ratio[m]))
    L = [f"Muestra: {origen}",
         f"Imágenes marcadas: {n} | bloque {lado}x{lado} en {tam}x{tam}",
         f"ENL mediano de la imagen original en esos bloques: {np.median(enl_o):.2f}", "",
         "RANKING POR ENL (cociente = ENL filtrada / ENL original; mayor = más supresión de speckle)",
         f"{'filtro':15s}{'cociente med.':>14s}{'IQR':>16s}{'ENL med.':>10s}{'beta*':>8s}{'ms/img':>9s}"]
    for m in orden:
        q1, q3 = np.percentile(ratio[m], [25, 75])
        L.append(f"{m:15s}{np.median(ratio[m]):14.2f}{f'[{q1:.2f}, {q3:.2f}]':>16s}"
                 f"{np.median(enl_f[m]):10.2f}{np.mean(betas[m]):8.3f}{1000 * np.mean(tiempos[m]):9.1f}")
    L += ["", "* beta (preservación de bordes) se reporta pero NO entra en el ranking."]

    # tests pareados (mismas imágenes y mismos bloques para todos)
    chi, p_fr = stats.friedmanchisquare(*[ratio[m] for m in nombres])
    L += ["", f"Friedman entre los {len(nombres)} filtros: chi2 = {chi:.1f}, p = {p_fr:.2g}"]
    top = orden[0]
    L.append(f"Wilcoxon pareado de '{top}' contra cada uno (p con Bonferroni, {len(nombres) - 1} comparaciones):")
    for m in orden[1:]:
        try:
            p = stats.wilcoxon(ratio[top], ratio[m]).pvalue
        except ValueError:
            p = 1.0
        L.append(f"  {top} vs {m:15s} p = {min(1.0, p * (len(nombres) - 1)):.3g}")
    # filtros no dominados: ningún otro tiene a la vez más ENL y más beta
    r = {m: float(np.median(ratio[m])) for m in nombres}
    b = {m: float(np.mean(betas[m])) for m in nombres}
    dominados = {m: [o for o in orden if o != m and r[o] >= r[m] and b[o] >= b[m]
                     and (r[o] > r[m] or b[o] > b[m])] for m in nombres}
    L += ["", "FILTROS NO DOMINADOS (ningún otro suaviza más Y conserva más bordes):",
          "  " + ", ".join(m for m in orden if not dominados[m])]
    L += ["",
          "ATENCIÓN: el ENL sube con cualquier suavizado, así que solo con ENL gana el filtro que más",
          "borronea. Mirá beta y las imágenes filtradas (comparar_preprocesamiento.py) antes de decidir,",
          "y validá el finalista con máscaras manuales (Dice) en unas 15-20 imágenes.",
          "Revisá reports/enl_rois_manual.png: los cuadrados verdes tienen que caer en tejido liso, no en bordes."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "enl_resumen_manual.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()