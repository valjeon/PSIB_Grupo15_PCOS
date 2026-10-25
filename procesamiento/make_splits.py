"""Genera splits.csv sin fuga entre folds (correr después de clean_data.py).

1. Agrupa near-duplicados (pHash con distancia <= --umbral-grupo, componentes conexas):
   cortes del mismo paciente se parecen sin ser idénticos y no deben quedar en folds distintos.
2. Separa un hold-out (por defecto 20 %) que NO se toca hasta la evaluación final.
3. Reparte el 80 % restante en 5 folds estratificados por clase y agrupados.

Salida: splits.csv con
  path, label, fold, split, clase, grupo, origen_kaggle
  label : 1 = infected (PCOS), 0 = notinfected
  split : "cv" (se usa con los folds 0..4) o "holdout" (fold = -1)
También: reports/resumen_splits.txt
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASE_POSITIVA = "infected"


def origen(nombre):
    pref = nombre.split("_", 1)[0]
    return pref if pref in ("train", "test") else "?"


def phash_int(ruta):
    with Image.open(ruta) as im:
        return int(str(imagehash.phash(im)), 16)


def componentes(hashes, umbral):
    """Union-find sobre todos los pares con distancia de Hamming <= umbral."""
    n = len(hashes)
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if bin(hi ^ hashes[j]).count("1") <= umbral:
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[rj] = ri
    return [raiz(i) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral-grupo", type=int, default=8,
                    help="distancia pHash máxima para agrupar near-duplicados")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default="splits.csv",
                    help="archivo de salida en la raíz del repo (usar otro nombre para experimentar)")
    ap.add_argument("--clase-positiva", default=CLASE_POSITIVA,
                    help="nombre de la carpeta de la clase PCOS/anormal (label = 1)")
    args = ap.parse_args()

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste clean_data.py?")
    n = len(rutas)
    clases = np.array([r.parent.name for r in rutas])
    y = (clases == args.clase_positiva).astype(int)
    if y.sum() == 0 or y.sum() == n:
        raise SystemExit(f"Se esperan dos clases y una es '{args.clase_positiva}'. "
                         f"Encontré: {set(clases)}. Usá --clase-positiva.")
    orig = np.array([origen(r.name) for r in rutas])

    print(f"Calculando pHash de {n} imágenes...")
    hashes = [phash_int(r) for r in rutas]
    raices = componentes(hashes, args.umbral_grupo)
    ids = {r: k for k, r in enumerate(sorted(set(raices)))}
    grupos = np.array([ids[r] for r in raices])

    # --- diagnóstico de grupos ---
    tam = Counter(grupos)
    multi = [g for g, c in tam.items() if c > 1]
    mixtos_clase = [g for g in multi if len(set(y[grupos == g])) > 1]
    mixtos_origen = [g for g in multi if len(set(orig[grupos == g])) > 1]
    mayor = max(tam.values())

    # --- hold-out y folds, siempre por grupo ---
    n_ho = max(2, round(1 / args.holdout))
    sgkf_ho = StratifiedGroupKFold(n_splits=n_ho, shuffle=True, random_state=args.seed)
    idx_cv, idx_ho = next(sgkf_ho.split(np.zeros(n), y, grupos))

    fold = np.full(n, -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for k, (_, val) in enumerate(sgkf.split(np.zeros(len(idx_cv)), y[idx_cv], grupos[idx_cv])):
        fold[idx_cv[val]] = k
    split = np.where(fold >= 0, "cv", "holdout")

    # --- verificaciones: ningún grupo repartido en más de un fold / hold-out ---
    por_grupo = defaultdict(set)
    for g, f in zip(grupos, fold):
        por_grupo[g].add(f)
    fugas = [g for g, fs in por_grupo.items() if len(fs) > 1]
    assert not fugas, f"Grupos repartidos en más de un fold: {fugas[:5]}"
    assert (fold[idx_ho] == -1).all() and (fold[idx_cv] >= 0).all()

    with open(ROOT / args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "fold", "split", "clase", "grupo", "origen_kaggle"])
        for i, r in enumerate(rutas):
            w.writerow([r.relative_to(ROOT).as_posix(), int(y[i]), int(fold[i]), split[i],
                        clases[i], int(grupos[i]), orig[i]])

    # --- resumen ---
    L = [
        f"Imágenes: {n} | infected: {int(y.sum())} | notinfected: {int((1 - y).sum())}",
        f"Umbral de agrupado (pHash): {args.umbral_grupo}",
        f"Grupos: {len(tam)} | grupos con >1 imagen: {len(multi)} "
        f"({sum(tam[g] for g in multi)} imágenes) | grupo más grande: {mayor}",
        f"Grupos con clases mezcladas: {len(mixtos_clase)}",
        f"Grupos con imágenes de train Y test de Kaggle: {len(mixtos_origen)} "
        f"({sum(tam[g] for g in mixtos_origen)} imágenes)  <- evidencia de fuga en la partición original",
    ]
    if mayor > max(20, 0.02 * n):
        L.append(f"ATENCIÓN: un grupo de {mayor} imágenes ({100 * mayor / n:.1f} %) sugiere "
                 f"encadenamiento de similitud. Probá --umbral-grupo más bajo (6, 5...).")
    L += ["", f"{'split':8s}{'fold':>5s}{'n':>7s}{'infected':>10s}{'notinf.':>9s}{'% inf.':>8s}"]
    for s, fo in [("cv", k) for k in range(args.folds)] + [("holdout", -1)]:
        m = fold == fo
        pos, tot = int(y[m].sum()), int(m.sum())
        L.append(f"{s:8s}{fo:>5d}{tot:>7d}{pos:>10d}{tot - pos:>9d}{100 * pos / max(tot, 1):>8.1f}")
    L += ["", "Verificación: ningún grupo cruza folds ni hold-out: OK"]
    texto = "\n".join(L)
    print(texto)
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / f"resumen_{Path(args.salida).stem}.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
