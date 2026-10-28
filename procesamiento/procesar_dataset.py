"""Aplica preprocess() a todo data/clean -> data/pre (PNG sin pérdida) y corre un
control de atajos sobre el resultado.

El control entrena un Random Forest que NO ve la imagen sino estadísticos de bajo nivel
(histograma, gradiente, ruido). Si todavía separa las clases, el preprocesamiento no alcanzó
a borrar las pistas de la fuente. Usa los grupos de splits.csv si existe (CV sin fuga).

Uso (desde cualquier carpeta):
    python procesar_dataset.py --filtro mediana --contraste clahe --tam 256
"""
from pathlib import Path
import argparse
import csv
import shutil

import cv2
import numpy as np

from preprocess import entropia, preprocess

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
PRE = ROOT / "data" / "pre"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def estadisticos(g):
    x = g.astype(np.float32) / 255.0
    m, s = float(x.mean()), float(x.std())
    z = (x - m) / (s + 1e-6)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1)
    return {
        "media": m, "desvio": s,
        "asimetria": float((z ** 3).mean()), "curtosis": float((z ** 4).mean()),
        "entropia": entropia(g),
        "grad_medio": float(np.hypot(gx, gy).mean()),
        "lap_var": float(cv2.Laplacian(x, cv2.CV_32F).var()),
        "frac_oscura": float((x < 0.1).mean()), "frac_clara": float((x > 0.9).mean()),
    }


def grupos_de_splits(claves, archivo="splits.csv"):
    """(clase, nombre sin extensión) -> grupo, desde el CSV de splits. None si falta algo."""
    f = ROOT / archivo
    if not f.exists():
        return None
    mapa = {}
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mapa[(r["clase"], Path(r["path"]).stem)] = int(r["grupo"])
    if not all(k in mapa for k in claves):
        return None
    return np.array([mapa[k] for k in claves])


def control_atajos(filas, claves, seed, archivo="splits.csv"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

    clases = sorted({k[0] for k in claves})
    if len(clases) != 2 or len(filas) < 50:
        return None
    nombres = list(filas[0])
    X = np.array([[f[n] for n in nombres] for f in filas])
    y = np.array([k[0] == clases[1] for k in claves], dtype=int)
    grupos = grupos_de_splits(claves, archivo)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if grupos is not None:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(rf, X, y, cv=cv, groups=grupos, method="predict_proba")[:, 1]
        cv_txt = f"CV agrupada por near-duplicados ({archivo})"
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
        cv_txt = "CV sin agrupar (no se encontró splits.csv completo)"
    solas = {}
    for k, n in enumerate(nombres):
        a = roc_auc_score(y, X[:, k])
        solas[n] = max(a, 1 - a)
    return roc_auc_score(y, p), solas, cv_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="mediana5")
    ap.add_argument("--contraste", default="ninguno")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--sin-marcas", action="store_true")
    ap.add_argument("--sin-recorte", action="store_true")
    ap.add_argument("--sin-texto", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="splits.csv",
                    help="CSV con los grupos para la CV del control (en la raíz del repo)")
    args = ap.parse_args()

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}.")
    if PRE.exists():
        shutil.rmtree(PRE)

    filas, claves, fallos = [], [], []
    for i, p in enumerate(rutas, 1):
        try:
            g = preprocess(p, args.filtro, args.contraste, args.tam,
                           marcas=not args.sin_marcas, recortar=not args.sin_recorte,
                           texto=not args.sin_texto)
        except Exception as e:
            fallos.append((p.name, str(e)))
            continue
        dest = PRE / p.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{p.stem}.png"), g)
        filas.append(estadisticos(g))
        claves.append((p.parent.name, p.stem))
        if i % 200 == 0:
            print(f"  {i}/{len(rutas)}")

    L = [f"Config: filtro={args.filtro} contraste={args.contraste} tam={args.tam} "
         f"marcas={not args.sin_marcas} recorte={not args.sin_recorte} texto={not args.sin_texto}",
         f"Procesadas: {len(filas)} de {len(rutas)} -> {PRE}"]
    if fallos:
        L.append(f"Fallaron {len(fallos)}: " + ", ".join(n for n, _ in fallos[:5]))
    res = control_atajos(filas, claves, args.seed, args.splits)
    if res is not None:
        auc, solas, cv_txt = res
        if auc >= 0.9:
            v = "las clases siguen siendo separables por estadísticos de bajo nivel: hay pistas de la fuente"
        elif auc >= 0.65:
            v = "señal moderada: puede ser patología o fuente, no se puede distinguir con este control"
        else:
            v = "sin señal fuerte"
        L += ["", f"Control de atajos tras preprocesar (solo estadísticos, sin ver el contenido): "
                  f"AUC = {auc:.3f} -> {v}", f"  ({cv_txt})",
              "  AUC por estadístico: " + ", ".join(f"{k}={v:.2f}" for k, v in
                                                    sorted(solas.items(), key=lambda kv: -kv[1]))]
    texto = "\n".join(L)
    print(texto)
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "resumen_procesamiento.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
