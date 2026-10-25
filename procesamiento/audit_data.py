"""Auditoría de data/clean (correr después de clean_data.py).

Genera en reports/:
  audit.csv                    una fila por imagen (tamaño, modo, brillo, saturación, color)
  resumen_auditoria.txt        tabla por clase + chequeo de sesgos
  muestra_azar_<clase>.jpg     montaje de imágenes al azar para revisión visual
  muestra_marcas_<clase>.jpg   las imágenes con más píxeles saturados/coloreados
                               (donde es más probable que haya texto, calibres o logos)

Idea central: si las clases difieren en algo que no es el ovario (tamaño, modo de
color, marcas grabadas), el modelo puede aprender eso en lugar de la patología.
"""
from collections import Counter
from pathlib import Path
import argparse
import csv
import random
import statistics as st

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def origen(nombre):
    pref = nombre.split("_", 1)[0]
    return pref if pref in ("train", "test") else "?"


def medir(ruta):
    with Image.open(ruta) as im:
        ancho, alto = im.size
        modo = im.mode
        gris = np.asarray(im.convert("L"), dtype=np.float32)
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    return {
        "path": ruta.relative_to(ROOT).as_posix(),
        "clase": ruta.parent.name,
        "origen": origen(ruta.name),
        "ancho": ancho,
        "alto": alto,
        "modo": modo,
        "bytes": ruta.stat().st_size,
        "brillo": float(gris.mean()),
        "desvio": float(gris.std()),
        # píxeles casi blancos: texto, calibres, marcas de escala
        "frac_sat": float((gris >= 250).mean()),
        # píxeles con color marcado: anotaciones amarillas/verdes sobre fondo gris
        "frac_color": float(((rgb.max(axis=2) - rgb.min(axis=2)) > 40).mean()),
    }


def miniatura(ruta, lado):
    with Image.open(ruta) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        tile = Image.new("RGB", (lado, lado), (40, 40, 40))
        tile.paste(im, ((lado - im.width) // 2, (lado - im.height) // 2))
    return tile


def montaje(rutas, destino, cols=8, lado=180):
    if not rutas:
        return
    filas = -(-len(rutas) // cols)
    hoja = Image.new("RGB", (cols * lado, filas * lado), (40, 40, 40))
    for k, r in enumerate(rutas):
        hoja.paste(miniatura(r, lado), ((k % cols) * lado, (k // cols) * lado))
    hoja.save(destino, quality=90)


def sesgo_por_metadatos(filas, seed):
    """AUC de un clasificador que NO ve el contenido: solo tamaño, bytes, brillo, etc."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        return None
    clases = sorted({f["clase"] for f in filas})
    if len(clases) != 2 or len(filas) < 50:
        return None
    X = np.array([[f["ancho"], f["alto"], f["ancho"] / f["alto"], f["bytes"], f["brillo"],
                   f["desvio"], f["frac_sat"], f["frac_color"]] for f in filas])
    y = np.array([f["clase"] == clases[1] for f in filas], dtype=int)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    p = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-azar", type=int, default=50, help="imágenes al azar por clase")
    ap.add_argument("--n-marcas", type=int, default=40, help="imágenes 'sospechosas' por clase")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste clean_data.py?")
    filas = [medir(r) for r in rutas]
    REPORTES.mkdir(exist_ok=True)

    with open(REPORTES / "audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    clases = sorted({f["clase"] for f in filas})
    rng = random.Random(args.seed)
    L = [f"Imágenes auditadas: {len(filas)}", ""]
    for cl in clases:
        fs = [f for f in filas if f["clase"] == cl]
        tam = Counter((f["ancho"], f["alto"]) for f in fs)
        (t_top, n_top), = tam.most_common(1)
        modos = Counter(f["modo"] for f in fs)
        L += [
            f"[{cl}]  n = {len(fs)}  (train={sum(f['origen'] == 'train' for f in fs)}, "
            f"test={sum(f['origen'] == 'test' for f in fs)})",
            f"  tamaños distintos: {len(tam)} | más frecuente: {t_top[0]}x{t_top[1]} "
            f"({100 * n_top / len(fs):.0f} %)",
            f"  ancho mediano: {st.median(f['ancho'] for f in fs):.0f} | "
            f"alto mediano: {st.median(f['alto'] for f in fs):.0f} | "
            f"bytes medianos: {st.median(f['bytes'] for f in fs):.0f}",
            "  modos: " + ", ".join(f"{k}={v}" for k, v in modos.most_common()),
            f"  brillo medio: {st.mean(f['brillo'] for f in fs):.1f} | "
            f"desvío medio: {st.mean(f['desvio'] for f in fs):.1f}",
            f"  con >1 % de píxeles saturados: "
            f"{100 * sum(f['frac_sat'] > 0.01 for f in fs) / len(fs):.0f} % | "
            f"con color (>0,1 % de píxeles): "
            f"{100 * sum(f['frac_color'] > 0.001 for f in fs) / len(fs):.0f} %",
            "",
        ]
        al = rng.sample(fs, min(args.n_azar, len(fs)))
        montaje([ROOT / f["path"] for f in al], REPORTES / f"muestra_azar_{cl}.jpg")
        sosp = sorted(fs, key=lambda f: f["frac_sat"] + f["frac_color"], reverse=True)
        montaje([ROOT / f["path"] for f in sosp[:args.n_marcas]],
                REPORTES / f"muestra_marcas_{cl}.jpg")

    auc = sesgo_por_metadatos(filas, args.seed)
    if auc is not None:
        veredicto = ("ALERTA: la clase se predice sin mirar el ovario; hay sesgo de adquisición"
                     if auc >= 0.65 else "sin señal fuerte de sesgo")
        L += [f"Chequeo de atajos: AUC solo con metadatos (tamaño, bytes, brillo, saturación, "
              f"color) = {auc:.3f} -> {veredicto}",
              "  (CV sin agrupar: los near-duplicados lo inflan un poco. ~0,5 es lo esperable.)", ""]
    L += ["Revisión visual pendiente (a mano):",
          f"  - {REPORTES}/muestra_azar_<clase>.jpg y muestra_marcas_<clase>.jpg",
          "  - anotar en el informe: ¿hay texto, calibres, logos? ¿en qué proporción por clase?"]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_auditoria.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
