"""Agrupa copias aumentadas de una misma ecografía aunque estén ROTADAS, DESPLAZADAS,
ESPEJADAS o con ZOOM, y rehace la partición sin fuga.

Por qué: make_splits.py agrupa con pHash, que solo reconoce copias casi idénticas. Una
copia rotada 30° o desplazada tiene un pHash muy distinto y queda como imagen "nueva".
Resultado: el mismo ovario puede caer en folds distintos o en el hold-out (fuga) y pesar
varias veces en cualquier métrica.

Cómo:
  1. Puntos característicos ORB de cada imagen (y de su versión espejada). ORB es
     invariante a rotación y bastante robusto a escala y brillo.
  2. Para no comparar los ~1,4 millones de pares, primero se preseleccionan, para cada
     imagen, los --vecinos candidatos más parecidos según dos descriptores globales que
     casi no cambian con las aumentaciones: una "bolsa de palabras visuales" (cuántas
     veces aparece cada tipo de punto ORB) y el histograma de intensidades.
  3. Cada par candidato se verifica geométricamente: se emparejan los puntos y RANSAC
     busca una transformación de similitud (rotación + escala + traslación) que los
     explique. Si la sostienen al menos --inliers-min puntos, es la misma imagen.
     Dos ecografías distintas casi nunca tienen decenas de puntos compatibles con una
     única rotación + traslación.
  4. Se unen estos vínculos con los grupos de pHash de splits.csv (union-find).
  5. Hold-out y folds con StratifiedGroupKFold, igual que make_splits.py.

NO pisa splits.csv. Salidas:
  splits2.csv                     mismas columnas que splits.csv (+ grupo_phash)
  reports/resumen_duplicados.txt  cuántos grupos, cuántas copias, verificaciones
  reports/duplicados_ejemplos.png los grupos más grandes, para revisar a ojo
  reports/duplicados_pares.csv    cada vínculo encontrado con su cantidad de inliers

Uso (desde procesamiento/):
    python3 agrupar_duplicados.py
    python3 comparar_segmentacion.py --reumbral --splits splits2.csv
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import time

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from preprocess import leer, a_gris, recortar_util

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"


# ---------------------------------------------------------------- descriptores
def preparar(ruta, tam=320):
    """Gris, recortado al área útil y con el lado mayor = tam (sin deformar)."""
    g = recortar_util(a_gris(leer(ruta)))
    s = tam / max(g.shape)
    return cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def histograma(g):
    """Histograma de intensidades del tejido (sin fondo negro ni relleno blanco),
    después de estirar entre los percentiles 2 y 98: así una copia con más o menos
    brillo/contraste (otra aumentación común) tiene casi el mismo histograma."""
    v = g[(g > 12) & (g < 250)].astype(np.float32)
    if v.size < 100:
        return np.full(64, 1 / 64, np.float32)
    lo, hi = np.percentile(v, (2, 98))
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 0.999) * 64
    h = np.bincount(v.astype(int), minlength=64).astype(np.float32)
    h = cv2.GaussianBlur(h.reshape(1, -1), (1, 5), 0).ravel()
    return h / h.sum()


def orb_desc(orb, g):
    # máscara: solo tejido, para no tomar puntos en el borde del relleno
    m = ((g > 12) & (g < 250)).astype(np.uint8)
    m = cv2.erode(m, np.ones((7, 7), np.uint8))
    kp, des = orb.detectAndCompute(g, m)
    if des is None or len(kp) < 10:
        return None
    return np.float32([k.pt for k in kp]), des


def vocabulario(descs, k=256, muestra=60000, seed=0):
    """Palabras visuales: k-means sobre descriptores ORB (bits desempaquetados)."""
    todos = np.vstack([d for d in descs if d is not None])
    rng = np.random.default_rng(seed)
    if len(todos) > muestra:
        todos = todos[rng.choice(len(todos), muestra, replace=False)]
    bits = np.unpackbits(todos, axis=1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centros = cv2.kmeans(bits, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return np.packbits((centros > 0.5).astype(np.uint8), axis=1)


def bolsa(des_list, centros, bf):
    """Histograma de palabras visuales (de la imagen y su espejo). Invariante a rotación,
    desplazamiento, espejado, brillo y, en buena medida, zoom."""
    v = np.zeros(len(centros), np.float32)
    for des in des_list:
        if des is None:
            continue
        for m in bf.match(des, centros):
            v[m.trainIdx] += 1
    return v


def verificar(a, b, bf, ratio=0.75):
    """Cantidad de inliers de una similitud estimada por RANSAC (0 si no hay)."""
    if a is None or b is None:
        return 0
    pa, da = a
    pb, db = b
    pares = bf.knnMatch(da, db, k=2)
    buenos = [p[0] for p in pares if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(buenos) < 8:
        return 0
    src = pa[[m.queryIdx for m in buenos]]
    dst = pb[[m.trainIdx for m in buenos]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0, maxIters=2000)
    if M is None or inl is None:
        return 0
    escala = float(np.hypot(M[0, 0], M[1, 0]))
    if not 0.5 <= escala <= 2.0:  # zoom absurdo: no es una aumentación
        return 0
    return int(inl.sum())


# ---------------------------------------------------------------- grupos
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def raiz(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def unir(self, i, j):
        ri, rj = self.raiz(i), self.raiz(j)
        if ri != rj:
            self.p[rj] = ri


def figura_grupos(rutas, grupos, n_grupos=8, n_por_grupo=6, lado=140):
    tam = Counter(grupos)
    mayores = [g for g, c in tam.most_common(n_grupos) if c > 1]
    filas = []
    for g in mayores:
        idx = [i for i in range(len(rutas)) if grupos[i] == g][:n_por_grupo]
        celdas = []
        for i in idx:
            im = cv2.resize(preparar(rutas[i]), (lado, lado), interpolation=cv2.INTER_AREA)
            celdas.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        while len(celdas) < n_por_grupo:
            celdas.append(np.zeros((lado, lado, 3), np.uint8))
        fila = np.hstack(celdas)
        cv2.putText(fila, f"grupo {g} ({tam[g]} img)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        filas.append(fila)
    return np.vstack(filas) if filas else None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.csv", help="partición actual (se usa su lista de imágenes y sus grupos pHash)")
    ap.add_argument("--salida", default="splits2.csv")
    ap.add_argument("--vecinos", type=int, default=40, help="candidatos por imagen a verificar")
    ap.add_argument("--inliers-min", type=int, default=25)
    ap.add_argument("--features", type=int, default=800)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(ROOT / args.splits, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    rutas = [ROOT / r["path"] for r in filas]
    y = np.array([int(r["label"]) for r in filas])
    n = len(rutas)
    faltan = [r for r in rutas if not r.exists()]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} imágenes (ej. {faltan[0]}). ¿Corriste clean_data.py?")

    # 1) descriptores
    t0 = time.time()
    orb = cv2.ORB_create(nfeatures=args.features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    hist, desc, desc_esp = [], [], []
    for i, r in enumerate(rutas):
        g = preparar(r)
        hist.append(histograma(g))
        desc.append(orb_desc(orb, g))
        desc_esp.append(orb_desc(orb, cv2.flip(g, 1)))
        if (i + 1) % 200 == 0:
            print(f"  descriptores {i + 1}/{n}  ({time.time() - t0:.0f} s)")
    H = np.stack(hist)

    # 2) candidatos: los --vecinos más parecidos según dos descriptores globales
    #    (bolsa de palabras visuales ORB con TF-IDF, e histograma de intensidades)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    centros = vocabulario([d[1] for d in desc if d is not None])
    B = np.stack([bolsa([None if a is None else a[1], None if b is None else b[1]], centros, bf)
                  for a, b in zip(desc, desc_esp)])
    idf = np.log(n / (1 + (B > 0).sum(0)))
    B = B / np.maximum(B.sum(1, keepdims=True), 1) * idf
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    sim_bolsa = B @ B.T
    k = min(args.vecinos, n - 1)
    cand = set()
    for i in range(n):
        d_h = np.abs(H - H[i]).sum(1)
        d_b = -sim_bolsa[i].copy()
        for d in (d_h, d_b):
            d[i] = np.inf
            for j in np.argpartition(d, k)[:k]:
                cand.add((min(i, int(j)), max(i, int(j))))
    print(f"Pares candidatos a verificar: {len(cand)}  ({time.time() - t0:.0f} s)")

    # 3) verificación geométrica (normal y espejada)
    uf = UnionFind(n)
    vinculos = []
    for c, (i, j) in enumerate(sorted(cand)):
        inl = max(verificar(desc[i], desc[j], bf), verificar(desc_esp[i], desc[j], bf))
        if inl >= args.inliers_min:
            uf.unir(i, j)
            vinculos.append((i, j, inl))
        if (c + 1) % 5000 == 0:
            print(f"  verificados {c + 1}/{len(cand)}  ({time.time() - t0:.0f} s)")

    # 4) unir con los grupos pHash previos
    ph = [r["grupo"] for r in filas]
    primero = {}
    for i, g in enumerate(ph):
        if g in primero:
            uf.unir(primero[g], i)
        else:
            primero[g] = i
    raices = [uf.raiz(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(sorted(set(raices)))}
    grupos = np.array([ids[r] for r in raices])

    # 5) partición
    n_ho = max(2, round(1 / args.holdout))
    idx_cv, idx_ho = next(StratifiedGroupKFold(n_splits=n_ho, shuffle=True, random_state=args.seed)
                          .split(np.zeros(n), y, grupos))
    fold = np.full(n, -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for kf, (_, val) in enumerate(sgkf.split(np.zeros(len(idx_cv)), y[idx_cv], grupos[idx_cv])):
        fold[idx_cv[val]] = kf
    split = np.where(fold >= 0, "cv", "holdout")

    por_grupo = defaultdict(set)
    for g, f in zip(grupos, fold):
        por_grupo[g].add(f)
    assert not [g for g, fs in por_grupo.items() if len(fs) > 1], "un grupo cruza folds"

    with open(ROOT / args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "fold", "split", "clase", "grupo", "origen_kaggle", "grupo_phash"])
        for i, r in enumerate(filas):
            w.writerow([r["path"], int(y[i]), int(fold[i]), split[i], r["clase"], int(grupos[i]),
                        r.get("origen_kaggle", ""), r["grupo"]])
    REPORTES.mkdir(exist_ok=True)
    with open(REPORTES / "duplicados_pares.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_a", "path_b", "inliers"])
        for i, j, inl in sorted(vinculos, key=lambda v: -v[2]):
            w.writerow([filas[i]["path"], filas[j]["path"], inl])
    fig = figura_grupos(rutas, grupos)
    if fig is not None:
        cv2.imwrite(str(REPORTES / "duplicados_ejemplos.png"), fig)

    # cuántos grupos pHash viejos quedaron repartidos entre folds en la partición anterior
    fold_viejo = np.array([int(r["fold"]) for r in filas])
    fuga_vieja = sum(1 for g in set(grupos) if len(set(fold_viejo[grupos == g])) > 1)

    tam = Counter(grupos)
    mixtos = [g for g in tam if len(set(y[grupos == g])) > 1]
    mayor = max(tam.values())
    L = [f"Imágenes: {n} | grupos pHash (antes): {len(set(ph))} | grupos ahora: {len(tam)}",
         f"Vínculos nuevos por ORB+RANSAC: {len(vinculos)} (umbral {args.inliers_min} inliers)",
         f"Grupos con >1 imagen: {sum(c > 1 for c in tam.values())} | grupo más grande: {mayor}",
         f"Imágenes 'únicas' reales (= grupos): {len(tam)} -> {n - len(tam)} eran copias",
         f"Grupos nuevos que en {args.splits} estaban repartidos entre folds/hold-out: {fuga_vieja}"
         "  <- fuga en la partición anterior",
         f"Grupos con clases mezcladas: {len(mixtos)}"
         + ("  <- REVISAR: la misma imagen con las dos etiquetas, o un vínculo falso" if mixtos else ""),
         ""]
    if mayor > max(30, 0.03 * n):
        L.append(f"ATENCIÓN: un grupo de {mayor} imágenes puede ser encadenamiento de vínculos falsos. "
                 f"Revisar duplicados_ejemplos.png y probar --inliers-min más alto (35, 50).")
    L += [f"{'split':8s}{'fold':>5s}{'n':>7s}{'grupos':>8s}{'PCOS':>7s}{'sanas':>7s}"]
    for s, fo in [("cv", kk) for kk in range(args.folds)] + [("holdout", -1)]:
        m = fold == fo
        L.append(f"{s:8s}{fo:>5d}{m.sum():>7d}{len(set(grupos[m])):>8d}{int(y[m].sum()):>7d}{int((1 - y[m]).sum()):>7d}")
    L += ["", f"Tiempo: {time.time() - t0:.0f} s",
          f"Salida: {args.salida} (no pisa {args.splits}). Revisar reports/duplicados_ejemplos.png:",
          "cada fila debería ser UNA misma ecografía con distintas rotaciones/desplazamientos."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_duplicados.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()