"""Limpieza y partición del dataset (módulo de A).

    python procesamiento/limpieza.py limpiar      # data/all -> data/clean (ilegibles, vacías, duplicadas)
    python procesamiento/limpieza.py auditar      # tamaños, marcas y control de atajos por metadatos
    python procesamiento/limpieza.py splits       # grupos pHash + hold-out + folds -> splits.csv
    python procesamiento/limpieza.py duplicados   # grupos ORB+RANSAC (rotadas/recortadas) -> splits2.csv

Orden: limpiar -> auditar -> splits -> duplicados. Cada comando acepta -h.
"""
from collections import Counter, defaultdict, namedtuple
import argparse
import csv
import random
import shutil
import statistics as st
import sys
import time

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageStat
from sklearn.model_selection import StratifiedGroupKFold

from preprocesamiento import a_gris, leer, recortar_util

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================
# COMANDO limpiar: data/all -> data/clean
# ============================================================================
SRC = ROOT / "data" / "all"
DST = ROOT / "data" / "clean"
UMBRAL = 4      # distancia de Hamming máxima entre pHash para considerar "duplicada"
STD_MIN = 2.0   # desvío estándar de grises por debajo del cual la imagen es casi vacía

Img = namedtuple("Img", "area clase ruta hash size mode")


def distancia(a, b):
    return bin(a ^ b).count("1")


def origen(nombre):
    """merge_data.py antepone 'train_' o 'test_' al nombre."""
    pref = nombre.split("_", 1)[0]
    return pref if pref in ("train", "test") else "?"


def fila(clase, archivo, motivo, grupo="", dist="", o_grupo=""):
    return {"clase": clase, "archivo": archivo, "motivo": motivo, "grupo": grupo,
            "distancia": dist, "origen": origen(archivo), "origen_grupo": o_grupo}


def cmd_limpiar(argv=None):
    """Limpieza de data/all -> data/clean.

Excluye: archivos que no abren, imágenes casi vacías, duplicados (pHash) y
grupos de duplicados con etiquetas contradictorias (misma imagen en dos clases).
Registra todo en reports/excluidas.csv y reports/resumen_limpieza.txt.
"""
    excluidas = []
    candidatas = []

    # 1) Lectura: archivos corruptos o casi vacíos se descartan acá
    for clase_dir in sorted(SRC.iterdir()):
        if not clase_dir.is_dir():
            continue
        for ruta in sorted(clase_dir.iterdir()):
            try:
                with Image.open(ruta) as img:
                    img.load()  # detecta también archivos truncados
                    gris = img.convert("L")
                    desvio = ImageStat.Stat(gris).stddev[0]
                    h = int(str(imagehash.phash(img)), 16)
                    size, mode = img.size, img.mode
            except Exception:
                excluidas.append(fila(clase_dir.name, ruta.name, "no se pudo abrir"))
                continue
            if desvio < STD_MIN:
                excluidas.append(fila(clase_dir.name, ruta.name, "casi vacía"))
                continue
            candidatas.append(Img(size[0] * size[1], clase_dir.name, ruta, h, size, mode))

    # 2) Agrupado: de mayor a menor resolución; cada imagen se asigna al
    #    representante más cercano (si está a <= UMBRAL) o funda un grupo nuevo.
    candidatas.sort(key=lambda c: (-c.area, c.ruta.name))
    grupos = []  # cada grupo: [representante, [(miembro, distancia), ...]]
    for c in candidatas:
        mejor, d_mejor = None, UMBRAL + 1
        for g in grupos:
            d = distancia(g[0].hash, c.hash)
            if d < d_mejor:
                mejor, d_mejor = g, d
        if mejor is None:
            grupos.append([c, []])
        else:
            mejor[1].append((c, d_mejor))

    # 3) Decisión por grupo
    conservadas = []
    n_conflictos = 0
    for rep, miembros in grupos:
        clases = {rep.clase} | {m.clase for m, _ in miembros}
        if len(clases) > 1:
            # misma imagen con etiquetas distintas: no se sabe cuál es la correcta
            n_conflictos += 1
            excluidas.append(fila(rep.clase, rep.ruta.name, "conflicto de clase",
                                  rep.ruta.name, 0, origen(rep.ruta.name)))
            for m, d in miembros:
                excluidas.append(fila(m.clase, m.ruta.name, "conflicto de clase",
                                      rep.ruta.name, d, origen(rep.ruta.name)))
            continue
        conservadas.append(rep)
        for m, d in miembros:
            excluidas.append(fila(m.clase, m.ruta.name, "duplicada",
                                  rep.ruta.name, d, origen(rep.ruta.name)))

    # 4) data/clean se regenera siempre desde cero
    if DST.exists():
        shutil.rmtree(DST)
    for c in conservadas:
        (DST / c.clase).mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.ruta, DST / c.clase / c.ruta.name)

    REPORTES.mkdir(exist_ok=True)
    campos = ["clase", "archivo", "motivo", "grupo", "distancia", "origen", "origen_grupo"]
    with open(REPORTES / "excluidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(excluidas)

    # 5) Resumen
    motivos = Counter(e["motivo"] for e in excluidas)
    dup_origen = Counter("-".join(sorted((e["origen"], e["origen_grupo"])))
                         for e in excluidas if e["motivo"] == "duplicada")
    total = len(candidatas) + motivos["no se pudo abrir"] + motivos["casi vacía"]

    lineas = [
        f"Umbral pHash: {UMBRAL} | desvío mínimo: {STD_MIN}",
        f"Imágenes en {SRC}: {total}",
        f"Excluidas: {len(excluidas)}",
        f"  no se pudo abrir: {motivos['no se pudo abrir']}",
        f"  casi vacías: {motivos['casi vacía']}",
        f"  duplicadas: {motivos['duplicada']}",
        "    por origen (copia - original conservado):",
        *[f"      {k}: {v}" for k, v in sorted(dup_origen.items())],
        f"  conflicto de clase: {motivos['conflicto de clase']} imágenes en {n_conflictos} grupos",
        f"Conservadas: {len(conservadas)}",
        *[f"  {cl}: {n}" for cl, n in sorted(Counter(c.clase for c in conservadas).items())],
        "  por origen: " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(Counter(origen(c.ruta.name) for c in conservadas).items())),
        "Modos de color: " + ", ".join(f"{k}={v}" for k, v in
                                        Counter(c.mode for c in conservadas).most_common()),
        "Tamaños más frecuentes: " + ", ".join(f"{s[0]}x{s[1]}={v}" for s, v in
                                                Counter(c.size for c in conservadas).most_common(5)),
        f"Tamaños distintos: {len({c.size for c in conservadas})}",
    ]
    texto = "\n".join(lineas)
    print(texto)
    (REPORTES / "resumen_limpieza.txt").write_text(texto + "\n", encoding="utf-8")


# ============================================================================
# COMANDO auditar: sesgos de adquisición en data/clean
# ============================================================================
def medir(ruta, sin_color=False):
    with Image.open(ruta) as im:
        ancho, alto = im.size
        modo = im.mode
        gris = np.asarray(im.convert("L"), dtype=np.float32)
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    color = (rgb.max(axis=2) - rgb.min(axis=2)) > 40  # píxeles con color marcado
    g = gris[~color] if (sin_color and (~color).any()) else gris.ravel()
    return {
        "path": ruta.relative_to(ROOT).as_posix(),
        "clase": ruta.parent.name,
        "origen": origen(ruta.name),
        "ancho": ancho,
        "alto": alto,
        "modo": modo,
        "bytes": ruta.stat().st_size,
        "brillo": float(g.mean()),
        "desvio": float(g.std()),
        # píxeles casi blancos: texto, calibres, marcas de escala
        "frac_sat": float((g >= 250).mean()),
        # píxeles con color marcado: anotaciones amarillas/verdes sobre fondo gris
        "frac_color": float(color.mean()),
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


def sesgo_por_metadatos(filas, seed, excluir=()):
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
    nombres = ["ancho", "alto", "aspecto", "bytes", "brillo", "desvio", "frac_sat", "frac_color"]
    X = np.array([[f["ancho"], f["alto"], f["ancho"] / f["alto"], f["bytes"], f["brillo"],
                   f["desvio"], f["frac_sat"], f["frac_color"]] for f in filas])
    y = np.array([f["clase"] == clases[1] for f in filas], dtype=int)
    keep = [k for k, nom in enumerate(nombres) if nom not in excluir]
    nombres = [nombres[k] for k in keep]
    X = X[:, keep]
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    p = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    # AUC de cada característica sola (simetrizado: 0,5 = no separa, 1 = separa perfecto)
    solas = {}
    for k, nom in enumerate(nombres):
        a = roc_auc_score(y, X[:, k])
        solas[nom] = max(a, 1 - a)
    return roc_auc_score(y, p), solas


def cmd_auditar(argv=None):
    """Auditoría de data/clean (correr después de limpieza.py limpiar).

Genera en reports/:
  audit.csv                    una fila por imagen (tamaño, modo, brillo, saturación, color)
  resumen_auditoria.txt        tabla por clase + chequeo de sesgos
  muestra_azar_<clase>.jpg     montaje de imágenes al azar para revisión visual
  muestra_marcas_<clase>.jpg   las imágenes con más píxeles saturados/coloreados
                               (donde es más probable que haya texto, calibres o logos)

Idea central: si las clases difieren en algo que no es el ovario (tamaño, modo de
color, marcas grabadas), el modelo puede aprender eso en lugar de la patología.
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-azar", type=int, default=50, help="imágenes al azar por clase")
    ap.add_argument("--n-marcas", type=int, default=40, help="imágenes 'sospechosas' por clase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sin-color", action="store_true",
                    help="estadísticas de gris sin los píxeles con color; excluye frac_color del chequeo")
    args = ap.parse_args(argv)
    sufijo = "_sin_color" if args.sin_color else ""

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste limpieza.py limpiar?")
    filas = [medir(r, args.sin_color) for r in rutas]
    REPORTES.mkdir(exist_ok=True)

    with open(REPORTES / f"audit{sufijo}.csv", "w", newline="", encoding="utf-8") as f:
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

    if args.sin_color:
        L.insert(1, "MODO --sin-color: brillo, desvío y saturación sin píxeles coloreados; "
                    "frac_color fuera del chequeo de atajos.")
    res = sesgo_por_metadatos(filas, args.seed, excluir=("frac_color",) if args.sin_color else ())
    if res is not None:
        auc, solas = res
        veredicto = ("ALERTA: la clase se predice sin mirar el ovario; hay sesgo de adquisición"
                     if auc >= 0.65 else "sin señal fuerte de sesgo")
        L += [f"Chequeo de atajos: AUC solo con metadatos (tamaño, bytes, brillo, saturación, "
              f"color) = {auc:.3f} -> {veredicto}",
              "  (CV sin agrupar: los near-duplicados lo inflan un poco. ~0,5 es lo esperable.)",
              "  AUC de cada característica sola (0,5 = no separa): " +
              ", ".join(f"{k}={v:.2f}" for k, v in sorted(solas.items(), key=lambda kv: -kv[1])), ""]
    L += ["Revisión visual pendiente (a mano):",
          f"  - {REPORTES}/muestra_azar_<clase>.jpg y muestra_marcas_<clase>.jpg",
          "  - anotar en el informe: ¿hay texto, calibres, logos? ¿en qué proporción por clase?"]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / f"resumen_auditoria{sufijo}.txt").write_text(texto + "\n", encoding="utf-8")


# ============================================================================
# COMANDO splits: hold-out y folds por grupos pHash
# ============================================================================
CLASE_POSITIVA = "infected"


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


def cmd_splits(argv=None):
    """Genera splits.csv sin fuga entre folds (correr después de limpieza.py limpiar).

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
    args = ap.parse_args(argv)

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste limpieza.py limpiar?")
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


# ============================================================================
# COMANDO duplicados: agrupa copias rotadas/recortadas (ORB+RANSAC) y rehace la partición
# ============================================================================
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
def cmd_duplicados(argv=None):
    """Agrupa copias aumentadas de una misma ecografía aunque estén ROTADAS, DESPLAZADAS,
ESPEJADAS o con ZOOM, y rehace la partición sin fuga.

Por qué: limpieza.py splits agrupa con pHash, que solo reconoce copias casi idénticas. Una
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
  5. Hold-out y folds con StratifiedGroupKFold, igual que limpieza.py splits.

NO pisa splits.csv. Salidas:
  splits2.csv                     mismas columnas que splits.csv (+ grupo_phash)
  reports/resumen_duplicados.txt  cuántos grupos, cuántas copias, verificaciones
  reports/duplicados_ejemplos.png los grupos más grandes, para revisar a ojo
  reports/duplicados_pares.csv    cada vínculo encontrado con su cantidad de inliers

Uso (desde procesamiento/):
    python limpieza.py duplicados
    python segmentacion.py comparar --reumbral --splits splits2.csv
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.csv", help="partición actual (se usa su lista de imágenes y sus grupos pHash)")
    ap.add_argument("--salida", default="splits2.csv")
    ap.add_argument("--vecinos", type=int, default=40, help="candidatos por imagen a verificar")
    ap.add_argument("--inliers-min", type=int, default=25)
    ap.add_argument("--features", type=int, default=800)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(ROOT / args.splits, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    rutas = [ROOT / r["path"] for r in filas]
    y = np.array([int(r["label"]) for r in filas])
    n = len(rutas)
    faltan = [r for r in rutas if not r.exists()]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} imágenes (ej. {faltan[0]}). ¿Corriste limpieza.py limpiar?")

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


COMANDOS = {
    "limpiar": cmd_limpiar,
    "auditar": cmd_auditar,
    "splits": cmd_splits,
    "duplicados": cmd_duplicados,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print("Uso: python procesamiento/limpieza.py <comando> [opciones]\n\nComandos:")
        for k, f in COMANDOS.items():
            print(f"  {k:18s}{(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        print("\nPara ver las opciones de uno: python procesamiento/limpieza.py <comando> -h")
        raise SystemExit(1)
    COMANDOS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
