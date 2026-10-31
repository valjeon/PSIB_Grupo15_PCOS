"""Características de folículos y estroma por imagen y por ecografía original (módulo de C).

Usa la segmentación de segmentacion.py (`segment_detalle`) y, para cada imagen, calcula:

  Cantidad y tamaño   n_foliculos, densidad, área total/media/mediana/máx/mín, cv_area,
                      dominancia_max (folículo mayor / área total de folículos: ~1 = un
                      quiste grande, ~1/n = muchos parecidos), diámetros equivalentes
                      relativos a sqrt(área del campo)
  Folículo-estroma    frac_area_foliculos, ratio_area_fol_estroma, ratio_brillo_fol_estroma,
                      contraste_michelson, cnr_fol_estroma, frac_area_rechazos
  Estroma             brillo, desvío, entropía y textura GLCM (solo píxeles de estroma)
  Forma               circularidad, excentricidad, solidez y relación de ejes medias
  Distribución        distancia al vecino más cercano e índice de Clark-Evans (R < 1 =
                      apiñados, R ~ 1 = al azar, R > 1 = parejos)
  Interfaz            contraste, CNR y gradiente en el borde del folículo

Definiciones:
  campo (ROI)  = abanico de la ecografía que devuelve la segmentación (NO es el ovario).
  estroma      = campo - folículos - un anillo de `margen_estroma` px alrededor de cada
                 folículo (el borde del folículo es difuso y contaminaría el estroma).
  Todo lo que es proporción (cociente, fracción, relativo a sqrt(área del campo)) no
  depende del zoom; los conteos y las áreas en píxeles sí dependen del recorte.

IMPORTANTE: las 1.699 imágenes son aumentaciones (zoom, rotación, recorte) de solo 8
ecografías (columna `grupo` de splits2.csv: 4 PCOS y 4 sanas). Las filas por imagen son
réplicas, no muestras independientes: el tamaño efectivo es 8. Por eso el script también
resume por grupo (mediana y rango intercuartílico entre las variantes) y compara PCOS vs
sanas con las medianas de los 8 grupos (AUC de grupos y prueba exacta de permutación sobre
las 70 asignaciones posibles). Es descriptivo: con 4 vs 4 el p mínimo posible es 2/70 = 0,029
y con ~35 características ninguna sobrevive a corrección por comparaciones múltiples.

Salidas (reports/): caracteristicas_imagenes.csv, caracteristicas_grupos.csv,
caracteristicas_resumen.txt, caracteristicas_montaje.png (una ecografía representativa por
grupo, con folículos en rojo y estroma en verde) y, con --imagenes N, figuras_caracteristicas/
con una figura por imagen: contornos numerados | estroma y folículos | mediciones.

Uso (desde cualquier carpeta):
    python procesamiento/texturas.py --n-por-grupo 40 --imagenes 3
    python procesamiento/texturas.py --n-por-grupo 0        # todas las imágenes
    python procesamiento/texturas.py --metodo chanvese --filtro frost7 --contraste ninguno
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import itertools
import random
import time

import cv2
import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

from segmentacion import metricas_foliculos, segment_detalle

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"
NAN = float("nan")


# ---------------------------------------------------------------- utilidades
def _dilatar(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _entropia(vals, bins=64):
    h, _ = np.histogram(vals, bins=bins, range=(0, 256))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def glcm_estroma(img, estroma, niveles=16, distancias=(1, 2),
                 angulos=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), min_pix=400):
    """Propiedades de Haralick calculadas SOLO con pares de píxeles que están ambos en el
    estroma. skimage no acepta máscaras: los píxeles fuera del estroma van al nivel 0 y
    se descartan de la matriz antes de normalizar. Promedio sobre distancias y ángulos."""
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    if estroma.sum() < min_pix:
        return {p: NAN for p in props}
    q = np.zeros(img.shape, np.uint8)
    q[estroma] = 1 + np.minimum(img[estroma].astype(np.int32) * niveles // 256, niveles - 1)
    g = graycomatrix(q, list(distancias), list(angulos), levels=niveles + 1,
                     symmetric=True, normed=False)[1:, 1:].astype(np.float64)
    tot = g.sum(axis=(0, 1), keepdims=True)
    if (tot == 0).all():
        return {p: NAN for p in props}
    g = g / np.maximum(tot, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {p: float(np.nanmean(graycoprops(g, p))) for p in props}


# ---------------------------------------------------------------- por imagen
def caracteristicas(d, margen_estroma=3):
    """Dict de características a partir del resultado de segment_detalle()."""
    roi, mask, lab, rech = d["roi"], d["mask"], d["labels"], d["rechazos"]
    base, pre = d["base"], d["pre"]
    a_roi = float(roi.sum())
    estroma = roi & ~_dilatar(mask, margen_estroma) if mask.any() else roi.copy()
    a_fol, a_est = float(mask.sum()), float(estroma.sum())
    escala = np.sqrt(a_roi)

    out = dict(n_foliculos=int(lab.max()), area_roi_px=a_roi,
               densidad_x1e4px=1e4 * int(lab.max()) / a_roi,
               frac_area_foliculos=a_fol / a_roi,
               ratio_area_fol_estroma=a_fol / a_est if a_est > 0 else NAN,
               frac_area_rechazos=float(rech.sum()) / a_roi)

    # ---- estroma
    if a_est > 0:
        pe = pre[estroma].astype(np.float32)
        out.update(brillo_estroma=float(base[estroma].mean()) / 255,
                   estroma_std=float(pe.std()) / 255,
                   estroma_entropia=_entropia(pre[estroma]))
        g = glcm_estroma(pre, estroma)
        out.update({f"glcm_{k}": v for k, v in g.items()})
    else:
        out.update(brillo_estroma=NAN, estroma_std=NAN, estroma_entropia=NAN,
                   glcm_contrast=NAN, glcm_dissimilarity=NAN, glcm_homogeneity=NAN,
                   glcm_energy=NAN, glcm_correlation=NAN)

    props = measure.regionprops(lab)
    if not props:  # sin folículos: lo que no está definido queda en NaN
        out.update(area_total_px=0.0, area_media_px=NAN, area_mediana_px=NAN, area_max_px=NAN,
                   area_min_px=NAN, area_cv=NAN, dominancia_max=NAN, diam_eq_rel_medio=NAN,
                   diam_eq_rel_max=NAN, circularidad_media=NAN, excentricidad_media=NAN,
                   solidez_media=NAN, ratio_ejes_medio=NAN, dist_vecino_rel=NAN,
                   clark_evans_R=NAN, brillo_foliculos=NAN, ratio_brillo_fol_estroma=NAN,
                   contraste_michelson=NAN, cnr_fol_estroma=NAN,
                   contraste_interfaz_medio=NAN, cnr_interfaz_medio=NAN,
                   gradiente_borde_medio=NAN)
        return out

    # ---- tamaño y forma
    a = np.array([r.area for r in props], float)
    deq = np.sqrt(4 * a / np.pi)
    circ = [4 * np.pi * r.area / r.perimeter ** 2 for r in props if r.perimeter > 0]
    out.update(
        area_total_px=float(a.sum()), area_media_px=float(a.mean()),
        area_mediana_px=float(np.median(a)), area_max_px=float(a.max()), area_min_px=float(a.min()),
        area_cv=float(a.std() / a.mean()), dominancia_max=float(a.max() / a.sum()),
        diam_eq_rel_medio=float(deq.mean() / escala), diam_eq_rel_max=float(deq.max() / escala),
        circularidad_media=float(np.mean(circ)) if circ else NAN,
        excentricidad_media=float(np.mean([r.eccentricity for r in props])),
        solidez_media=float(np.mean([r.solidity for r in props])),
        ratio_ejes_medio=float(np.mean([r.axis_minor_length / max(r.axis_major_length, 1e-6)
                                        for r in props])))

    # ---- distribución espacial
    if len(props) >= 2:
        c = np.array([r.centroid for r in props])
        dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
        np.fill_diagonal(dd, np.inf)
        dmin = float(dd.min(1).mean())
        out.update(dist_vecino_rel=dmin / escala,
                   clark_evans_R=dmin / (0.5 * np.sqrt(a_roi / len(props))))
    else:
        out.update(dist_vecino_rel=NAN, clark_evans_R=NAN)

    # ---- folículo vs estroma (intensidad)
    bf = float(base[mask].mean()) / 255
    out["brillo_foliculos"] = bf
    if a_est > 0:
        be = out["brillo_estroma"]
        out["ratio_brillo_fol_estroma"] = bf / be if be > 0 else NAN
        out["contraste_michelson"] = (be - bf) / (be + bf + 1e-6)
        pf, pe = pre[mask].astype(np.float32), pre[estroma].astype(np.float32)
        out["cnr_fol_estroma"] = float((pe.mean() - pf.mean()) / np.sqrt(pe.var() + pf.var() + 1e-6))
    else:
        out.update(ratio_brillo_fol_estroma=NAN, contraste_michelson=NAN, cnr_fol_estroma=NAN)

    # ---- interfaz (de segmentacion.metricas_foliculos)
    fol = metricas_foliculos(d)
    for nombre, clave in (("contraste_interfaz_medio", "contraste_interfaz"),
                          ("cnr_interfaz_medio", "cnr_interfaz"),
                          ("gradiente_borde_medio", "gradiente_borde")):
        out[nombre] = float(np.mean([f[clave] for f in fol])) if fol else NAN
    return out


# ---------------------------------------------------------------- imágenes de resultado
FONT = cv2.FONT_HERSHEY_SIMPLEX  # solo ASCII: las fuentes de OpenCV no dibujan tildes


def _f(v, nd=2):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _zoom_img(img, z):
    return cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)


def _zoom_mask(m, z):
    return np.repeat(np.repeat(m, z, axis=0), z, axis=1)


def panel_foliculos(d, z=2):
    """Imagen base ampliada con el contorno del campo (celeste) y cada folículo numerado."""
    img = _zoom_img(cv2.cvtColor(d["base"], cv2.COLOR_GRAY2RGB), z)
    roi = _zoom_mask(d["roi"], z).astype(np.uint8)
    cr, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cr, -1, (0, 160, 255), 1)
    labz = _zoom_mask(d["labels"], z)
    for r in measure.regionprops(labz):
        cs, _ = cv2.findContours((labz == r.label).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (255, 40, 40), 1)
        txt = str(r.label)
        (w, h), _ = cv2.getTextSize(txt, FONT, 0.4, 1)
        cy, cx = r.centroid
        cv2.putText(img, txt, (int(cx - w / 2), int(cy + h / 2)), FONT, 0.4, (255, 255, 0), 1,
                    cv2.LINE_AA)
    return img


def panel_estroma(d, margen_estroma=3, z=2):
    """Imagen base con el estroma en verde, los folículos en rojo y, con contorno amarillo,
    las zonas oscuras que la segmentación descartó (sombras, estroma oscuro)."""
    img = _zoom_img(cv2.cvtColor(d["base"], cv2.COLOR_GRAY2RGB), z).astype(np.float32)
    roi, mask = _zoom_mask(d["roi"], z), _zoom_mask(d["mask"], z)
    estroma = roi & ~_dilatar(mask, margen_estroma * z) if mask.any() else roi
    img[estroma] = 0.65 * img[estroma] + 0.35 * np.array([0, 200, 0])
    img[mask] = 0.45 * img[mask] + 0.55 * np.array([255, 0, 0])
    out = img.astype(np.uint8)
    rech = _zoom_mask(d["rechazos"], z)
    if rech.any():
        cr, _ = cv2.findContours(rech.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cr, -1, (255, 220, 0), 1)
    return out


def panel_texto(d, c, titulo, lado):
    """Panel oscuro con las mediciones de la imagen y la tabla de folículos."""
    p = np.full((lado, lado, 3), 30, np.uint8)
    L = [titulo, "",
         f"foliculos: {c['n_foliculos']}    campo (ROI): {_f(c['area_roi_px'], 0)} px",
         f"area fol. (px): total {_f(c['area_total_px'], 0)} | media {_f(c['area_media_px'], 0)}"
         f" | max {_f(c['area_max_px'], 0)}",
         f"fraccion area fol/campo:    {_f(c['frac_area_foliculos'], 3)}",
         f"relacion area fol/estroma:  {_f(c['ratio_area_fol_estroma'], 3)}",
         f"brillo fol/estroma: {_f(c['ratio_brillo_fol_estroma'])}"
         f" (Michelson {_f(c['contraste_michelson'])})",
         f"CNR fol-estroma: {_f(c['cnr_fol_estroma'])}",
         f"circularidad {_f(c['circularidad_media'])} | excentr. {_f(c['excentricidad_media'])}",
         f"Clark-Evans R: {_f(c['clark_evans_R'])}  (<1 apinados)",
         f"estroma: brillo {_f(c['brillo_estroma'])} | GLCM contr. {_f(c['glcm_contrast'])}"
         f" | homog. {_f(c['glcm_homogeneity'])}",
         "", "id   area(px)  circ.   (rojo en la imagen)"]
    props = measure.regionprops(d["labels"])
    ancho_fila, y = 17, 22
    libres = (lado - y - ancho_fila * len(L)) // ancho_fila
    for r in props[:max(libres, 0)]:
        circ = 4 * np.pi * r.area / r.perimeter ** 2 if r.perimeter > 0 else float("nan")
        L.append(f"{r.label:<4d} {r.area:8.0f}  {_f(circ)}")
    if len(props) > max(libres, 0):
        L.append(f"... (+{len(props) - max(libres, 0)} mas)")
    for s in L:
        cv2.putText(p, s, (10, y), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        y += ancho_fila
    return p


def figura_imagen(d, c, titulo, margen_estroma=3, z=2):
    """Tres paneles: contornos numerados | estroma (verde) y folículos (rojo) | mediciones."""
    a = panel_foliculos(d, z)
    return np.hstack([a, panel_estroma(d, margen_estroma, z), panel_texto(d, c, titulo, a.shape[0])])


def montaje_grupos(items, cols=4, pie=40):
    """items: lista de (panel RGB cuadrado, [líneas de pie]). Devuelve la hoja RGB."""
    lado = items[0][0].shape[0]
    filas = -(-len(items) // cols)
    hoja = np.full((filas * (lado + pie), cols * lado, 3), 20, np.uint8)
    for k, (panel, lineas) in enumerate(items):
        r, col = divmod(k, cols)
        y0, x0 = r * (lado + pie), col * lado
        hoja[y0:y0 + lado, x0:x0 + lado] = panel
        for j, s in enumerate(lineas):
            cv2.putText(hoja, s, (x0 + 6, y0 + lado + 16 + 16 * j), FONT, 0.42, (235, 235, 235), 1,
                        cv2.LINE_AA)
    return hoja


# ---------------------------------------------------------------- por grupo
def resumir_grupos(filas, claves):
    """mediana y rango intercuartílico (entre las variantes) de cada característica, por grupo."""
    por_grupo = {}
    for f in filas:
        por_grupo.setdefault(f["grupo"], []).append(f)
    res = {}
    for g, fs in sorted(por_grupo.items(), key=lambda kv: int(kv[0])):
        fila = dict(grupo=g, clase=fs[0]["clase"], label=fs[0]["label"], n_imagenes=len(fs))
        for k in claves:
            v = np.array([x[k] for x in fs], float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                fila[f"{k}_med"], fila[f"{k}_iqr"] = NAN, NAN
            else:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                fila[f"{k}_med"], fila[f"{k}_iqr"] = float(med), float(q3 - q1)
        res[g] = fila
    return res


def auc_grupos(v, et):
    pos, neg = v[et == 1], v[et == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def permutacion_exacta(v, et):
    """p bilateral exacto: diferencia de medias entre grupos PCOS y sanos, sobre todas las
    asignaciones posibles de las etiquetas (70 con 4 vs 4)."""
    obs = abs(v[et == 1].mean() - v[et == 0].mean())
    n, k = len(v), int(et.sum())
    cuenta = total = 0
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        cuenta += abs(v[m].mean() - v[~m].mean()) >= obs - 1e-12
        total += 1
    return cuenta / total


def comparar(grupos, claves):
    gs = list(grupos.values())
    et = np.array([int(g["label"]) for g in gs])
    if et.sum() == 0 or et.sum() == len(et):
        return []
    filas = []
    for k in claves:
        v = np.array([g[f"{k}_med"] for g in gs], float)
        if not np.isfinite(v).all():
            continue
        iqr = np.array([g[f"{k}_iqr"] for g in gs], float)
        rango = v.max() - v.min()
        filas.append(dict(
            caracteristica=k, med_pcos=float(np.median(v[et == 1])), med_sana=float(np.median(v[et == 0])),
            auc_grupos=auc_grupos(v, et), p_exacto=permutacion_exacta(v, et),
            inestabilidad=float(np.mean(iqr) / rango) if rango > 0 else NAN))
    return sorted(filas, key=lambda f: (f["p_exacto"], -abs(f["auc_grupos"] - 0.5)))


# ---------------------------------------------------------------- entrada / salida
def resolver_imagen(path_rel, clase):
    """data/clean/... si existe; si no, la imagen original en data/raw/<split>/<clase>/."""
    p = ROOT / path_rel
    if p.exists():
        return p
    pref, _, resto = Path(path_rel).name.partition("_")
    raw = ROOT / "data" / "raw" / pref / clase / resto
    return raw if raw.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default="splits2.csv", help="CSV con path, label, clase, grupo")
    ap.add_argument("--n-por-grupo", type=int, default=40, help="0 = todas las imágenes")
    ap.add_argument("--metodo", default="multiotsu")
    ap.add_argument("--filtro", default="mediana5")
    ap.add_argument("--contraste", default="estirar")
    ap.add_argument("--margen-estroma", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imagenes", type=int, default=0,
                    help="guardar una figura con las mediciones de N imágenes por grupo")
    ap.add_argument("--sin-montaje", action="store_true", help="no generar el montaje por grupo")
    args = ap.parse_args()

    with open(ROOT / args.splits, encoding="utf-8") as f:
        filas_csv = list(csv.DictReader(f))
    por_grupo = {}
    for r in filas_csv:
        por_grupo.setdefault(r["grupo"], []).append(r)
    rng = random.Random(args.seed)
    muestra = []
    for g, rs in sorted(por_grupo.items(), key=lambda kv: int(kv[0])):
        rs = sorted(rs, key=lambda r: r["path"])
        muestra += rs if args.n_por_grupo <= 0 else rng.sample(rs, min(args.n_por_grupo, len(rs)))

    filas, claves, t0, saltadas = [], None, time.time(), 0
    guardadas = defaultdict(int)
    carpeta_fig = REPORTES / "figuras_caracteristicas"
    for i, r in enumerate(muestra, 1):
        ruta = resolver_imagen(r["path"], r["clase"])
        if ruta is None:
            saltadas += 1
            continue
        d = segment_detalle(ruta, filtro=args.filtro, contraste=args.contraste, metodo=args.metodo)
        f = caracteristicas(d, args.margen_estroma)
        claves = claves or list(f)
        filas.append(dict(path=r["path"], clase=r["clase"], label=r["label"], grupo=r["grupo"], **f))
        if guardadas[r["grupo"]] < args.imagenes:
            guardadas[r["grupo"]] += 1
            carpeta_fig.mkdir(parents=True, exist_ok=True)
            fig = figura_imagen(d, f, f"{r['clase']} | grupo {r['grupo']} | {ruta.name[:36]}",
                                args.margen_estroma)
            cv2.imwrite(str(carpeta_fig / f"g{r['grupo']}__{r['clase']}__{ruta.stem}.png"),
                        cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            print(f"  {i}/{len(muestra)}  ({time.time() - t0:.0f} s)", flush=True)
    if not filas:
        raise SystemExit("No encontré ninguna imagen (ni en data/clean ni en data/raw).")

    REPORTES.mkdir(exist_ok=True)
    with open(REPORTES / "caracteristicas_imagenes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    grupos = resumir_grupos(filas, claves)
    with open(REPORTES / "caracteristicas_grupos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(grupos.values()))))
        w.writeheader()
        w.writerows(grupos.values())

    if not args.sin_montaje:
        # una imagen representativa por grupo: la que está más cerca de la mediana del grupo
        items = []
        for g in sorted(grupos.values(), key=lambda g: (-int(g["label"]), int(g["grupo"]))):
            fs = [x for x in filas if x["grupo"] == g["grupo"]]
            rep = min(fs, key=lambda x: abs(x["frac_area_foliculos"] - g["frac_area_foliculos_med"]))
            d = segment_detalle(resolver_imagen(rep["path"], rep["clase"]), filtro=args.filtro,
                                contraste=args.contraste, metodo=args.metodo)
            items.append((panel_estroma(d, args.margen_estroma, 1),
                          [f"g{g['grupo']} {rep['clase']}  n={rep['n_foliculos']}",
                           f"fol/campo {_f(rep['frac_area_foliculos'], 3)}  "
                           f"fol/estroma {_f(rep['ratio_area_fol_estroma'], 3)}"]))
        cv2.imwrite(str(REPORTES / "caracteristicas_montaje.png"),
                    cv2.cvtColor(montaje_grupos(items), cv2.COLOR_RGB2BGR))

    comp = comparar(grupos, claves)
    L = [f"Imágenes procesadas: {len(filas)} de {len(muestra)} ({saltadas} sin archivo) | "
         f"grupos: {len(grupos)} | segmentación: {args.metodo}+{args.filtro}+{args.contraste}",
         "Grupos (= ecografías originales):"]
    for g in grupos.values():
        L.append(f"  grupo {g['grupo']}: {g['clase']:12s} n={g['n_imagenes']:4d}  "
                 f"folículos (mediana) = {g['n_foliculos_med']:.0f}")
    L += ["", "PCOS vs sanas con las medianas de los grupos (n efectivo = 8: 4 vs 4)",
          f"{'característica':28s}{'PCOS':>10s}{'sana':>10s}{'AUC':>7s}{'p exacto':>10s}{'inestab.':>10s}"]
    for c in comp:
        L.append(f"{c['caracteristica']:28s}{c['med_pcos']:10.3f}{c['med_sana']:10.3f}"
                 f"{c['auc_grupos']:7.2f}{c['p_exacto']:10.3f}{c['inestabilidad']:10.2f}")
    L += ["", "AUC: sobre medianas de grupo (1 = todas las PCOS por encima de todas las sanas, 0,5 = nada).",
          "p exacto: permutación de las 70 asignaciones; mínimo posible 0,029; SIN corregir por",
          f"  comparaciones múltiples ({len(comp)} características): tomarlo como orientativo.",
          "inestab.: rango intercuartílico entre variantes / rango entre grupos. < 1 = la diferencia",
          "  entre ecografías supera lo que cambia una misma ecografía al rotarla, hacerle zoom o recortarla.",
          "Con 8 ecografías no se estima una capacidad diagnóstica: se describe qué mide cada característica."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "caracteristicas_resumen.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
