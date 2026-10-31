"""Segmentación de folículos (módulo de B). Usa el preprocesamiento de preprocesamiento.py.

Pipeline de `segment(entrada)`:
  1. imagen base: sin marcas, recortada y a tamaño común, CONSERVANDO el aspecto
     (relleno negro). Estirar a cuadrado deforma los folículos y falsea
     circularidad, excentricidad y ejes, que después usa C.
  2. máscara del abanico (zona con señal). Sin ella, Otsu toma el fondo negro y
     la sombra del transductor como si fueran folículos.
  3. filtro de speckle + contraste (los mismos nombres que en preprocesamiento.py);
     el estiramiento se calcula solo dentro del abanico.
  4. umbral: los folículos son la clase más oscura (Otsu de 3 clases por defecto,
     Otsu clásico o Chan-Vese morfológico para comparar).
  5. morfología: apertura, cierre, relleno de huecos.
  6. descarta lo que toca el borde del abanico.
  7. watershed sobre la transformada de distancia (separa folículos pegados).
  8. filtra regiones por área, circularidad y solidez.

Como módulo:
    from segmentacion import segment, segment_detalle, resumen_imagen
    mask = segment("data/clean/infected/x.jpg", filtro="mediana5", contraste="estirar")
    d = segment_detalle("data/clean/infected/x.jpg")   # base, pre, roi, labels, mask
    fila = resumen_imagen(d)                             # n_foliculos, interfaz, etc.

Como script:
    python procesamiento/segmentacion.py comparar          # compara combinaciones (conteo / Dice / --vista)
    python procesamiento/segmentacion.py ver-casos         # peores conteos de la última comparación
    python procesamiento/segmentacion.py anotar-mascaras   # dibujar folículos a mano (referencia)
"""
from collections import Counter, defaultdict
import argparse
import csv
import random
import sys
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from skimage import feature, filters, measure, segmentation

from preprocesamiento import (a_gris, armar_muestra, contrastar, filtrar, hoja, leer,
                              quitar_marcas, recortar_util)

from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================
# SEGMENTACIÓN DE FOLÍCULOS
# ============================================================================
# ---------------------------------------------------------------- imagen base
def base_aspecto(entrada, tam=256, marcas=True, texto=True):
    """Como preprocess.base, pero escala el lado mayor a `tam` y rellena con negro.

    El relleno negro queda fuera del abanico, así que no entra en ninguna medida.
    """
    rgb = leer(entrada) if isinstance(entrada, (str, Path)) else entrada
    gris = quitar_marcas(rgb, quitar_texto=texto)[0] if marcas else a_gris(rgb)
    gris = recortar_util(gris)
    h, w = gris.shape
    s = tam / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(gris, (nw, nh), interpolation=interp)
    out = np.zeros((tam, tam), np.uint8)
    y0, x0 = (tam - nh) // 2, (tam - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def mascara_relleno(g, largo=21, rango_h=10, rango_v=25, area_min=150):
    """Bandas horizontales que dejó la aumentación del dataset (relleno replicando el
    borde). En esas bandas cada fila es casi constante a lo largo de x (rango
    horizontal chico en una ventana de `largo` píxeles) pero cambia mucho de una fila
    a otra (rango vertical grande). El interior de un folículo también es parejo,
    pero en las dos direcciones, así que no cae acá."""
    kh = np.ones((1, largo), np.uint8)
    kv = np.ones((largo, 1), np.uint8)
    rh = cv2.dilate(g, kh).astype(np.int16) - cv2.erode(g, kh)
    rv = cv2.dilate(g, kv).astype(np.int16) - cv2.erode(g, kv)
    m = ((rh <= rango_h) & (rv >= rango_v)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 9), np.uint8))
    m = _componentes_grandes(m, area_min)
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _componentes_grandes(binaria, area_min):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8))
    ids = np.where(stats[:, cv2.CC_STAT_AREA] >= area_min)[0]
    return np.isin(lab, ids[ids > 0])


def mascara_abanico(g, umbral=12, erosion=4, sacar_relleno=True):
    """Componente no negra más grande, con huecos rellenos (los folículos anecoicos
    son casi negros y quedan adentro), sin las bandas de relleno de la aumentación,
    y erosionada para no pegarse al borde."""
    m = (cv2.GaussianBlur(g, (5, 5), 0) > umbral).astype(np.uint8)
    if sacar_relleno:
        m[mascara_relleno(g)] = 0
        # relleno blanco de las rotaciones de la aumentación (triángulos saturados)
        m[_componentes_grandes(cv2.dilate((g >= 250).astype(np.uint8),
                                          np.ones((5, 5), np.uint8)), 150)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones_like(g, bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = ndi.binary_fill_holes(lab == i)
    if erosion > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
        roi = cv2.erode(roi.astype(np.uint8), k).astype(bool)
    return roi


def _estirar_roi(img, roi, p=(1, 99)):
    lo, hi = np.percentile(img[roi], p)
    if hi <= lo:
        return img
    return (np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def preprocesar_roi(g, roi, filtro="mediana5", contraste="estirar"):
    f = filtrar(g, filtro)
    if contraste == "estirar":
        return _estirar_roi(f, roi)
    return contrastar(f, contraste)


# ---------------------------------------------------------------- segmentación
def _umbral_oscuro(pre, roi, metodo, sigma=2.0):
    """El umbral se aplica sobre una versión suavizada: el speckle forma manchitas
    oscuras de pocos píxeles que, sin suavizar, caen en la clase más oscura."""
    if sigma > 0:
        pre = cv2.GaussianBlur(pre, (0, 0), sigma)
    vals = pre[roi]
    if metodo == "multiotsu":
        try:
            t = filters.threshold_multiotsu(vals, classes=3)[0]
        except ValueError:  # muy pocos niveles de gris
            t = filters.threshold_otsu(vals)
        return (pre < t) & roi
    if metodo == "otsu":
        return (pre < filters.threshold_otsu(vals)) & roi
    if metodo == "chanvese":
        ls = segmentation.morphological_chan_vese(pre.astype(float) / 255, 150, smoothing=2)
        if not ls.any() or ls.all():
            return np.zeros_like(roi)
        fase = 1 if pre[ls == 1].mean() < pre[ls == 0].mean() else 0
        return (ls == fase) & roi
    if metodo == "triangle":
        return (pre < filters.threshold_triangle(vals)) & roi
    if metodo == "li":
        return (pre < filters.threshold_li(vals)) & roi
    if metodo == "adaptativo":
        return (pre < filters.threshold_local(pre, block_size=51, offset=15)) & roi
    raise ValueError(f"método desconocido: {metodo}")


def _reumbralar(lab, pre, area_max, elong_min, area_min, k, niveles=2):
    out = lab.copy()
    for _ in range(niveles):
        nuevo = np.zeros_like(out)
        siguiente = out.max() + 1
        cambio = False
        for r in measure.regionprops(out):
            elong = r.axis_minor_length / max(r.axis_major_length, 1e-6)
            if r.area <= area_max and elong >= elong_min and r.solidity >= 0.85:
                nuevo[out == r.label] = r.label
                continue
            reg = out == r.label
            vals = pre[reg]
            if vals.max() - vals.min() < 20:
                continue  # región pareja: no hay nada más oscuro adentro
            t = filters.threshold_otsu(vals)
            sub = (reg & (pre < t)).astype(np.uint8)
            sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k)
            sub_lab = measure.label(sub)
            for s_r in measure.regionprops(sub_lab):
                if s_r.area >= area_min:
                    nuevo[sub_lab == s_r.label] = siguiente
                    siguiente += 1
            cambio = True
        out = nuevo
        if not cambio:
            break
    return out


def segment_detalle(entrada, filtro="mediana5", contraste="estirar", tam=256,
                    conservar_aspecto=True, metodo="multiotsu",
                    area_min=80, area_max_frac=0.05, circ_min=0.50, solidez_min=0.85,
                    reumbral=True,
                    elong_min=0.35, contraste_min=30, contacto_max=0.25, anillo=4,
                    sigma_umbral=2.0,
                    separar=True, min_dist=7):
    """Devuelve un dict con base, pre, roi, labels (folículos), mask y rechazos.

    elong_min: eje menor / eje mayor mínimo (descarta bandas alargadas).
    contacto_max: fracción máxima del anillo que puede tocar OTRA región oscura. Cuando
        el watershed corta una banda oscura en pedazos, cada pedazo es redondeado pero
        está pegado a sus vecinos sin tabique claro; dos folículos vecinos, en cambio,
        están separados por un tabique brillante.
    contraste_min: diferencia mínima (en niveles de gris, 0-255) entre el anillo de
    estroma que rodea a la región y el interior. Un folículo es anecoico (negro) con
    borde nítido; una zona oscura de estroma o una sombra acústica no tienen ese salto.
    """
    if conservar_aspecto:
        g = base_aspecto(entrada, tam)
    else:
        from preprocess import base
        g = base(entrada, tam)
    roi = mascara_abanico(g)
    pre = preprocesar_roi(g, roi, filtro, contraste)

    m = _umbral_oscuro(pre, roi, metodo, sigma_umbral).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = ndi.binary_fill_holes(m)

    lab = measure.label(m)

    # Re-umbral de regiones demasiado grandes o alargadas: cuando el estroma que
    # rodea a folículos chicos también es oscuro, el umbral global los funde con él
    # en una sola banda. Dentro de esa banda, un Otsu local separa lo más oscuro.
    if reumbral:
        lab = _reumbralar(lab, pre, area_max_frac * roi.sum(), elong_min, area_min, k)
    # lo que sigue tocando el borde del abanico se descarta (después del re-umbral,
    # para no perder los folículos de una banda que llega al borde)
    lab = segmentation.clear_border(lab, mask=roi)

    if separar and lab.max() > 0:
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(lab > 0), 1)
        picos = feature.peak_local_max(dist, min_distance=min_dist, labels=lab,
                                       exclude_border=False)
        marc = np.zeros(lab.shape, np.int32)
        marc[tuple(picos.T)] = np.arange(1, len(picos) + 1)
        # regiones sin pico (muy chicas) conservan su etiqueta
        sin_pico = (lab > 0) & ~np.isin(lab, np.unique(lab[marc > 0]))
        lab = segmentation.watershed(-dist, marc, mask=(lab > 0) & ~sin_pico)
        if sin_pico.any():
            extra = measure.label(sin_pico)
            lab[extra > 0] = extra[extra > 0] + lab.max()

    area_max = area_max_frac * roi.sum()
    final = np.zeros(lab.shape, np.int32)
    rechazos = np.zeros(lab.shape, np.int32)
    ka = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * anillo + 1, 2 * anillo + 1))
    libre = roi & (lab == 0)
    oscuro_total = lab > 0
    prefl = pre.astype(np.float32)
    k_id = 0
    for r in measure.regionprops(lab):
        if r.perimeter == 0:
            continue
        circ = 4 * np.pi * r.area / r.perimeter ** 2
        elong = r.axis_minor_length / max(r.axis_major_length, 1e-6)
        ok = (area_min <= r.area <= area_max and circ >= circ_min
              and r.solidity >= solidez_min and elong >= elong_min)
        if ok:
            m_r = (lab == r.label).astype(np.uint8)
            anillo_r = cv2.dilate(m_r, ka).astype(bool) & (m_r == 0) & roi
            fuera = anillo_r & libre
            contacto = (anillo_r & oscuro_total).sum() / max(anillo_r.sum(), 1)
            ok = fuera.sum() > 0 and contacto <= contacto_max
            if ok and contraste_min > 0:
                ok = prefl[fuera].mean() - prefl[m_r > 0].mean() >= contraste_min
        if ok:
            k_id += 1
            final[lab == r.label] = k_id
        else:
            rechazos[lab == r.label] = r.label

    return dict(base=g, pre=pre, roi=roi, labels=final, mask=final > 0,
                rechazos=rechazos > 0, n_foliculos=k_id,
                params=dict(filtro=filtro, contraste=contraste, metodo=metodo))


def segment(entrada, **kw):
    """img -> mask (bool). Es la función que usa el resto del proyecto."""
    return segment_detalle(entrada, **kw)["mask"]


# ---------------------------------------------------------------- métricas
def metricas_foliculos(d, ancho=4):
    """Una fila por folículo: forma + interfaz folículo-exterior.

    dentro = anillo interno del folículo; fuera = anillo de estroma inmediato
    (dentro del abanico y sin otros folículos). Contraste, CNR y gradiente en el
    borde miden qué tan nítida es la interfaz.
    """
    img, lab, roi = d["pre"].astype(np.float32), d["labels"], d["roi"]
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ancho + 1, 2 * ancho + 1))
    libre = roi & (lab == 0)
    filas = []
    for r in measure.regionprops(lab):
        m = (lab == r.label).astype(np.uint8)
        dentro = m.astype(bool) & ~cv2.erode(m, k).astype(bool)
        fuera = cv2.dilate(m, k).astype(bool) & libre
        borde = segmentation.find_boundaries(m.astype(bool), mode="inner")
        if not dentro.any() or not fuera.any():
            continue
        md, mf = img[dentro].mean(), img[fuera].mean()
        vd, vf = img[dentro].var(), img[fuera].var()
        filas.append(dict(
            foliculo=r.label, area=r.area, perimetro=r.perimeter,
            eje_mayor=r.axis_major_length, eje_menor=r.axis_minor_length,
            excentricidad=r.eccentricity, solidez=r.solidity,
            circularidad=4 * np.pi * r.area / r.perimeter ** 2,
            cy=r.centroid[0], cx=r.centroid[1],
            media_dentro=md, media_fuera=mf,
            contraste_interfaz=mf - md,
            cnr_interfaz=(mf - md) / np.sqrt(vd + vf + 1e-6),
            gradiente_borde=float(grad[borde].mean()),
            desvio_fuera=float(np.sqrt(vf)),
        ))
    return filas


def resumen_imagen(d):
    """Características por imagen a partir de la segmentación (para C)."""
    fol = metricas_foliculos(d)
    area_roi = float(d["roi"].sum())
    out = dict(n_foliculos=len(fol), frac_area_foliculos=float(d["mask"].sum()) / area_roi)
    if fol:
        a = np.array([f["area"] for f in fol], float)
        c = np.array([[f["cy"], f["cx"]] for f in fol])
        out.update(
            area_media=float(a.mean() / area_roi),
            area_cv=float(a.std() / a.mean()),
            circularidad_media=float(np.mean([f["circularidad"] for f in fol])),
            contraste_interfaz_medio=float(np.mean([f["contraste_interfaz"] for f in fol])),
            cnr_interfaz_medio=float(np.mean([f["cnr_interfaz"] for f in fol])),
            gradiente_borde_medio=float(np.mean([f["gradiente_borde"] for f in fol])),
        )
        if len(fol) > 1:  # apiñamiento: distancia al vecino más cercano, normalizada
            dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
            np.fill_diagonal(dd, np.inf)
            out["dist_vecino_media"] = float(dd.min(1).mean() / np.sqrt(area_roi))
    return out


def superponer(d, color=(255, 0, 0), ver_rechazos=False):
    """RGB con el contorno de los folículos (rojo) y del abanico (celeste) sobre la
    imagen preprocesada. ver_rechazos=True dibuja en amarillo las regiones oscuras que
    se descartaron (sirve para ajustar los parámetros)."""
    rgb = cv2.cvtColor(d["pre"], cv2.COLOR_GRAY2RGB)
    if ver_rechazos and d.get("rechazos") is not None and d["rechazos"].any():
        cr, _ = cv2.findContours(d["rechazos"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cr, -1, (255, 220, 0), 1)
    cnts, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, color, 1)
    roi_c, _ = cv2.findContours(d["roi"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, roi_c, -1, (0, 160, 255), 1)
    return rgb


# ============================================================================
# COMANDO comparar: elige método/filtro/contraste (conteo, Dice, vista)
# ============================================================================
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


# ---------------------------------------------------------------- modo Dice
def deteccion_foliculos(pred_lab, gt_lab, iou_min=0.5):
    """(n_gt, n_pred, gt_encontrados, pred_acertados). Un folículo manual cuenta como encontrado
    si algún folículo segmentado lo cubre con IoU >= iou_min; y al revés para la precisión."""
    ng, npd = int(gt_lab.max()), int(pred_lab.max())
    n_gt = len(np.unique(gt_lab[gt_lab > 0]))
    if n_gt == 0 or npd == 0:
        return n_gt, npd, 0, 0
    cruce = np.bincount(gt_lab.ravel().astype(np.int64) * (npd + 1) + pred_lab.ravel(),
                        minlength=(ng + 1) * (npd + 1)).reshape(ng + 1, npd + 1)
    inter = cruce[1:, 1:].astype(float)
    a_gt = cruce[1:, :].sum(1, keepdims=True)    # área de cada folículo manual
    a_pr = cruce[:, 1:].sum(0, keepdims=True)    # área de cada folículo segmentado
    iou = inter / np.maximum(a_gt + a_pr - inter, 1)
    ok = iou >= iou_min
    return n_gt, npd, int(ok.any(1).sum()), int(ok.any(0).sum())

def metricas_mascara(pred, gt):
    """Dice, Jaccard, sensibilidad y precisión píxel a píxel entre dos máscaras booleanas.
    Si las dos están vacías (imagen sin folículos y nada detectado) cuenta como acierto total."""
    inter = float((pred & gt).sum())
    sp, sg = float(pred.sum()), float(gt.sum())
    if sp == 0 and sg == 0:
        return dict(dice=1.0, jaccard=1.0, sens=1.0, prec=1.0)
    return dict(dice=2 * inter / (sp + sg),
                jaccard=inter / (sp + sg - inter),
                sens=inter / sg if sg else float("nan"),
                prec=inter / sp if sp else float("nan"))


def recortar_a_zona(lab, zona, frac_min=0.5):
    """Deja solo los folículos con al menos frac_min de su área dentro de la zona (máscara
    booleana), recortados a la zona, y los renumera 1..N."""
    out = np.zeros(lab.shape, np.int32)
    k = 0
    for i in np.unique(lab[lab > 0]):
        m = lab == i
        if (m & zona).sum() >= frac_min * m.sum():
            k += 1
            out[m & zona] = k
    return out


def superponer_gt(d, gt, zona=None):
    """Como superponer() (rojo = automática) más el contorno manual en verde y la zona en amarillo."""
    rgb = superponer(d)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 1)
    if zona is not None:
        x, y, l = zona
        cv2.rectangle(rgb, (x, y), (x + l, y + l), (255, 255, 0), 1)
    return rgb


def modo_dice(args, kw):
    with open(ROOT / args.mascaras, encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f)
                 if args.conjunto == "todos" or r["conjunto"] == args.conjunto]
    if len(filas) < 10:
        raise SystemExit(f"Solo {len(filas)} imágenes en '{args.conjunto}': "
                         "anotá más con anotar_mascaras.py (mínimo 10).")
    if args.conjunto != "ajuste":
        print("AVISO: 'reporte' es para medir el resultado final con la combinación y los "
              "parámetros ya elegidos. Si elegís mirando estos números, deja de ser independiente.")
    if args.combos:
        combos = [tuple(c.split("+")) for c in args.combos]
    else:
        combos = [(m, f, c) for m in args.metodos for f in args.filtros for c in args.contrastes]
    nombre = lambda fc: "+".join(fc)
    print(f"{len(filas)} imágenes ({args.conjunto}) x {len(combos)} combinaciones")

    por_clase = defaultdict(list)
    for r in filas:
        por_clase[r["clase"]].append(r["imagen"])
    ids_fig = [i for v in por_clase.values() for i in v[:max(1, args.n_fig // 2)]]

    res = {fc: {} for fc in combos}
    tiempos = defaultdict(list)
    fig = defaultdict(dict)
    REPORTES.mkdir(exist_ok=True)
    for r in filas:
        gt = cv2.imread(str(ROOT / r["mascara"]), cv2.IMREAD_UNCHANGED)
        if gt is None:
            raise SystemExit(f"No se pudo leer la máscara {r['mascara']}")
        zl = int(r.get("zlado") or 0)
        zona, zrect = np.ones(gt.shape, bool), None
        if zl > 0:   # zona de evaluación: solo se mide dentro del cuadrado
            zx, zy = int(r["zx"]), int(r["zy"])
            zona = np.zeros(gt.shape, bool)
            zona[zy:zy + zl, zx:zx + zl] = True
            zrect = (zx, zy, zl)
        gt = recortar_a_zona(gt, zona)
        gtb = gt > 0
        for fc in combos:
            t0 = time.perf_counter()
            d = segment_detalle(ROOT / r["imagen"], filtro=fc[1], contraste=fc[2],
                                metodo=fc[0], tam=int(r["tam"]), **kw)
            tiempos[fc].append(time.perf_counter() - t0)
            if d["mask"].shape != gtb.shape:
                raise SystemExit(f"{r['imagen']}: la máscara manual es {gtb.shape} y la "
                                 f"segmentación {d['mask'].shape}. ¿Cambió --tam o base_aspecto?")
            pred = recortar_a_zona(d["labels"], zona)
            m = metricas_mascara(pred > 0, gtb)
            m["clase"] = r["clase"]
            m.update(zip(("n_gt", "n_pred", "tp_gt", "tp_pred"),
                         deteccion_foliculos(pred, gt, args.iou)))
            res[fc][r["imagen"]] = m
            if r["imagen"] in ids_fig:
                fig[r["imagen"]][fc] = superponer_gt(d, gtb, zrect)

    with open(REPORTES / "seg_dice.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["imagen", "clase", "metodo", "filtro", "contraste", "dice", "jaccard",
                    "n_gt", "n_pred", "tp_gt", "tp_pred"])
        for (mt, f, c), por_img in res.items():
            for img, m in por_img.items():
                w.writerow([img, m["clase"], mt, f, c, round(m["dice"], 4), round(m["jaccard"], 4),
                            m["n_gt"], m["n_pred"], m["tp_gt"], m["tp_pred"]])
    

    imgs = [r["imagen"] for r in filas]
    dice = {fc: np.array([res[fc][i]["dice"] for i in imgs]) for fc in combos}
    sin_fol = sum(int(r["n_foliculos"]) == 0 for r in filas)
    L = [f"Imágenes: {len(imgs)} (conjunto {args.conjunto}; {sin_fol} sin folículos en la "
         f"máscara manual) | métodos: {' '.join(args.metodos)} | parámetros: {kw or 'por defecto'}", "",
         f"{'combinación':32s}{'Dice':>7s}{'±DE':>7s}{'mediana':>8s}{'P':>7s}{'R':>7s}{'F1':>7s}"
         f"{'|Δn|':>8s}{'sesgo n':>9s}{'s/img':>7s}"]
    orden = sorted(combos, key=lambda fc: -dice[fc].mean())
    f1s = {}
    for fc in orden:
        ms = [res[fc][i] for i in imgs]
        tpg, tpp = sum(m["tp_gt"] for m in ms), sum(m["tp_pred"] for m in ms)
        ng, npd = sum(m["n_gt"] for m in ms), sum(m["n_pred"] for m in ms)
        P = tpp / npd if npd else float("nan")
        R = tpg / ng if ng else float("nan")
        F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
        f1s[fc] = F1
        L.append(f"{nombre(fc):32s}{dice[fc].mean():7.3f}{dice[fc].std():7.3f}"
                 f"{np.median(dice[fc]):8.3f}{P:7.3f}{R:7.3f}{F1:7.3f}"
                 f"{np.mean([abs(m['n_pred'] - m['n_gt']) for m in ms]):8.2f}"
                 f"{np.mean([m['n_pred'] - m['n_gt'] for m in ms]):+9.2f}"
                 f"{np.mean(tiempos[fc]):7.2f}")
    L += ["", "Dice: por imagen, unión de folículos vs. máscara manual. P/R/F1: por folículo, sumando",
          f"todas las imágenes; un folículo manual está 'encontrado' si algún segmentado lo cubre con",
          f"IoU >= {args.iou} (y al revés para la precisión). |Δn|: error absoluto medio del conteo;",
          "sesgo n: automático - manual (+ = de más)."]

    # --- selección automática de finalistas
    n_vacias = sum(res[combos[0]][i]["n_gt"] == 0 for i in imgs)
    piso = n_vacias / len(imgs)          # Dice de un predictor que nunca detecta nada
    media = {fc: float(dice[fc].mean()) for fc in combos}
    motivo = {}
    for fc in orden:
        if media[fc] <= piso:
            motivo[fc] = "no supera al predictor vacío"
            continue
        dom = [o for o in orden if o != fc and media[o] >= media[fc] + args.margen
               and f1s[o] >= f1s[fc] + args.margen]
        if dom:
            motivo[fc] = f"dominada por {nombre(dom[0])}"
    finalistas = [fc for fc in orden if fc not in motivo]
    L += ["", f"FINALISTAS (selección automática; {len(finalistas)} de {len(combos)})",
          f"  Regla 1: Dice mayor que el del predictor vacío ({piso:.3f} = {n_vacias} de {len(imgs)} zonas sin folículos).",
          f"  Regla 2: ninguna otra la supera en Dice Y en F1 por folículo, las dos por al menos {args.margen}.",
          "  " + "  ".join(nombre(fc) for fc in finalistas) if finalistas else
          "  Ninguna supera al predictor vacío: revisar los parámetros de la segmentación.",
          f"  Descartadas por regla 1: {sum(m.startswith('no supera') for m in motivo.values())} | "
          f"por regla 2: {sum(m.startswith('dominada') for m in motivo.values())}"]
    if finalistas:
        L.append("  Para el paso siguiente: --combos " + " ".join(nombre(fc) for fc in finalistas))
    with open(REPORTES / "seg_finalistas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["combinacion", "dice", "f1", "finalista", "motivo"])
        for fc in orden:
            w.writerow([nombre(fc), round(media[fc], 4), round(f1s[fc], 4),
                        int(fc not in motivo), motivo.get(fc, "")])

    cv2.imwrite(str(REPORTES / "seg_dice_overlays.png"),
                hoja([[fig[i][fc] for fc in orden[:6]] for i in ids_fig],
                     [nombre(fc) for fc in orden[:6]], lado=200))
    top = orden[0]
    if len(combos) >= 3:
        chi, p_fr = stats.friedmanchisquare(*[dice[fc] for fc in combos])
        L += ["", f"Friedman entre las {len(combos)} combinaciones (Dice por imagen): "
                  f"chi2 = {chi:.1f}, p = {p_fr:.2g}"]
    if len(combos) >= 2:
        L.append(f"Wilcoxon pareado de '{nombre(top)}' contra cada una "
                 f"(p con Bonferroni, {len(combos) - 1} comparaciones):")
        for fc in orden[1:]:
            try:
                p = stats.wilcoxon(dice[top], dice[fc]).pvalue
            except ValueError:   # diferencias todas cero
                p = 1.0
            L.append(f"  {nombre(top)} vs {nombre(fc):32s} p = "
                     f"{min(1.0, p * (len(combos) - 1)):.3g}")
    L += ["", "No elegir solo por Dice: mirar reports/seg_dice_overlays.png (verde = manual, rojo =",
          "automática). Con pocas imágenes las diferencias chicas entre combinaciones son ruido."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_dice.txt").write_text(texto + "\n", encoding="utf-8")

def modo_vista(args, kw):
    """Figura para decidir a ojo: una fila por imagen y una columna por combinación (sin máscara manual).
    Con --ciego las columnas se mezclan y se rotulan A, B, C...; la clave queda en un archivo aparte."""
    if not args.combos:
        raise SystemExit("--vista necesita --combos metodo+filtro+contraste ...")
    combos = [tuple(c.split("+")) for c in args.combos]
    rng = random.Random(args.seed)
    if args.ciego:
        rng.shuffle(combos)
    filas = leer_splits(ROOT / args.splits, not args.todas_del_grupo, args.seed)
    por_clase = defaultdict(list)
    for r in filas:
        por_clase[int(r["label"])].append(r)
    muestra = [r for y in sorted(por_clase)
               for r in rng.sample(por_clase[y], min(max(1, args.n_fig // 2), len(por_clase[y])))]
    grilla = [[superponer(segment_detalle(ROOT / r["path"], metodo=m, filtro=f, contraste=c, **kw),
                          ver_rechazos=args.ver_rechazos) for m, f, c in combos] for r in muestra]
    letras = [chr(65 + i) for i in range(len(combos))]
    REPORTES.mkdir(exist_ok=True)
    cv2.imwrite(str(REPORTES / "seg_vista.png"),
                hoja(grilla, letras if args.ciego else ["+".join(c) for c in combos], lado=220))
    print("Filas de reports/seg_vista.png (de arriba hacia abajo):")
    for i, r in enumerate(muestra, 1):
        print(f"  {i}: {r['path']} (label {r['label']})")
    if args.ciego:
        clave = ["Clave de columnas (no mirar hasta decidir):"] + \
                [f"  {l}: {'+'.join(c)}" for l, c in zip(letras, combos)]
        (REPORTES / "seg_vista_clave.txt").write_text("\n".join(clave) + "\n", encoding="utf-8")
        print("Columnas rotuladas A, B, C...; la clave está en reports/seg_vista_clave.txt")

def cmd_comparar(argv=None):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtros", nargs="+", default=["ninguno", "wiener3", "wiener5", "gauss1.0", "frost7"])
    ap.add_argument("--contrastes", nargs="+", default=["ninguno", "estirar"])
    ap.add_argument("--metodo", default="multiotsu", choices=["multiotsu", "otsu", "chanvese", "triangle", "li", "adaptativo"])
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
    ap.add_argument("--iou", type=float, default=0.5, help="umbral de IoU para la detección por folículo")
    ap.add_argument("--mascaras", default=None, help="mascaras.csv: activa el modo Dice")
    ap.add_argument("--conjunto", choices=["ajuste", "reporte", "todos"], default="ajuste")
    ap.add_argument("--metodos", nargs="+", default=["otsu", "multiotsu", "chanvese", "adaptativo"])
    ap.add_argument("--combos", nargs="+", default=None,
                    help="combinaciones metodo+filtro+contraste (reemplaza al producto de --metodos/--filtros/--contrastes)")
    ap.add_argument("--margen", type=float, default=0.02,
                    help="margen de dominancia para elegir finalistas (en Dice y en F1)")
    ap.add_argument("--vista", action="store_true",
                    help="figura para decidir a ojo con las --combos, sin máscaras manuales")
    ap.add_argument("--ciego", action="store_true",
                    help="en --vista, oculta los nombres (A, B, C...) y guarda la clave en un archivo")
    args = ap.parse_args(argv)
    kw = {k: v for k, v in dict(area_min=args.area_min, contraste_min=args.contraste_min,
                                circ_min=args.circ_min, sigma_umbral=args.sigma_umbral,
                                reumbral=False if args.sin_reumbral else None).items()
          if v is not None}
    if args.vista:
        modo_vista(args, kw)
        return
    if args.mascaras:
       modo_dice(args, kw)
       return

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


# ============================================================================
# COMANDO ver-casos: peores conteos
# ============================================================================
def cmd_ver_casos(argv=None):
    """Muestra los casos donde el conteo de folículos más se equivoca (correr después de
segmentacion.py comparar).

  fila 1: las PCOS con MENOS folículos  -> ¿se pierden folículos o la imagen no los muestra?
  fila 2: las sanas con MÁS folículos   -> ¿qué está tomando como folículo?

Salida: reports/seg_peores.png (rojo = folículo, amarillo = descartado, celeste = abanico)
y la lista de archivos en la terminal, para abrirlos y mirarlos en tamaño original.

Uso:
    python segmentacion.py ver-casos                         # combinación mediana+estirar
    python segmentacion.py ver-casos --filtro lee --contraste clahe --n 8
    python segmentacion.py ver-casos --contraste-min 20      # mismos parámetros que usaste al comparar
"""
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
    args = ap.parse_args(argv)
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


# ============================================================================
# COMANDO anotar-mascaras: folículos dibujados a mano (referencia para Dice)
# ============================================================================
CARPETA = ROOT / "mascaras"
INDICE = ROOT / "mascaras.csv"
CAMPOS = ["imagen", "clase", "mascara", "n_foliculos", "conjunto", "tam", "zx", "zy", "zlado"]


class Mascara:
    """Estado de los polígonos de una imagen (separado de la ventana para poder probarlo)."""

    def __init__(self, g):
        self.g = g
        self.poligonos = []   # cada uno: array (n, 2) int32 de (x, y)
        self.actual = []      # vértices del folículo en curso
        self.aviso = ""
        self.zona = None      # (x, y, lado) del cuadrado de evaluación
        self.fase = "zona"    # "zona" (ubicar el cuadrado) o "dibujo" (marcar folículos)

    def fijar_zona(self, cx, cy, lado):
        t = self.g.shape[0]
        x = min(max(int(cx) - lado // 2, 0), t - lado)
        y = min(max(int(cy) - lado // 2, 0), t - lado)
        self.zona = (x, y, lado)
        self.aviso = ""

    def vertice(self, x, y):
        t = self.g.shape[0]
        self.actual.append((min(max(int(x), 0), t - 1), min(max(int(y), 0), t - 1)))
        self.aviso = ""

    def cerrar(self):
        if len(self.actual) < 3:
            self.aviso = "un folículo necesita al menos 3 vértices"
            return False
        self.poligonos.append(np.array(self.actual, np.int32))
        self.actual = []
        self.aviso = ""
        return True

    def deshacer(self):
        if self.actual:
            self.actual.pop()
        elif self.poligonos:
            self.poligonos.pop()
        self.aviso = ""

    def etiquetas(self):
        lab = np.zeros(self.g.shape, np.uint8)
        for i, p in enumerate(self.poligonos, 1):
            cv2.fillPoly(lab, [p], i)
        return lab


def asignar_conjunto(muestra):
    """Mitad 'ajuste' y mitad 'reporte' dentro de cada clase, de forma determinista."""
    por_clase = defaultdict(list)
    for p, cl in sorted(muestra):
        por_clase[cl].append(p)
    return {p: ("ajuste" if i % 2 == 0 else "reporte")
            for rutas in por_clase.values() for i, p in enumerate(rutas)}


def ya_anotadas():
    if not INDICE.exists():
        return set()
    with open(INDICE, encoding="utf-8") as fh:
        return {ROOT / r["imagen"] for r in csv.DictReader(fh)}


def migrar_indice():
    """Si mascaras.csv es de una versión sin zona, agrega zx, zy, zlado = 0 (0 = imagen completa)."""
    if not INDICE.exists():
        return
    with open(INDICE, encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        if "zlado" in (lector.fieldnames or []):
            return
        filas = list(lector)
    for r in filas:
        r.update(zx=0, zy=0, zlado=0)
    with open(INDICE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def procesar_tecla(est, tecla, estirar):
    """Aplica una tecla al estado. Devuelve (accion, estirar); accion: None, 'guardar', 'saltear' o 'salir'."""
    if tecla == ord("c"):
        return None, not estirar
    if tecla in (ord("q"), 27):
        return "salir", estirar
    if tecla == ord("s"):
        return "saltear", estirar
    if est.fase == "zona":
        if tecla in (13, 32):
            if est.zona is None:
                est.aviso = "hacé clic para ubicar la zona de evaluación"
            else:
                est.fase, est.aviso = "dibujo", ""
        return None, estirar
    if tecla == ord("u"):
        est.deshacer()
    elif tecla == ord("r"):
        est.fase, est.aviso = "zona", ""
    elif tecla in (13, 32):
        if est.actual:
            est.aviso = "hay un folículo sin cerrar: clic derecho para cerrarlo (u para borrarlo)"
        elif est.poligonos:
            return "guardar", estirar
        else:
            est.aviso = "marcá un folículo, o z si la zona no tiene ninguno"
    elif tecla == ord("z"):
        if est.actual or est.poligonos:
            est.aviso = "ya marcaste folículos: z es solo para zonas sin ninguno"
        else:
            return "guardar", estirar
    return None, estirar


def guardar(p, clase, est, conjunto, tam):
    destino = CARPETA / f"{clase}__{p.stem}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), est.etiquetas())
    nuevo = not INDICE.exists()
    with open(INDICE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(CAMPOS)
        zx, zy, zl = est.zona if est.zona else (0, 0, 0)
        w.writerow([p.relative_to(ROOT).as_posix(), clase,
                    destino.relative_to(ROOT).as_posix(), len(est.poligonos), conjunto, tam,
                    zx, zy, zl])


def dibujar(est, zoom, texto, estirar):
    g = est.g
    if estirar and (g > 0).any():
        lo, hi = np.percentile(g[g > 0], (1, 99))
        g = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    im = cv2.cvtColor(cv2.resize(g, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC),
                      cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(est.poligonos, 1):
        cv2.polylines(im, [(p * zoom).astype(np.int32)], True, (0, 0, 255), 1)
        cx, cy = (p.mean(axis=0) * zoom).astype(int)
        cv2.putText(im, str(i), (int(cx), int(cy)), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if est.zona:
        zx, zy, zl = est.zona
        cv2.rectangle(im, (zx * zoom, zy * zoom), ((zx + zl) * zoom, (zy + zl) * zoom), (0, 255, 255), 1)
    if est.actual:
        pts = (np.array(est.actual, np.int32) * zoom).astype(np.int32)
        cv2.polylines(im, [pts], False, (0, 255, 0), 1)
        for q in pts:
            cv2.circle(im, (int(q[0]), int(q[1])), 2, (0, 255, 0), -1)
    cv2.putText(im, texto, (6, 20), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if est.aviso:
        cv2.putText(im, est.aviso, (6, 42), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def cmd_anotar_mascaras(argv=None):
    """Dibujá a mano los folículos de una muestra de imágenes (referencia para calcular Dice).

Cada folículo es un polígono. Se dibuja sobre la imagen base de segmentacion.py (sin filtros),
así que la máscara queda en el mismo marco de 256x256 que la segmentación y se puede comparar
píxel a píxel. No cambies `base_aspecto` ni --tam después de empezar a anotar.

Zona de evaluación: en cada imagen primero ubicás un cuadrado de --zona px (un clic, ENTER para
confirmar) y marcás TODOS los folículos que toquen esa zona. Las métricas de segmentacion.py comparar
se calculan solo dentro del cuadrado, así que lo que quede afuera no cuenta. Mismo tamaño en todas
las imágenes, y elegí siempre una parte del ovario con folículos nítidos, sin mirar ningún resultado.

Controles (ventana de OpenCV, imagen ampliada --zoom veces):
  fase zona:        clic = ubicar el cuadrado (se puede repetir), ENTER/espacio = confirmar
  r                 vuelve a la fase zona para mover el cuadrado
  clic izquierdo    agrega un vértice al folículo en curso
  clic derecho      cierra el folículo (hacen falta al menos 3 vértices)
  u                 deshace el último vértice; si no hay, el último folículo
  c                 alterna estirar el contraste SOLO para ver (no cambia la máscara)
  ENTER / espacio   guarda la imagen y pasa a la siguiente (hace falta al menos 1 folículo)
  z                 guarda la imagen como "sin folículos"
  s                 saltea la imagen (no se guarda nada)
  q / ESC           sale (lo ya guardado queda)

Criterio (el mismo en todas las imágenes, y fijado antes de empezar):
  marcá estructuras oscuras de borde nítido, redondeadas u ovales, dentro del ovario; el contorno
  va sobre el borde del folículo. No marques sombras acústicas ni zonas oscuras del estroma sin
  borde claro. Ante la duda, siempre la misma decisión (por ejemplo, no marcar).

Muestra: imágenes de split == "cv" (una por grupo), balanceadas por clase, con la misma semilla
que preprocesamiento.py anotar-rois. La mitad de cada clase queda como "ajuste" (para calibrar parámetros de la
segmentación) y la otra mitad como "reporte" (solo para medir el resultado final).
No cambies --n-por-clase ni --seed a mitad de camino: cambiaría la muestra.

Salidas:
  mascaras/<clase>__<imagen>.png   etiquetas (0 = fondo, 1..N = folículo i)
  mascaras.csv                     índice: imagen, clase, máscara, n, conjunto, tam, zx, zy, zlado
                                   (zlado = 0 significa que la zona es la imagen completa)

Uso:
    python segmentacion.py anotar-mascaras --n-por-clase 15
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-por-clase", type=int, default=15)
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--zona", type=int, default=128, help="lado del cuadrado de evaluación (px de 256)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    muestra, origen = armar_muestra(args.n_por_clase, rng)
    conjunto = asignar_conjunto(muestra)
    rng.shuffle(muestra)   # mezcla las clases para no anotar todas las infected seguidas
    migrar_indice()
    hechas = ya_anotadas()
    pendientes = [(p, cl) for p, cl in muestra if p not in hechas]
    print(f"Muestra: {origen}. Total {len(muestra)} | ya anotadas: "
          f"{len(muestra) - len(pendientes)} | pendientes {len(pendientes)}")

    vent = "anotar mascaras"
    cv2.namedWindow(vent, cv2.WINDOW_AUTOSIZE)
    estirar = False
    for i, (p, cl) in enumerate(pendientes, 1):
        est = Mascara(base_aspecto(p, args.tam))

        def raton(ev, mx, my, fl, par, e=est):
            x, y = mx // args.zoom, my // args.zoom
            if e.fase == "zona":
                if ev == cv2.EVENT_LBUTTONDOWN:
                    e.fijar_zona(x, y, args.zona)
            elif ev == cv2.EVENT_LBUTTONDOWN:
                e.vertice(x, y)
            elif ev == cv2.EVENT_RBUTTONDOWN:
                e.cerrar()
        cv2.setMouseCallback(vent, raton)
        while True:
            if est.fase == "zona":
                txt = (f"{i}/{len(pendientes)} [{conjunto[p]}] ZONA: clic = ubicar el cuadrado de "
                       f"{args.zona}px | ENTER = confirmar | s=saltear  c=contraste  q=salir")
            else:
                txt = (f"{i}/{len(pendientes)} [{conjunto[p]}] folículos: {len(est.poligonos)} "
                       f"| clic=vértice  clic der.=cerrar  u=deshacer  ENTER=guardar  z=sin folículos  "
                       f"r=mover zona  s=saltear  c=contraste  q=salir")
            cv2.imshow(vent, dibujar(est, args.zoom, txt, estirar))
            accion, estirar = procesar_tecla(est, cv2.waitKey(30) & 0xFF, estirar)
            if accion == "guardar":
                guardar(p, cl, est, conjunto[p], args.tam)
                break
            if accion == "saltear":
                break
            if accion == "salir":
                cv2.destroyAllWindows()
                print(f"Listo. Índice en {INDICE}")
                return
    cv2.destroyAllWindows()
    hechas = ya_anotadas()
    cuenta = Counter((conjunto[p], cl) for p, cl in muestra if p in hechas)
    print("Terminaste la muestra. Anotadas por conjunto y clase:", dict(cuenta))
    print(f"Índice en {INDICE}")


COMANDOS = {
    "comparar": cmd_comparar,
    "ver-casos": cmd_ver_casos,
    "anotar-mascaras": cmd_anotar_mascaras,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print("Uso: python procesamiento/segmentacion.py <comando> [opciones]\n\nComandos:")
        for k, f in COMANDOS.items():
            print(f"  {k:18s}{(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        print("\nPara ver las opciones de uno: python procesamiento/segmentacion.py <comando> -h")
        raise SystemExit(1)
    COMANDOS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
