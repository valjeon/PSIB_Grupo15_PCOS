"""Segmentación de folículos (módulo de B). Usa el preprocesamiento de preprocess.py.

Pipeline de `segment(entrada)`:
  1. imagen base: sin marcas, recortada y a tamaño común, CONSERVANDO el aspecto
     (relleno negro). Estirar a cuadrado deforma los folículos y falsea
     circularidad, excentricidad y ejes, que después usa C.
  2. máscara del abanico (zona con señal). Sin ella, Otsu toma el fondo negro y
     la sombra del transductor como si fueran folículos.
  3. filtro de speckle + contraste (los mismos nombres que en preprocess.py);
     el estiramiento se calcula solo dentro del abanico.
  4. umbral: los folículos son la clase más oscura (Otsu de 3 clases por defecto,
     Otsu clásico o Chan-Vese morfológico para comparar).
  5. morfología: apertura, cierre, relleno de huecos.
  6. descarta lo que toca el borde del abanico.
  7. watershed sobre la transformada de distancia (separa folículos pegados).
  8. filtra regiones por área, circularidad y solidez.

Uso:
    from segmentacion import segment, segment_detalle, resumen_imagen
    mask = segment("data/clean/infected/x.jpg", filtro="mediana", contraste="estirar")
    d = segment_detalle("data/clean/infected/x.jpg")   # base, pre, roi, labels, mask
    fila = resumen_imagen(d)                             # n_foliculos, interfaz, etc.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, segmentation

from preprocess import (a_gris, contrastar, filtrar, leer, quitar_marcas,
                        recortar_util)


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