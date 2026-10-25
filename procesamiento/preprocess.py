"""Preprocesamiento de ecografías ováricas (módulo de B).

Pipeline de `preprocess(entrada)`:
  1. gris + quitar marcas (flechas/anotaciones de color, texto y cruces blancos) por inpainting
  2. recortar el área útil (saca bordes negros)
  3. llevar a un tamaño común (INTER_AREA al achicar)  -> reduce pistas de resolución
  4. filtro de speckle (mediana, gaussiano, bilateral, nlm, lee, kuan, frost)
  5. contraste (global, clahe, estirar)

El tamaño común se aplica ANTES de filtrar: el speckle de una imagen de 984x848 y el de una
de 300x300 no son comparables hasta que tienen la misma resolución.

Uso:
    from preprocess import preprocess
    img = preprocess("data/clean/infected/x.jpg", filtro="mediana", contraste="clahe")
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------- lectura
def leer(ruta):
    """RGB uint8 (H, W, 3) sin importar el modo del archivo (L, P, RGBA...)."""
    with Image.open(ruta) as im:
        return np.asarray(im.convert("RGB"))


def a_gris(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------- marcas
def _componentes_chicas(binaria, area_max):
    """Máscara con solo las componentes conexas de área <= area_max."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    ids = np.where(stats[:, cv2.CC_STAT_AREA] <= area_max)[0]
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def mascara_marcas(rgb, sat_min=0.55, val_min=0.45, max_frac_color=0.03,
                   quitar_texto=True, max_frac_texto=0.002, max_frac_total=0.15, dilatar=3):
    """Máscara (uint8 0/255) de anotaciones: color muy saturado (flechas, calibres) y
    componentes blancas chicas (texto, cruces). Las zonas grandes de color (Doppler) y los
    tintes suaves de toda la imagen NO se enmascaran. Si la máscara cubriría más de
    `max_frac_total` de la imagen se descarta (es un tinte, no una anotación)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    n = rgb.shape[0] * rgb.shape[1]
    color = (s >= sat_min) & (v >= val_min)
    m = _componentes_chicas(color, int(max_frac_color * n))
    if quitar_texto:
        blanco = a_gris(rgb) >= 245
        m |= _componentes_chicas(blanco, max(30, int(max_frac_texto * n)))
    if m.mean() > max_frac_total:
        return np.zeros(m.shape, np.uint8)
    m = m.astype(np.uint8) * 255
    if dilatar > 0 and m.any():
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilatar)
    return m


def quitar_marcas(rgb, **kw):
    """Devuelve (gris sin marcas, máscara usada)."""
    gris = a_gris(rgb)
    m = mascara_marcas(rgb, **kw)
    if m.any():
        gris = cv2.inpaint(gris, m, 3, cv2.INPAINT_TELEA)
    return gris, m


# ---------------------------------------------------------------- geometría
def recortar_util(gris, umbral=12, margen=2, min_frac=0.25):
    """Recorta al rectángulo que contiene la componente más grande no negra."""
    m = (cv2.GaussianBlur(gris, (5, 5), 0) > umbral).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return gris
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    if w * h < min_frac * gris.size:  # recorte absurdo: no tocar
        return gris
    return gris[max(y - margen, 0):y + h + margen, max(x - margen, 0):x + w + margen]


def a_tamano(gris, tam=256):
    """tam x tam. Estira sin conservar el aspecto: rellenar con negro delataría la fuente."""
    achica = gris.shape[0] * gris.shape[1] > tam * tam
    interp = cv2.INTER_AREA if achica else cv2.INTER_CUBIC
    return cv2.resize(gris, (tam, tam), interpolation=interp)


# ---------------------------------------------------------------- speckle
def _locales(img, w):
    f = img.astype(np.float32)
    m = cv2.boxFilter(f, -1, (w, w), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(f * f, -1, (w, w), borderType=cv2.BORDER_REFLECT)
    v = np.maximum(m2 - m * m, 0)
    return f, m, v


def _cu2(m, v, cu2):
    """Coeficiente de variación del ruido al cuadrado: mediana sobre zonas no oscuras."""
    ci2 = v / (m * m + 1e-6)
    if cu2 is None:
        zona = m > 10
        cu2 = float(np.median(ci2[zona])) if zona.any() else float(np.median(ci2))
    return ci2, cu2


def lee(img, w=7, cu2=None):
    f, m, v = _locales(img, w)
    ci2, cu2 = _cu2(m, v, cu2)
    k = np.clip(1 - cu2 / np.maximum(ci2, 1e-6), 0, 1)
    return np.clip(m + k * (f - m), 0, 255).astype(np.uint8)


def kuan(img, w=7, cu2=None):
    f, m, v = _locales(img, w)
    ci2, cu2 = _cu2(m, v, cu2)
    k = np.clip((1 - cu2 / np.maximum(ci2, 1e-6)) / (1 + cu2), 0, 1)
    return np.clip(m + k * (f - m), 0, 255).astype(np.uint8)


def frost(img, w=7, K=2.0):
    f, m, v = _locales(img, w)
    ci2 = v / (m * m + 1e-6)
    r = w // 2
    pad = cv2.copyMakeBorder(f, r, r, r, r, cv2.BORDER_REFLECT)
    H, W = f.shape
    num = np.zeros_like(f)
    den = np.zeros_like(f)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            wgt = np.exp(-K * ci2 * np.hypot(dy, dx))
            num += wgt * pad[r + dy:r + dy + H, r + dx:r + dx + W]
            den += wgt
    return np.clip(num / np.maximum(den, 1e-9), 0, 255).astype(np.uint8)


FILTROS = {
    "ninguno": lambda im: im,
    "mediana": lambda im: cv2.medianBlur(im, 5),
    "gaussiano": lambda im: cv2.GaussianBlur(im, (0, 0), 1.5),
    "bilateral": lambda im: cv2.bilateralFilter(im, 7, 40, 5),
    "nlm": lambda im: cv2.fastNlMeansDenoising(im, None, 10, 7, 21),
    "lee": lee,
    "kuan": kuan,
    "frost": frost,
}


def filtrar(img, metodo="mediana"):
    return FILTROS[metodo](img)


# ---------------------------------------------------------------- contraste
def _estirar(img, p=(1, 99)):
    lo, hi = np.percentile(img, p)
    if hi <= lo:
        return img
    return (np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


CONTRASTES = {
    "ninguno": lambda im: im,
    "global": cv2.equalizeHist,
    "clahe": lambda im: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im),
    "estirar": _estirar,
}


def contrastar(img, metodo="clahe"):
    return CONTRASTES[metodo](img)


# ---------------------------------------------------------------- pipeline
def base(entrada, tam=256, marcas=True, recortar=True, texto=True):
    """Gris sin marcas, recortado y a tamaño común (sin filtro ni contraste).
    `entrada`: ruta o array RGB."""
    rgb = leer(entrada) if isinstance(entrada, (str, Path)) else entrada
    if marcas:
        gris, _ = quitar_marcas(rgb, quitar_texto=texto)
    else:
        gris = a_gris(rgb)
    if recortar:
        gris = recortar_util(gris)
    return a_tamano(gris, tam)


def preprocess(entrada, filtro="mediana", contraste="clahe", tam=256,
               marcas=True, recortar=True, texto=True):
    """uint8 (tam, tam). Es la función que usa el resto del proyecto."""
    g = base(entrada, tam, marcas, recortar, texto)
    return contrastar(filtrar(g, filtro), contraste)


# ---------------------------------------------------------------- métricas
def ssi(orig, filt):
    """Índice de supresión de speckle (σ/μ filtrada ÷ σ/μ original). < 1 = suaviza."""
    o, f = orig.astype(np.float32), filt.astype(np.float32)
    return float((f.std() / (f.mean() + 1e-6)) / (o.std() / (o.mean() + 1e-6) + 1e-9))


def beta(orig, filt):
    """Preservación de bordes: correlación entre laplacianos (1 = bordes intactos)."""
    lo = cv2.Laplacian(orig.astype(np.float32), cv2.CV_32F)
    lf = cv2.Laplacian(filt.astype(np.float32), cv2.CV_32F)
    lo, lf = lo - lo.mean(), lf - lf.mean()
    return float((lo * lf).sum() / (np.sqrt((lo ** 2).sum() * (lf ** 2).sum()) + 1e-9))


def entropia(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())
