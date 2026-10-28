"""Dibujá a mano los folículos de una muestra de imágenes (referencia para calcular Dice).

Cada folículo es un polígono. Se dibuja sobre la imagen base de segmentacion.py (sin filtros),
así que la máscara queda en el mismo marco de 256x256 que la segmentación y se puede comparar
píxel a píxel. No cambies `base_aspecto` ni --tam después de empezar a anotar.

Controles (ventana de OpenCV, imagen ampliada --zoom veces):
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
que anotar_rois.py. La mitad de cada clase queda como "ajuste" (para calibrar parámetros de la
segmentación) y la otra mitad como "reporte" (solo para medir el resultado final).
No cambies --n-por-clase ni --seed a mitad de camino: cambiaría la muestra.

Salidas:
  mascaras/<clase>__<imagen>.png   etiquetas (0 = fondo, 1..N = folículo i)
  mascaras.csv                                índice: imagen, clase, máscara, n, conjunto

Uso:
    python anotar_mascaras.py --n-por-clase 15
"""
from collections import Counter, defaultdict
import argparse
import csv
import random

import cv2
import numpy as np

from anotar_rois import ROOT, armar_muestra
from segmentacion import base_aspecto

CARPETA = ROOT / "mascaras"
INDICE = ROOT / "mascaras.csv"
CAMPOS = ["imagen", "clase", "mascara", "n_foliculos", "conjunto", "tam"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Mascara:
    """Estado de los polígonos de una imagen (separado de la ventana para poder probarlo)."""

    def __init__(self, g):
        self.g = g
        self.poligonos = []   # cada uno: array (n, 2) int32 de (x, y)
        self.actual = []      # vértices del folículo en curso
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


def guardar(p, clase, est, conjunto, tam):
    destino = CARPETA / f"{clase}__{p.stem}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), est.etiquetas())
    nuevo = not INDICE.exists()
    with open(INDICE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(CAMPOS)
        w.writerow([p.relative_to(ROOT).as_posix(), clase,
                    destino.relative_to(ROOT).as_posix(), len(est.poligonos), conjunto, tam])


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
    if est.actual:
        pts = (np.array(est.actual, np.int32) * zoom).astype(np.int32)
        cv2.polylines(im, [pts], False, (0, 255, 0), 1)
        for q in pts:
            cv2.circle(im, (int(q[0]), int(q[1])), 2, (0, 255, 0), -1)
    cv2.putText(im, texto, (6, 20), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if est.aviso:
        cv2.putText(im, est.aviso, (6, 42), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-por-clase", type=int, default=15)
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    muestra, origen = armar_muestra(args.n_por_clase, rng)
    conjunto = asignar_conjunto(muestra)
    rng.shuffle(muestra)   # mezcla las clases para no anotar todas las infected seguidas
    hechas = ya_anotadas()
    pendientes = [(p, cl) for p, cl in muestra if p not in hechas]
    print(f"Muestra: {origen}. Total {len(muestra)} | ya anotadas:"
          f"{len(muestra) - len(pendientes)} | pendientes {len(pendientes)}")

    vent = "anotar mascaras"
    cv2.namedWindow(vent, cv2.WINDOW_AUTOSIZE)
    estirar = False
    for i, (p, cl) in enumerate(pendientes, 1):
        est = Mascara(base_aspecto(p, args.tam))

        def raton(ev, mx, my, fl, par, e=est):
            if ev == cv2.EVENT_LBUTTONDOWN:
                e.vertice(mx // args.zoom, my // args.zoom)
            elif ev == cv2.EVENT_RBUTTONDOWN:
                e.cerrar()
        cv2.setMouseCallback(vent, raton)
        while True:
            txt = (f"{i}/{len(pendientes)} [{conjunto[p]}] folículos: {len(est.poligonos)} "
                   f"| clic=vértice  clic der.=cerrar  u=deshacer  ENTER=guardar  z=sin folículos  "
                   f"s=saltear  c=contraste  q=salir")
            cv2.imshow(vent, dibujar(est, args.zoom, txt, estirar))
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord("u"):
                est.deshacer()
            elif tecla == ord("c"):
                estirar = not estirar
            elif tecla in (13, 32):
                if est.actual:
                    est.aviso = "hay un folículo sin cerrar: clic derecho para cerrarlo (u para borrarlo)"
                elif est.poligonos:
                    guardar(p, cl, est, conjunto[p], args.tam)
                    break
                else:
                    est.aviso = "marcá un folículo, o z si la imagen no tiene ninguno"
            elif tecla == ord("z"):
                if est.actual or est.poligonos:
                    est.aviso = "ya marcaste folículos: z es solo para imágenes sin ninguno"
                else:
                    guardar(p, cl, est, conjunto[p], args.tam)
                    break
            elif tecla == ord("s"):
                break
            elif tecla in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Listo. Índice en {INDICE}")
                return
    cv2.destroyAllWindows()
    hechas = ya_anotadas()
    cuenta = Counter((conjunto[p], cl) for p, cl in muestra if p in hechas)
    print("Terminaste la muestra. Anotadas por conjunto y clase:", dict(cuenta))
    print(f"Índice en {INDICE}")


if __name__ == "__main__":
    main()