"""Marcá a mano bloques de tejido homogéneo en una muestra de imágenes (para el ENL).

Controles (ventana de OpenCV, imagen ampliada 3x):
  clic izquierdo    agrega un bloque centrado en ese punto
  u                 deshace el último bloque
  ENTER / espacio   guarda esta imagen y pasa a la siguiente
  s                 saltea la imagen (no se guarda nada)
  q / ESC           guarda lo hecho y sale

Criterio para elegir un bloque (el mismo en todas las imágenes, y SIN mirar ningún filtro):
  tejido liso (estroma): sin bordes, sin folículos, sin sombras acústicas, sin marcas
  ni cruces, y lejos del borde del ovario. Marcá 3 bloques por imagen (mínimo 1, máximo --k).

Se puede retomar: las imágenes que ya están en rois.csv se saltean.
Los bloques se guardan en coordenadas de la imagen base de --tam x --tam (la misma que usa
elegir_filtro_enl.py), así que no cambies --tam ni --lado entre sesiones.

Uso:
    python anotar_rois.py --n-por-clase 20          # 40 imágenes en total
    python elegir_filtro_enl.py --rois rois.csv
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import random

import cv2

from preprocess import base

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ZOOM = 3
ROIS = ROOT / "rois.csv"
CAMPOS = ["path", "clase", "x", "y", "lado", "tam"]


def leer_splits(archivo="splits.csv"):
    f = ROOT / archivo
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def armar_muestra(n_por_clase, rng):
    """Devuelve [(ruta, clase)], balanceada por clase, independiente de near-duplicados.
    Con splits.csv usa solo split == "cv" (el hold-out no se toca) y una imagen por grupo."""
    filas = leer_splits()
    por_clase = defaultdict(list)
    if filas is not None:
        por_grupo = {}
        for r in filas:
            if r["split"] != "cv":
                continue
            por_grupo.setdefault((r["clase"], r["grupo"]), []).append(ROOT / r["path"])
        for (cl, _), rutas in por_grupo.items():
            por_clase[cl].append(rng.choice(rutas))
        origen = "splits.csv (solo split=cv, una imagen por grupo)"
    else:
        for p in sorted(CLEAN.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXT:
                por_clase[p.parent.name].append(p)
        origen = "data/clean al azar (no hay splits.csv)"
    muestra = []
    for cl, rutas in sorted(por_clase.items()):
        muestra += [(p, cl) for p in rng.sample(rutas, min(n_por_clase, len(rutas)))]
    return muestra, origen


class Bloques:
    """Estado de los bloques de una imagen (separado de la ventana para poder probarlo)."""

    def __init__(self, g, lado, k):
        self.g, self.lado, self.k = g, lado, k
        self.lista = []      # [(y, x)]
        self.aviso = ""

    def agregar(self, cx, cy):
        """cx, cy: clic en coordenadas de la imagen base. Devuelve True si se agregó."""
        t = self.g.shape[0]
        x = min(max(cx - self.lado // 2, 0), t - self.lado)
        y = min(max(cy - self.lado // 2, 0), t - self.lado)
        if len(self.lista) >= self.k:
            self.aviso = f"máximo {self.k} bloques (u para deshacer)"
            return False
        if (self.g[y:y + self.lado, x:x + self.lado] <= 10).any():
            self.aviso = "ese bloque toca fondo negro: no se agregó"
            return False
        self.lista.append((y, x))
        self.aviso = ""
        return True

    def deshacer(self):
        if self.lista:
            self.lista.pop()
        self.aviso = ""


def ya_anotadas():
    if not ROIS.exists():
        return set(), None
    with open(ROIS, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    conf = {(int(r["lado"]), int(r["tam"])) for r in filas}
    return {ROOT / r["path"] for r in filas}, (conf.pop() if len(conf) == 1 else "mezcla")


def guardar(p, clase, bloques, lado, tam):
    nuevo = not ROIS.exists()
    with open(ROIS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(CAMPOS)
        for y, x in bloques:
            w.writerow([p.relative_to(ROOT).as_posix(), clase, x, y, lado, tam])


def dibujar(estado, texto):
    im = cv2.resize(estado.g, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for y, x in estado.lista:
        cv2.rectangle(im, (x * ZOOM, y * ZOOM),
                      ((x + estado.lado) * ZOOM, (y + estado.lado) * ZOOM), (0, 255, 0), 2)
    cv2.putText(im, texto, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if estado.aviso:
        cv2.putText(im, estado.aviso, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-por-clase", type=int, default=20)
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--lado", type=int, default=16)
    ap.add_argument("--k", type=int, default=5, help="máximo de bloques por imagen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hechas, conf = ya_anotadas()
    if conf not in (None, (args.lado, args.tam)):
        raise SystemExit(f"rois.csv se anotó con (lado, tam) = {conf}; usá los mismos valores.")
    rng = random.Random(args.seed)
    muestra, origen = armar_muestra(args.n_por_clase, rng)
    rng.shuffle(muestra)   # mezcla las clases para no anotar todas las infected seguidas
    pendientes = [(p, cl) for p, cl in muestra if p not in hechas]
    print(f"Muestra: {origen}. Total {len(muestra)} | ya anotadas {len(muestra) - len(pendientes)} "
          f"| pendientes {len(pendientes)}")

    vent = "anotar ROI"
    cv2.namedWindow(vent, cv2.WINDOW_AUTOSIZE)
    for i, (p, cl) in enumerate(pendientes, 1):
        est = Bloques(base(p, args.tam), args.lado, args.k)
        cv2.setMouseCallback(vent, lambda ev, mx, my, fl, par, e=est:
                             e.agregar(mx // ZOOM, my // ZOOM) if ev == cv2.EVENT_LBUTTONDOWN else None)
        while True:
            cv2.imshow(vent, dibujar(est, f"{i}/{len(pendientes)}  bloques: {len(est.lista)}/{args.k}  "
                                          f"| clic=agregar  u=deshacer  ENTER=guardar  s=saltear  q=salir"))
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord("u"):
                est.deshacer()
            elif tecla in (13, 32):
                if est.lista:
                    guardar(p, cl, est.lista, args.lado, args.tam)
                    break
                est.aviso = "marcá al menos un bloque (o s para saltear)"
            elif tecla == ord("s"):
                break
            elif tecla in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Listo. Marcas en {ROIS}")
                return
    cv2.destroyAllWindows()
    print(f"Terminaste la muestra. Marcas en {ROIS}")


if __name__ == "__main__":
    main()