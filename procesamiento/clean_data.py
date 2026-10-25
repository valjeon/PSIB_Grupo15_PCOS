"""Limpieza de data/all -> data/clean.

Excluye: archivos que no abren, imágenes casi vacías, duplicados (pHash) y
grupos de duplicados con etiquetas contradictorias (misma imagen en dos clases).
Registra todo en reports/excluidas.csv y reports/resumen_limpieza.txt.
"""
from collections import Counter, namedtuple
from pathlib import Path
import csv
import shutil

import imagehash
from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
SRC = ROOT / "data" / "all"
DST = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
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
