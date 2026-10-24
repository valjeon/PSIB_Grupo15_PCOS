from pathlib import Path
import csv
import shutil
import imagehash
from PIL import Image

SRC = Path("data/all")
DST = Path("data/clean")
UMBRAL = 4  # distancia máxima entre hashes para considerar "duplicada"

excluidas = []
candidatas = []

for clase_dir in sorted(SRC.iterdir()):
    if not clase_dir.is_dir():
        continue
    for ruta in sorted(clase_dir.iterdir()):
        try:
            with Image.open(ruta) as img:
                img.load()  # detecta también archivos truncados
                h = int(str(imagehash.phash(img)), 16)
                area = img.size[0] * img.size[1]
        except Exception:
            excluidas.append((clase_dir.name, ruta.name, "no se pudo abrir", ""))
            continue
        candidatas.append((area, clase_dir.name, ruta, h))

# De cada grupo de copias se conserva la de mayor resolución
candidatas.sort(key=lambda c: (-c[0], c[2].name))

conservadas = []
for area, clase, ruta, h in candidatas:
    previa = next((k for k in conservadas if (k[3] ^ h).bit_count() <= UMBRAL), None)
    if previa is None:
        conservadas.append((area, clase, ruta, h))
    else:
        motivo = "duplicada" if previa[1] == clase else "duplicada (clase distinta)"
        excluidas.append((clase, ruta.name, motivo, previa[2].name))

# data/clean se regenera siempre desde cero
if DST.exists():
    shutil.rmtree(DST)
for _, clase, ruta, _ in conservadas:
    (DST / clase).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ruta, DST / clase / ruta.name)

with open("excluidas.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["clase", "archivo", "motivo", "copia_de"])
    w.writerows(excluidas)

print("Umbral:", UMBRAL)
print("Imágenes en data/all:", len(candidatas) + sum(e[2] == "no se pudo abrir" for e in excluidas))
print("Excluidas:", len(excluidas))
print("  no se pudo abrir:", sum(e[2] == "no se pudo abrir" for e in excluidas))
print("  duplicadas:", sum(e[2].startswith("duplicada") for e in excluidas))
print("Conservadas:", len(conservadas))
for clase in sorted({c[1] for c in conservadas}):
    print(f"  {clase}:", sum(c[1] == clase for c in conservadas))