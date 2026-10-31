# PSIB - Grupo 15: Detección de PCOS en ecografías

## Dataset

[PCOS detection using ultrasound images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)
(Kaggle). 3856 imágenes divididas originalmente en train y test, cada una
con sus clases infected / non-infected. El dataset original está en
`data/raw/`.

## Estructura

- `data/raw/`: dataset original, tal cual se descargó
- `data/all/`: train y test unificados (se genera con el script)
- `merge_data.py`: script de unificación

## Código (`procesamiento/`)

Cuatro archivos, cada uno con comandos (`python procesamiento/<archivo>.py <comando> -h`):

| Archivo | Comandos |
|---|---|
| `limpieza.py` | `limpiar`, `auditar`, `splits`, `duplicados` |
| `preprocesamiento.py` | `comparar`, `anotar-rois`, `elegir-filtro`, `procesar` |
| `segmentacion.py` | `comparar`, `ver-casos`, `anotar-mascaras` |
| `texturas.py` | (sin comandos) características de folículos y estroma |

## Orden en que se corren

```
python merge_data.py
python procesamiento/limpieza.py limpiar
python procesamiento/limpieza.py auditar
python procesamiento/limpieza.py splits
python procesamiento/limpieza.py duplicados
python procesamiento/preprocesamiento.py comparar
python procesamiento/preprocesamiento.py procesar
python procesamiento/segmentacion.py comparar --splits splits2.csv
python procesamiento/texturas.py --imagenes 3
```
