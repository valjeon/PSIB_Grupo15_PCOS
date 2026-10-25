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

## Orden en que se corren los archivos:
python merge_data.py
python clean_data.py
python audit_data.py
python make_splits.py
python comparar_preprocesamiento.py
python procesar_dataset.py