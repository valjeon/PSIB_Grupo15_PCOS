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
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import random
import time

import cv2
import numpy as np

from comparar_preprocesamiento import hoja
from segmentacion import segment_detalle, superponer
from scipy import stats

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"


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

def main():
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
    args = ap.parse_args()
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


if __name__ == "__main__":
    main()