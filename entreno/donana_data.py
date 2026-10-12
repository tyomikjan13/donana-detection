"""
Preparación de datos para detección de animales en DONA_24 (cámaras trampa de Doñana).

Solo depende de numpy / pandas / Pillow / scikit-learn, así que se puede ejecutar sin GPU.

Flujo:
    1. build_single_animal_images  -> imágenes con un único animal y su clase (misma limpieza que
                                      donana_bbox_detection.ipynb).
    2. build_annotations            -> une esas imágenes con las cajas de MegaDetector
                                      (megadetector_animal_boxes.csv).
    3. merge_rare_classes           -> agrupa las clases con muy pocas imágenes.
    4. assign_events                -> agrupa fotos de la misma ráfaga (misma cámara, segundos de diferencia).
    5. split_dataset                -> train / valid / test estratificado por clase y agrupado por ráfaga.
    6. cache_resized_images         -> copia reducida de las imágenes (entrenamiento mucho más rápido)
                                      y clasificación día / noche (IR).
"""
from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------------------------
# Clases
# ----------------------------------------------------------------------------------------------
FAMILY_OF_SPECIES = {
    "Cervus elaphus": "Cervidae",
    "Dama dama": "Cervidae",
    "Oryctolagus cuniculus": "Leporidae",
    "Lepus granatensis": "Leporidae",
}

SPANISH_CLASS_NAMES = {
    "Cervidae": "Cérvido",
    "Leporidae": "Lepórido",
    "Sus scrofa": "Jabalí",
    "Bos taurus": "Vaca",
    "Vulpes vulpes": "Zorro",
    "Meles meles": "Tejón",
    "Equus caballus": "Caballo",
    "Herpestes ichneumon": "Meloncillo",
    "Genetta genetta": "Gineta",
    "Lynx pardinus": "Lince",
}

BOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]  # normalizadas a [0, 1]


# ----------------------------------------------------------------------------------------------
# 1. Etiquetas de imagen (CamTrap-DP)
# ----------------------------------------------------------------------------------------------
def extract_camtrap_package(dataset_dir: Path, package_dir: Path) -> None:
    """Descomprime camtrapdp-local.zip (observations.csv, media.csv...) si no se ha hecho ya."""
    if (package_dir / "observations.csv").exists():
        return
    with zipfile.ZipFile(dataset_dir / "camtrapdp-local.zip") as archive:
        archive.extractall(package_dir)


def build_single_animal_images(dataset_dir: Path, package_dir: Path) -> pd.DataFrame:
    """
    Imágenes en disco con una sola observación de un único animal (count == 1).
    En ellas la especie de la foto es, sin ambigüedad, la especie de la caja.
    Devuelve: mediaID, deploymentID, timestamp, filePath, className, specificSpecies.
    """
    observations = pd.read_csv(package_dir / "observations.csv")
    media = pd.read_csv(package_dir / "media.csv")

    media = media[media["filePath"].map(lambda p: (dataset_dir / p).exists())]

    obs = observations[observations["observationLevel"] == "media"]
    obs = obs[obs["mediaID"].isin(media["mediaID"])]
    obs = obs[obs.groupby("mediaID")["mediaID"].transform("size") == 1]
    obs = obs[(obs["observationType"] == "animal") & obs["scientificName"].notna() & (obs["count"] == 1)]

    images = media[["mediaID", "deploymentID", "timestamp", "filePath"]].merge(
        obs[["mediaID", "scientificName"]], on="mediaID"
    )
    images["className"] = images["scientificName"].replace(FAMILY_OF_SPECIES)
    images["specificSpecies"] = images["scientificName"].where(images["scientificName"].isin(FAMILY_OF_SPECIES))
    return images.drop(columns="scientificName").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 2. Cajas de MegaDetector -> anotaciones
# ----------------------------------------------------------------------------------------------
def build_annotations(
    images: pd.DataFrame,
    boxes_csv: Path,
    min_box_confidence: float = 0.5,
    drop_ambiguous: bool = True,
    ambiguous_confidence: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Una caja por imagen: la de mayor confianza de MegaDetector (en las fotos de un animal es la suya).

    :param min_box_confidence: se descartan imágenes cuya mejor caja no llega a esta confianza
        (pseudo-etiquetas poco fiables).
    :param drop_ambiguous: descarta imágenes con 2+ cajas >= ambiguous_confidence. En una foto
        etiquetada como "un animal" eso suele indicar un segundo animal no anotado o un error;
        si se dejaran, el modelo aprendería que ese segundo animal es fondo.
    :return: (anotaciones, embudo con cuántas imágenes quedan en cada paso)
    """
    boxes = pd.read_csv(boxes_csv)
    boxes = boxes[boxes["filePath"].isin(images["filePath"])]

    best = boxes.sort_values("detectionConfidence", ascending=False).drop_duplicates("filePath")
    confident_counts = boxes[boxes["detectionConfidence"] >= ambiguous_confidence].groupby("filePath").size()

    ann = images.merge(best, on="filePath", how="inner")
    ann["nConfidentBoxes"] = ann["filePath"].map(confident_counts).fillna(0).astype(int)

    funnel = [("Imágenes de un animal", len(images)), ("Con caja de MegaDetector", len(ann))]
    ann = ann[ann["detectionConfidence"] >= min_box_confidence]
    funnel.append((f"Mejor caja con confianza >= {min_box_confidence}", len(ann)))
    if drop_ambiguous:
        ann = ann[ann["nConfidentBoxes"] <= 1]
        funnel.append((f"Sin 2ª caja con confianza >= {ambiguous_confidence}", len(ann)))

    # CamTrap-DP / MegaDetector: (x, y, ancho, alto) normalizados -> esquinas normalizadas
    ann = ann.assign(
        x_min=ann["bboxX"].clip(0, 1),
        y_min=ann["bboxY"].clip(0, 1),
        x_max=(ann["bboxX"] + ann["bboxWidth"]).clip(0, 1),
        y_max=(ann["bboxY"] + ann["bboxHeight"]).clip(0, 1),
    )
    ann = ann[(ann["x_max"] > ann["x_min"]) & (ann["y_max"] > ann["y_min"])]
    ann = ann.drop(columns=["bboxX", "bboxY", "bboxWidth", "bboxHeight"]).reset_index(drop=True)
    return ann, pd.DataFrame(funnel, columns=["Paso", "Imágenes"])


# ----------------------------------------------------------------------------------------------
# 3. Clases raras
# ----------------------------------------------------------------------------------------------
def merge_rare_classes(ann: pd.DataFrame, min_images: int | None, other_name: str = "Otros") -> pd.DataFrame:
    """
    Clases con menos de `min_images` imágenes se agrupan en `other_name` (con 5 linces no se puede
    ni entrenar ni evaluar de forma fiable). min_images=None mantiene todas las clases.
    Añade label_name (nombre en español) y label (entero, 1..N; 0 es el fondo en torchvision).
    """
    ann = ann.copy()
    ann["label_name"] = ann["className"].map(SPANISH_CLASS_NAMES).fillna(ann["className"])
    if min_images:
        counts = ann["label_name"].value_counts()
        rare = counts[counts < min_images].index
        ann.loc[ann["label_name"].isin(rare), "label_name"] = other_name

    order = ann["label_name"].value_counts().index.tolist()  # de más a menos frecuente
    if other_name in order:  # "Otros" siempre al final
        order = [c for c in order if c != other_name] + [other_name]
    ann["label"] = ann["label_name"].map({name: i + 1 for i, name in enumerate(order)}).astype(int)
    return ann


def class_names_from(ann: pd.DataFrame) -> list[str]:
    """Lista de nombres; class_names[label - 1] es el nombre de la etiqueta `label`."""
    pairs = ann[["label", "label_name"]].drop_duplicates().sort_values("label")
    return pairs["label_name"].tolist()


# ----------------------------------------------------------------------------------------------
# 4. Ráfagas (eventos)
# ----------------------------------------------------------------------------------------------
def assign_events(ann: pd.DataFrame, gap_seconds: int = 60) -> pd.DataFrame:
    """
    Las cámaras trampa disparan ráfagas de fotos casi idénticas. Si una foto de la ráfaga cae en train
    y otra en test, el test mide memoria y no generalización. Se agrupan en un mismo evento las fotos
    de la misma cámara separadas menos de `gap_seconds`.
    """
    ann = ann.assign(_ts=pd.to_datetime(ann["timestamp"], utc=True, errors="coerce"))
    ann = ann.sort_values(["deploymentID", "_ts"]).reset_index(drop=True)
    ts = ann.pop("_ts")

    new_camera = ann["deploymentID"] != ann["deploymentID"].shift()
    gap = ts.diff().dt.total_seconds()
    new_event = new_camera | gap.isna() | (gap > gap_seconds)
    ann["eventID"] = ann["deploymentID"].astype(str) + "_" + new_event.cumsum().astype(str)
    return ann


# ----------------------------------------------------------------------------------------------
# 5. Split
# ----------------------------------------------------------------------------------------------
def _stratified_group_holdout(labels, groups, fraction, seed):
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, int(round(1 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, holdout_idx = next(splitter.split(np.zeros(len(labels)), labels, groups))
    return keep_idx, holdout_idx


def split_dataset(
    ann: pd.DataFrame,
    test_fraction: float = 0.2,
    valid_fraction: float = 0.15,
    group_by: str = "eventID",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Añade la columna split (train / valid / test):
      - estratificado por clase (todas las clases aparecen en cada partición, en proporción similar);
      - agrupado por `group_by`: "eventID" (ráfagas) o "deploymentID" (cámaras completas: test con
        cámaras nunca vistas, más exigente).
    valid_fraction es la fracción de lo que queda tras separar el test.
    """
    ann = ann.reset_index(drop=True).copy()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # avisos de clases con menos grupos que folds
        rest_idx, test_idx = _stratified_group_holdout(ann["label"].values, ann[group_by].values, test_fraction, seed)
        rest = ann.iloc[rest_idx]
        train_rel, valid_rel = _stratified_group_holdout(
            rest["label"].values, rest[group_by].values, valid_fraction, seed
        )
    ann["split"] = "train"
    ann.loc[test_idx, "split"] = "test"
    ann.loc[rest.index[valid_rel], "split"] = "valid"
    return ann


def split_summary(ann: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(ann["label_name"], ann["split"], margins=True, margins_name="Total")
    cols = [c for c in ["train", "valid", "test", "Total"] if c in table.columns]
    order = class_names_from(ann) + ["Total"]
    return table.loc[[r for r in order if r in table.index], cols]


# ----------------------------------------------------------------------------------------------
# 6. Caché de imágenes reducidas + día / noche
# ----------------------------------------------------------------------------------------------
INFRARED_MAX_MEAN_SATURATION = 0.05  # foto nocturna IR = prácticamente sin color


def _cache_one(src: Path, dst: Path, max_side: int) -> tuple[int, int, float]:
    if not dst.exists():
        with Image.open(src) as im:
            im.draft("RGB", (max_side, max_side))  # decodificación JPEG reducida: mucho más rápido
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=92)
    with Image.open(dst) as im:
        width, height = im.size
        small = im.convert("RGB").resize((160, 90)).convert("HSV")
        saturation = float(np.asarray(small)[..., 1].mean() / 255)
    return width, height, saturation


def cache_resized_images(
    ann: pd.DataFrame, dataset_dir: Path, cache_dir: Path, max_side: int = 1024, workers: int = 8
) -> pd.DataFrame:
    """
    Guarda una copia de cada imagen con el lado mayor = max_side (las cajas están normalizadas, así que
    no cambian). Añade width, height (de la copia) y lighting ("día" / "noche (IR)").
    """
    from tqdm.auto import tqdm

    cache_dir = Path(cache_dir)
    jobs = [(Path(dataset_dir) / p, cache_dir / p, max_side) for p in ann["filePath"]]
    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(lambda job: _cache_one(*job), jobs), total=len(jobs), desc="Caché"))
    ann = ann.copy()
    ann["width"] = [r[0] for r in results]
    ann["height"] = [r[1] for r in results]
    ann["lighting"] = np.where(np.array([r[2] for r in results]) < INFRARED_MAX_MEAN_SATURATION, "noche (IR)", "día")
    return ann


def zip_directory(src_dir: Path, zip_path: Path) -> None:
    """Empaqueta la caché (sin comprimir: los JPEG ya lo están) para guardarla en Drive."""
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for file in sorted(src_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(src_dir.parent))


def ensure_image_cache(images_dir: Path, zip_path: Path) -> None:
    """Si la caché de imágenes no está en el disco local (sesión nueva de Colab), la descomprime desde Drive."""
    images_dir = Path(images_dir)
    if images_dir.exists() and any(images_dir.rglob("*.JPEG")):
        print(f"Imágenes listas en {images_dir}")
        return
    print(f"Descomprimiendo {zip_path} -> {images_dir.parent} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(images_dir.parent)
    print("Hecho")


def save_dataset(ann: pd.DataFrame, output_dir: Path, extra_info: dict | None = None) -> None:
    """Guarda annotations.csv (todas las particiones), un CSV por partición y dataset_info.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ann.to_csv(output_dir / "annotations.csv", index=False)
    for split in ["train", "valid", "test"]:
        ann[ann["split"] == split].to_csv(output_dir / f"annotations_{split}.csv", index=False)
    info = {"class_names": class_names_from(ann), **(extra_info or {})}
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
