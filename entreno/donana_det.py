"""
Detección de animales en DONA_24 con modelos de torchvision + aumentación con Albumentations.

Adaptado de los notebooks train/test de clase (robots) a:
  - varias clases (especies / familias) en lugar de una sola;
  - anotaciones desde CSV (generadas en 01_preparar_datos.ipynb) en lugar de XML;
  - aumentación de datos con Albumentations aplicada solo al conjunto de entrenamiento.

Cambios respecto a la plantilla original (y por qué):
  - No se normaliza la imagen a mano: los modelos de detección de torchvision ya normalizan
    internamente (model.transform). Normalizar dos veces degrada el preentrenamiento en COCO.
  - Las imágenes se pasan como lista y no con torch.stack: así pueden tener tamaños distintos
    (los recortes de la aumentación cambian el tamaño).
  - El mejor modelo se elige por mAP en validación (no por la pérdida de entrenamiento).
  - Se guarda un checkpoint "last.pt" por época para poder reanudar si Colab se desconecta.
"""
from __future__ import annotations

import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models import detection
from torchvision.ops import box_iou
from tqdm.auto import tqdm


# ==============================================================================================
# Utilidades
# ==============================================================================================
def set_seed(seed: int) -> None:
    """Semilla para reproducibilidad (python, numpy y torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(use_gpu: bool = True) -> torch.device:
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def collate_fn(batch: list) -> tuple:
    """Agrupa imágenes y anotaciones como tuplas (no como tensor), como esperan los modelos de detección."""
    return tuple(zip(*batch))


def seed_worker(worker_id: int) -> None:
    """Cada worker del DataLoader con su propia semilla (si no, todos generarían la misma aumentación)."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None) if info else None
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


# ==============================================================================================
# Dataset
# ==============================================================================================
class DonanaDataset(Dataset):
    """
    Dataset de detección a partir de un DataFrame de anotaciones (una fila por caja).

    Aumentación de datos ("aumentar el dataset"):
        Cada imagen original aparece una vez SIN transformar y, además, `copies[i]` veces CON una
        transformación aleatoria de `transform`. Las copias se generan al vuelo, así que en cada época
        son versiones distintas, sin ocupar disco.
          - num_augments=0  -> solo originales (baseline).
          - num_augments=2  -> dataset x3: original + 2 versiones aumentadas por imagen.
          - copies=[...]    -> número de copias por imagen (p. ej. más copias para clases raras).

    :param annotations: DataFrame con filePath, x_min, y_min, x_max, y_max (normalizadas), label.
    :param images_dir: carpeta raíz donde están las imágenes (las rutas filePath son relativas a ella).
    :param transform: albumentations.Compose con bbox_params en formato pascal_voc (ver get_augmentation).
    :param augment_originals: si True la imagen original también se transforma (aumentación "online"
        clásica, sin aumentar el tamaño del dataset).
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        images_dir: str | Path,
        transform: A.Compose | None = None,
        num_augments: int = 0,
        copies: list[int] | None = None,
        augment_originals: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.augment_originals = augment_originals

        self.records = []
        for file_path, group in annotations.groupby("filePath", sort=False):
            self.records.append(
                {
                    "filePath": file_path,
                    "boxes": group[["x_min", "y_min", "x_max", "y_max"]].to_numpy(np.float32),
                    "labels": group["label"].to_numpy(np.int64),
                }
            )

        if copies is None:
            copies = [num_augments] * len(self.records)
        # Nota: sin transform, las "copias" son repeticiones exactas de la imagen. Sirve de control para
        # separar el efecto de la aumentación del simple hecho de hacer más iteraciones por época.
        assert len(copies) == len(self.records)

        # Índice: (registro, ¿aumentada?)
        self.index = [(i, False) for i in range(len(self.records))]
        for i, n in enumerate(copies):
            self.index += [(i, True)] * int(n)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def num_originals(self) -> int:
        return len(self.records)

    def load(self, record_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Imagen RGB (H, W, 3) uint8, cajas en píxeles [x_min, y_min, x_max, y_max] y etiquetas."""
        rec = self.records[record_idx]
        image = np.array(Image.open(self.images_dir / rec["filePath"]).convert("RGB"))
        h, w = image.shape[:2]
        boxes = rec["boxes"] * np.array([w, h, w, h], dtype=np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return image, boxes, rec["labels"].copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        record_idx, augmented = self.index[idx]
        image, boxes, labels = self.load(record_idx)

        if self.transform is not None and (augmented or self.augment_originals):
            out = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = out["image"]
            boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(out["class_labels"], dtype=np.int64).reshape(-1)

        # Cajas degeneradas (ancho/alto < 1 px) rompen la pérdida de Faster R-CNN
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        boxes, labels = boxes[keep], labels[keep]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return image_t, target


def balanced_copies(annotations: pd.DataFrame, target_per_class: int, max_copies: int = 10) -> list[int]:
    """
    Copias aumentadas por imagen para que cada clase llegue (aprox.) a `target_per_class` imágenes:
    las clases raras se aumentan mucho y las frecuentes nada. Mismo orden que DonanaDataset.records.
    """
    per_image = annotations.groupby("filePath", sort=False)["label"].first()
    counts = per_image.value_counts()
    copies_per_class = {
        c: int(np.clip(math.ceil(target_per_class / n) - 1, 0, max_copies)) for c, n in counts.items()
    }
    return [copies_per_class[c] for c in per_image.values]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2, seed: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )


# ==============================================================================================
# Aumentación de datos (Albumentations >= 2.0)
# ==============================================================================================
def _bbox_params() -> A.BboxParams:
    # pascal_voc = [x_min, y_min, x_max, y_max] en píxeles.
    # min_visibility: si tras un recorte queda < 30 % del animal, la caja se elimina.
    return A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3, min_area=16, clip=True)


def geometric_transforms(p: float = 1.0) -> list:
    """Cambios de posición / escala: el animal puede estar en cualquier sitio y a cualquier distancia."""
    return [
        A.HorizontalFlip(p=0.5 * p),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), rotate=(-5, 5), p=0.5 * p),
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.3 * p),  # recorte que nunca corta al animal (zoom)
    ]


def photometric_transforms(p: float = 1.0) -> list:
    """Cambios de iluminación y color: amanecer / mediodía / atardecer, distintas cámaras."""
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5 * p),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3 * p),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=0.3 * p),
        A.CLAHE(clip_limit=2.0, p=0.15 * p),
    ]


def camera_trap_transforms(p: float = 1.0) -> list:
    """
    Efectos propios de las cámaras trampa:
      - ToGray: imita las fotos nocturnas con infrarrojo (~50 % del dataset está en escala de grises).
      - Ruido gaussiano: sensor a ISO alto de noche.
      - Desenfoque de movimiento: animales en movimiento con exposiciones largas.
      - Compresión JPEG, niebla y sombras de la vegetación.
    """
    return [
        A.ToGray(p=0.2 * p),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3 * p),
        A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 5))], p=0.25 * p),
        A.ImageCompression(quality_range=(50, 90), p=0.3 * p),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.1 * p),
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.15 * p),
    ]


def occlusion_transforms(p: float = 1.0) -> list:
    """Huecos aleatorios: simulan ramas / hierba que tapan parte del animal (como el dirty_aug original)."""
    return [
        A.CoarseDropout(
            num_holes_range=(3, 8), hole_height_range=(0.03, 0.08), hole_width_range=(0.03, 0.08), fill=0, p=0.5 * p
        )
    ]


AUGMENTATIONS = {
    "geometrica": lambda: geometric_transforms(),
    "fotometrica": lambda: photometric_transforms(),
    "camara_trampa": lambda: camera_trap_transforms(),
    "oclusion": lambda: occlusion_transforms(),
    # Todo junto, con probabilidades más bajas para no deformar demasiado cada imagen
    "combinada": lambda: geometric_transforms(1.0)
    + photometric_transforms(0.6)
    + camera_trap_transforms(0.6)
    + occlusion_transforms(0.5),
}


def get_augmentation(name: str | None) -> A.Compose | None:
    """Pipeline de Albumentations por nombre (None / "ninguna" = sin aumentación)."""
    if name in (None, "ninguna", "none"):
        return None
    if name not in AUGMENTATIONS:
        raise ValueError(f"Aumentación '{name}' no definida. Opciones: {list(AUGMENTATIONS)}")
    return A.Compose(AUGMENTATIONS[name](), bbox_params=_bbox_params())


def show_augmentations(dataset: DonanaDataset, transform: A.Compose, class_names: list[str], record_idx: int = 0,
                       n: int = 4, title: str = "") -> None:
    """Original + `n` versiones aumentadas de una imagen, con sus cajas (equivale a test_augmentor)."""
    image, boxes, labels = dataset.load(record_idx)
    samples = [(image, boxes, labels, "Original")]
    for k in range(n):
        out = transform(image=image, bboxes=boxes, class_labels=labels)
        samples.append((out["image"], np.asarray(out["bboxes"]).reshape(-1, 4), out["class_labels"], f"Aumentada {k + 1}"))
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.2))
    for ax, (img, bxs, lbls, name) in zip(axes, samples):
        ax.imshow(img)
        for (x1, y1, x2, y2), lab in zip(bxs, lbls):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
            ax.text(x1, y1 - 4, class_names[int(lab) - 1], color="black", backgroundcolor="lime", fontsize=7)
        ax.set_title(f"{name} ({img.shape[1]}x{img.shape[0]})", fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ==============================================================================================
# Modelos
# ==============================================================================================
MODEL_BUILDERS = {
    "faster_rcnn_v1": detection.fasterrcnn_resnet50_fpn,
    "faster_rcnn_v2": detection.fasterrcnn_resnet50_fpn_v2,
    "faster_rcnn_v3": detection.fasterrcnn_mobilenet_v3_large_fpn,
    "retinanet_v1": detection.retinanet_resnet50_fpn,
    "retinanet_v2": detection.retinanet_resnet50_fpn_v2,
    "ssd_v1": detection.ssd300_vgg16,
    "ssd_v2": detection.ssdlite320_mobilenet_v3_large,
}


def torchvision_model(model_name: str, num_classes: int, pretrained: bool = True, image_size: int | None = 1024) -> Any:
    """
    Modelo de torchvision con la cabeza de clasificación adaptada a `num_classes`
    (= número de especies + 1, porque la clase 0 es el fondo).

    :param image_size: lado mayor con el que el modelo procesa la imagen. Con min_size = max_size = image_size
        torchvision escala la imagen hasta que su lado mayor mide image_size. Más grande = animales pequeños
        (conejos, ginetas) mejor detectados, pero más lento. No aplica a SSD (tamaño fijo 300/320).
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Modelo {model_name} no está en la lista: {list(MODEL_BUILDERS)}")
    kwargs = {} if pretrained else {"weights_backbone": None}  # al cargar un checkpoint no hace falta descargar nada
    if image_size and not model_name.startswith("ssd"):
        kwargs.update(min_size=image_size, max_size=image_size)
    model = MODEL_BUILDERS[model_name](weights="DEFAULT" if pretrained else None, **kwargs)

    if model_name.startswith("faster_rcnn"):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    elif model_name.startswith("retinanet"):
        head = model.head.classification_head
        in_channels = head.cls_logits.in_channels
        norm = {"norm_layer": partial(torch.nn.GroupNorm, 32)} if model_name == "retinanet_v2" else {}
        model.head.classification_head = detection.retinanet.RetinaNetClassificationHead(
            in_channels, head.num_anchors, num_classes, **norm
        )
    elif model_name == "ssd_v1":
        in_channels = [m.in_channels for m in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = detection.ssd.SSDClassificationHead(in_channels, num_anchors, num_classes)
    elif model_name == "ssd_v2":
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

        in_channels = [m[0][0].in_channels for m in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        )
    return model


def build_optimizer(model, lr: float, momentum: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def save_checkpoint(path, model, optimizer, scheduler, history, config) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "history": history,
            "config": config,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> tuple[Any, dict]:
    """Reconstruye el modelo a partir de un checkpoint (la config va guardada dentro) y carga los pesos."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = torchvision_model(cfg["model_name"], cfg["num_classes"], pretrained=False, image_size=cfg["image_size"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


# ==============================================================================================
# Entrenamiento
# ==============================================================================================
def _to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def _loss_mode(model) -> None:
    """
    Los modelos de torchvision solo devuelven pérdidas en modo train. Para la pérdida de validación se pone
    en train pero con las BatchNorm en eval, para no modificar sus estadísticas con datos de validación.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train_one_epoch(model, optimizer, loader, device, scaler=None, max_grad_norm: float = 0.0) -> float:
    model.train()
    losses = []
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = _to_device(images, targets, device)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Pérdida no finita ({loss.item()}): baja el learning rate o activa max_grad_norm.")

        optimizer.zero_grad(set_to_none=True)  # Reset gradients
        if scaler is not None:
            scaler.scale(loss).backward()  # Backpropagation
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Clip gradients
            scaler.step(optimizer)  # Update weights
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    _loss_mode(model)
    losses = []
    for images, targets in tqdm(loader, desc="Val loss", leave=False):
        images, targets = _to_device(images, targets, device)
        losses.append(sum(model(images, targets).values()).item())
    model.eval()
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[dict], list[dict]]:
    """Predicciones y anotaciones reales (en CPU) para todo el loader."""
    model.eval()
    preds, gts = [], []
    for images, targets in tqdm(loader, desc="Predict", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds += [{k: v.cpu() for k, v in o.items()} for o in outputs]
        gts += [{k: v.cpu() for k, v in t.items()} for t in targets]
    return preds, gts


def fit(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    device,
    num_epochs: int,
    output_dir: str | Path,
    config: dict,
    early_stopping_patience: int | None = None,
    use_amp: bool = True,
    max_grad_norm: float = 0.0,
) -> dict:
    """
    Bucle de entrenamiento. En cada época: train -> pérdida de validación -> mAP de validación.
    Guarda last.pt (para reanudar) y best.pt (mejor mAP@[.5:.95] en validación).
    Si existe output_dir/last.pt se reanuda desde ahí.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"

    history = {"train_loss": [], "valid_loss": [], "valid_map": [], "valid_map_50": [], "lr": [],
               "best_value": -1.0, "best_epoch": 0, "current_epoch": 0, "finished": False}
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        if history.get("finished"):
            print(f"Entrenamiento ya terminado ({last_path}); no se repite.")
            return history
        print(f"Reanudando desde la época {history['current_epoch']} ({last_path})")

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    for epoch in range(history["current_epoch"], num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, max_grad_norm)
        valid_loss = validation_loss(model, valid_loader, device)
        metrics = compute_map(*predict(model, valid_loader, device), class_metrics=False)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_map"].append(metrics["map"])
        history["valid_map_50"].append(metrics["map_50"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["current_epoch"] = epoch + 1

        if scheduler is not None:
            scheduler.step(metrics["map"])  # ReduceLROnPlateau(mode="max") sobre el mAP de validación

        improved = metrics["map"] > history["best_value"]
        if improved:
            history["best_value"], history["best_epoch"] = metrics["map"], epoch + 1
            save_checkpoint(best_path, model, None, None, history, config)
        save_checkpoint(last_path, model, optimizer, scheduler, history, config)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(
            f"Época {epoch + 1:>3}/{num_epochs} | train loss {train_loss:.4f} | valid loss {valid_loss:.4f} | "
            f"valid mAP {metrics['map']:.4f} | mAP@50 {metrics['map_50']:.4f} | lr {history['lr'][-1]:.1e}"
            + ("  <- mejor" if improved else "")
        )

        if early_stopping_patience and epoch + 1 - history["best_epoch"] >= early_stopping_patience:
            print(f"Early stopping: {early_stopping_patience} épocas sin mejorar el mAP de validación.")
            break
    history["finished"] = True
    save_checkpoint(last_path, model, optimizer, scheduler, history, config)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def plot_history(history: dict, save_path: str | Path | None = None) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["valid_loss"], label="valid loss")
    ax1.set_xlabel("Época"), ax1.set_ylabel("Pérdida"), ax1.legend()
    ax2.plot(epochs, history["valid_map"], label="valid mAP@[.5:.95]")
    ax2.plot(epochs, history["valid_map_50"], label="valid mAP@0.5")
    ax2.axvline(history["best_epoch"], color="gray", linestyle="--", label="mejor época")
    ax2.set_xlabel("Época"), ax2.set_ylabel("mAP"), ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()


# ==============================================================================================
# Evaluación
# ==============================================================================================
def compute_map(preds: list[dict], gts: list[dict], class_metrics: bool = True, class_agnostic: bool = False) -> dict:
    """
    mAP estilo COCO con torchmetrics.
      - class_agnostic=False: la caja y la especie tienen que ser correctas.
      - class_agnostic=True : solo importa encontrar al animal (todas las clases = "animal").
    """
    if class_agnostic:
        preds = [{**p, "labels": torch.ones_like(p["labels"])} for p in preds]
        gts = [{**g, "labels": torch.ones_like(g["labels"])} for g in gts]
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=class_metrics)
    metric.update(preds, gts)
    res = metric.compute()
    out = {k: v.item() for k, v in res.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
    if class_metrics and "map_per_class" in res:
        out["classes"] = res["classes"].reshape(-1).tolist()
        out["map_per_class"] = res["map_per_class"].reshape(-1).tolist()
    return out


def single_animal_metrics(preds: list[dict], gts: list[dict], iou_threshold: float = 0.5) -> dict:
    """
    Métricas fáciles de interpretar (cada foto tiene un único animal): se toma la detección con más score.
      - localizacion: % de fotos donde esa caja solapa (IoU >= 0.5) con el animal real.
      - acierto     : % de fotos con caja bien localizada Y especie correcta.
    """
    localized, correct = [], []
    for p, g in zip(preds, gts):
        if len(p["boxes"]) == 0 or len(g["boxes"]) == 0:
            localized.append(False), correct.append(False)
            continue
        top = int(p["scores"].argmax())
        iou = box_iou(p["boxes"][top : top + 1], g["boxes"]).max().item()
        ok = iou >= iou_threshold
        localized.append(ok)
        correct.append(ok and int(p["labels"][top]) == int(g["labels"][0]))
    return {"localizacion": float(np.mean(localized)), "acierto": float(np.mean(correct))}


def evaluate(model, loader, device, class_names: list[str], subsets: dict[str, np.ndarray] | None = None) -> dict:
    """
    Evaluación completa sobre un loader (sin barajar):
      mAP por especie, mAP agnóstico, métricas de foto única y, opcionalmente, por subconjunto
      (p. ej. {"día": máscara, "noche (IR)": máscara}).
    """
    preds, gts = predict(model, loader, device)
    results = {
        "map": compute_map(preds, gts, class_metrics=True),
        "map_agnostico": compute_map(preds, gts, class_metrics=False, class_agnostic=True),
        **single_animal_metrics(preds, gts),
    }
    per_class = dict(zip(results["map"].get("classes", []), results["map"].get("map_per_class", [])))
    results["ap_por_clase"] = {class_names[int(c) - 1]: v for c, v in per_class.items() if 0 < int(c) <= len(class_names)}
    for name, mask in (subsets or {}).items():
        sp = [p for p, m in zip(preds, mask) if m]
        sg = [g for g, m in zip(gts, mask) if m]
        if sp:
            results[f"map_{name}"] = compute_map(sp, sg, class_metrics=False)["map"]
    results["_preds"], results["_gts"] = preds, gts
    return results


def summary_row(results: dict) -> dict:
    """Fila resumen (para tablas comparativas)."""
    row = {
        "mAP": results["map"]["map"],
        "mAP@50": results["map"]["map_50"],
        "mAP@75": results["map"]["map_75"],
        "mAP agnóstico": results["map_agnostico"]["map"],
        "mAP@50 agnóstico": results["map_agnostico"]["map_50"],
        "Localización": results["localizacion"],
        "Acierto especie": results["acierto"],
    }
    row.update({k: v for k, v in results.items() if k.startswith("map_") and isinstance(v, float)})
    return row


def plot_predictions(image, prediction: dict, class_names: list[str], target: dict | None = None,
                     score_threshold: float = 0.5, ax=None, title: str = "") -> None:
    """Dibuja las predicciones (verde) y, si se da, la caja real (rojo discontinuo)."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    if target is not None:
        for (x1, y1, x2, y2), lab in zip(target["boxes"].tolist(), target["labels"].tolist()):
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linestyle="--", linewidth=2))
            ax.text(x1, y2 + 12, f"real: {class_names[lab - 1]}", color="white", fontsize=8,
                    bbox=dict(facecolor="red", alpha=0.6, edgecolor="none", pad=1))
    for (x1, y1, x2, y2), lab, score in zip(prediction["boxes"].tolist(), prediction["labels"].tolist(), prediction["scores"].tolist()):
        if score < score_threshold:
            continue
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        ax.text(x1, y1 - 6, f"{class_names[lab - 1]} {score:.2f}", color="white", fontsize=8, weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if show:
        plt.show()


def plot_confusion(preds: list[dict], gts: list[dict], class_names: list[str], iou_threshold: float = 0.5,
                   score_threshold: float = 0.3) -> pd.DataFrame:
    """
    Matriz de confusión por foto: especie real vs especie de la detección con más score.
    "No detectado" = ninguna detección >= score_threshold con IoU >= iou_threshold.
    """
    names = class_names + ["No detectado"]
    mat = np.zeros((len(class_names), len(names)), dtype=int)
    for p, g in zip(preds, gts):
        true = int(g["labels"][0]) - 1
        keep = p["scores"] >= score_threshold
        boxes, labels, scores = p["boxes"][keep], p["labels"][keep], p["scores"][keep]
        if len(boxes):
            top = int(scores.argmax())
            if box_iou(boxes[top : top + 1], g["boxes"]).max().item() >= iou_threshold:
                mat[true, int(labels[top]) - 1] += 1
                continue
        mat[true, -1] += 1
    df = pd.DataFrame(mat, index=class_names, columns=names)
    norm = df.div(df.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3, 0.7 * len(class_names) + 2))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j]:
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                        color="white" if norm.values[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicción"), ax.set_ylabel("Real")
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.show()
    return df
