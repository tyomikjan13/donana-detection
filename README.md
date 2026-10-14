# 🦌 Doñana Wildlife Detection

Reconocimiento de animales y sus bounding boxes en datos de cámaras trampa del **Parque Nacional de Doñana**, procedentes del proyecto [WildINTEL](https://wildintel.eu/).

## Dataset

El dataset **DONA_24** contiene observaciones de fauna silvestre capturadas entre el **19 de octubre y el 18 de noviembre de 2020** en Doñana, publicadas en formato [CamTrap-DP](https://camtrap-dp.tdwg.org/).

| Característica | Valor |
|---|---|
| Observaciones | 21.998 registros |
| Cámaras desplegadas | 38 (37 tienen datos) |
| Especies identificadas | 15 taxones |
| Imágenes | 19.616 registros / ~9.700 descargadas |
| Período | Oct–Nov 2020 |
| Licencia | CC-BY-4.0 |

### Especies registradas

| Nombre científico | Nombre común | Registros |
|---|---|---|
| *Cervidae* | Cérvido (ciervo/gamo) | 4.618 |
| *Cervus elaphus* | Ciervo rojo | 4.183 |
| *Sus scrofa* | Jabalí | 3.558 |
| *Bos taurus* | Vaca / Toro | 2.107 |
| *Dama dama* | Gamo | 874 |
| *Leporidae* | Lepórido (liebre/conejo) | 518 |
| *Vulpes vulpes* | Zorro rojo | 474 |
| *Oryctolagus cuniculus* | Conejo europeo | 174 |
| *Meles meles* | Tejón europeo | 144 |
| *Equus caballus* | Caballo | 90 |
| *Lepus granatensis* | Liebre ibérica | 65 |
| *Homo sapiens* | Humano | 55 |
| *Herpestes ichneumon* | Meloncillo | 50 |
| *Genetta genetta* | Gineta | 21 |
| *Lynx pardinus* | **Lince ibérico** 🐱 | 12 |

> **Nota:** 675 observaciones fueron excluidas del export público por estar vinculadas a imágenes no públicas (política de privacidad).

## Estructura del repositorio

```
.
├── README.md
├── pyproject.toml                  ← Configuración del proyecto (Python ≥3.14, uv)
├── instrucciones.txt               ← Pasos para descargar el dataset
├── dona_24_exploration.ipynb       ← Notebook de exploración del dataset
├── donana_lookup_photos.ipynb      ← Notebook de consulta de fotos
├── donana_bbox_detection.ipynb     ← Limpieza + detección de cajas con MegaDetector (Colab)
├── salida_megadetector/            ← Cajas detectadas (salida del notebook anterior)
└── dona_24/                        ← Dataset DONA_24 (gitignored)
    ├── datapackage.json            ← Metadatos CamTrap-DP
    ├── deployments.csv             ← Despliegues de cámaras
    ├── camtrapdp-local.zip         ← Paquete con rutas locales
    ├── checksums-sha256.txt        ← Verificación de integridad
    ├── images/                     ← Imágenes JPEG de las cámaras
    └── dona_24_pkg/                ← Paquete extraído
        ├── observations.csv        ← Observaciones (archivo principal)
        ├── media.csv               ← Registro de medios
        └── deployments.csv         ← Despliegues
```

## Instalación y uso

```bash
# 1. Instalar dependencias
uv sync

# 2. Descargar el dataset desde Hugging Face
hf download wildintelproject/dona_24 --repo-type dataset --local-dir ./dona_24

# 3. Extraer el paquete local
cd dona_24 && unzip camtrapdp-local.zip -d dona_24_pkg

# 4. Abrir los notebooks
jupyter lab
```

> Requiere [uv](https://docs.astral.sh/uv/) y Python ≥ 3.14.

## Usar las bounding boxes en un `Dataset` de PyTorch

### Cómo se obtuvieron las cajas

Usamos [MegaDetector](https://github.com/agentmorris/MegaDetector), un modelo ya entrenado con millones de fotos de cámaras trampa. Encuentra los animales de cada foto (sin decir de qué especie son) y da a cada uno una caja y una **confianza** de 0 a 1. Nos quedamos con las cajas de confianza ≥ 0,2.

Solo usamos fotos en las que los voluntarios vieron **un único animal**, así que la especie de la foto es la de la caja. Aun así, en algunas fotos aparece más de una caja: puede ser otro animal que no se contó, o algo que el modelo confundió con un animal (un arbusto, una sombra). En esos casos la especie se asigna a la caja con más confianza.

### Qué fotos usar

Recomendamos quedarse solo con las fotos que tienen **una sola caja y confianza alta**:

| Filtro | Fotos |
|---|---|
| Todas las fotos de un animal | 5.442 |
| Con una sola caja | 5.080 |
| Con una sola caja y confianza ≥ 0,8 (**recomendado**) | **4.646** (2.382 de día, 2.264 de noche) |
| Con una sola caja y confianza ≥ 0,9 | 3.770 (se pierden sobre todo fotos de noche) |

### Columnas de `salida_megadetector/single_animal_bboxes.csv`

| Columna | Contenido |
|---|---|
| `filePath` | Ruta de la imagen dentro de `dona_24/` |
| `className` | Clase para entrenar (ciervo rojo y gamo van juntos como `Cervidae`; conejo y liebre, como `Leporidae`) |
| `specificSpecies` | Especie concreta de esas dos familias, si se conoce |
| `bboxX, bboxY, bboxWidth, bboxHeight` | Caja: esquina superior izquierda, ancho y alto, como fracción del tamaño de la imagen (0 a 1) |
| `detectionConfidence` | Confianza de la caja |
| `extraAnimalBoxes` | Cuántas cajas más se encontraron en la foto |
| `deploymentID` | Cámara que hizo la foto |
| `lighting` | `día (color)` / `noche (IR)` |

### Ejemplo

Descargar antes el dataset (ver *Instalación y uso*). El `Dataset` recorta la caja de cada foto y devuelve el recorte con su clase. Train y test se separan **por cámara**, porque las fotos de una misma cámara se parecen mucho y mezclarlas daría resultados demasiado optimistas.

```python
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

DATASET_DIR = "dona_24"
MIN_CONFIDENCE = 0.8

boxes = pd.read_csv("salida_megadetector/single_animal_bboxes.csv")
boxes = boxes[(boxes["extraAnimalBoxes"] == 0) & (boxes["detectionConfidence"] >= MIN_CONFIDENCE)]

class_names = sorted(boxes["className"].unique())
class_index = {name: index for index, name in enumerate(class_names)}


class AnimalCropDataset(Dataset):
    def __init__(self, boxes, transform=None, margin=0.1):
        self.boxes = boxes.reset_index(drop=True)
        self.transform = transform
        self.margin = margin

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        row = self.boxes.iloc[index]
        image = Image.open(f"{DATASET_DIR}/{row.filePath}").convert("RGB")
        width, height = image.size
        margin_x, margin_y = row.bboxWidth * self.margin, row.bboxHeight * self.margin
        crop = image.crop((
            max(0, row.bboxX - margin_x) * width,
            max(0, row.bboxY - margin_y) * height,
            min(1, row.bboxX + row.bboxWidth + margin_x) * width,
            min(1, row.bboxY + row.bboxHeight + margin_y) * height,
        ))
        if self.transform:
            crop = self.transform(crop)
        return crop, class_index[row.className]


splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_rows, test_rows = next(splitter.split(boxes, groups=boxes["deploymentID"]))

# train_transform / test_transform: las transformaciones de cada uno (resize, aumentación, ToTensor…)
train_dataset = AnimalCropDataset(boxes.iloc[train_rows], transform=train_transform)
test_dataset = AnimalCropDataset(boxes.iloc[test_rows], transform=test_transform)
```

Hay especies con muy pocas fotos (lince 5, gineta 8, meloncillo 9). Será difícil que el modelo las aprenda; se pueden quitar o juntar en una clase "otros".

## Fuente y citación

Dataset publicado por el proyecto WildINTEL en Hugging Face: [`wildintelproject/dona_24`](https://huggingface.co/datasets/wildintelproject/dona_24)

> Manuel Emilio Gegundez Arias, Javier Calzada Samperio, Simone Santoro, Santiago Gutiérrez Zapata, Lenka Strakova, & Iñaki Fernández de Viana y González (2026). *DONA_24* (Version 1.0) [Data set]. Institute of Nature Conservation PAS.

## Licencia

El dataset se distribuye bajo **CC-BY-4.0**. El código de análisis de este repositorio es de uso académico.
