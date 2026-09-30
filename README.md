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
| *Cervidae* | Cérvido (sin especificar) | 4.618 |
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

## Fuente y citación

Dataset publicado por el proyecto WildINTEL en Hugging Face: [`wildintelproject/dona_24`](https://huggingface.co/datasets/wildintelproject/dona_24)

> Manuel Emilio Gegundez Arias, Javier Calzada Samperio, Simone Santoro, Santiago Gutiérrez Zapata, Lenka Strakova, & Iñaki Fernández de Viana y González (2026). *DONA_24* (Version 1.0) [Data set]. Institute of Nature Conservation PAS.

## Licencia

El dataset se distribuye bajo **CC-BY-4.0**. El código de análisis de este repositorio es de uso académico.
