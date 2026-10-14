"""Mapa de las cámaras (deployments.csv) sobre el Parque Nacional de Doñana.

Uso: python mapa_camaras.py donana_parque_nacional.geojson
"""
import io
import json
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

ZOOM = 11
TILE_SIZE = 256
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
HEADERS = {"User-Agent": "metodos-uni/1.0"}
DEPLOYMENTS_CSV = "../dona_24/deployments.csv"
OUTPUT_PNG = "mapa_camaras_donana.png"
MARGIN_LAT, MARGIN_LON = 0.03, 0.05

CAM_COLOR = "#d62828"
PARK_FILL, PARK_EDGE = "#2a9d3f", "#1b6b2a"


def latlon_to_tile(lat, lon, zoom=ZOOM):
    """Coordenadas fraccionarias de tesela (Web Mercator)."""
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def load_park_rings(path):
    """Anillos exteriores del polígono/multipolígono, como listas de (lon, lat)."""
    geom = json.load(open(path))["features"][0]["geometry"]
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]


def fetch_basemap(lats, lons):
    """Descarga y une las teselas que cubren el área. Devuelve (imagen, x0, y0)."""
    x0, y1 = latlon_to_tile(min(lats) - MARGIN_LAT, min(lons) - MARGIN_LON)
    x1, y0 = latlon_to_tile(max(lats) + MARGIN_LAT, max(lons) + MARGIN_LON)
    x0, x1, y0, y1 = int(x0), int(x1) + 1, int(y0), int(y1) + 1

    img = Image.new("RGB", ((x1 - x0) * TILE_SIZE, (y1 - y0) * TILE_SIZE))
    for x in range(x0, x1):
        for y in range(y0, y1):
            resp = requests.get(TILE_URL.format(z=ZOOM, x=x, y=y), headers=HEADERS)
            resp.raise_for_status()
            tile = Image.open(io.BytesIO(resp.content))
            img.paste(tile, ((x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE))
    return img, x0, y0


def draw_map(img, to_px, cams, rings):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(img)

    for ring in rings:
        xs, ys = zip(*[to_px(lat, lon) for lon, lat in ring])
        ax.fill(xs, ys, color=PARK_FILL, alpha=.25)
        ax.plot(xs, ys, color=PARK_EDGE, lw=1.5)

    cx, cy = zip(*[to_px(lat, lon) for lat, lon in zip(cams.latitude, cams.longitude)])
    ax.scatter(cx, cy, s=45, c=CAM_COLOR, edgecolor="white", lw=.8, zorder=3)
    for x, y, name in zip(cx, cy, cams.locationID.str.replace("dona_", "")):
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6,
                    bbox=dict(boxstyle="round,pad=.15", fc="white", alpha=.7, lw=0))

    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Cámaras trampa en el Parque Nacional de Doñana ({len(cams)} ubicaciones)")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=CAM_COLOR, label="Cámara"),
        plt.Rectangle((0, 0), 1, 1, fc=PARK_FILL, alpha=.4, ec=PARK_EDGE, label="Parque Nacional"),
    ], loc="lower left")
    ax.text(.99, .005, "© OpenStreetMap contributors", transform=ax.transAxes,
            ha="right", fontsize=6)
    fig.tight_layout()
    return fig


def main(polygon_path):
    cams = pd.read_csv(DEPLOYMENTS_CSV).drop_duplicates("locationID")
    rings = load_park_rings(polygon_path)

    lats = list(cams.latitude) + [lat for r in rings for _, lat in r]
    lons = list(cams.longitude) + [lon for r in rings for lon, _ in r]
    img, x0, y0 = fetch_basemap(lats, lons)

    def to_px(lat, lon):
        x, y = latlon_to_tile(lat, lon)
        return (x - x0) * TILE_SIZE, (y - y0) * TILE_SIZE

    fig = draw_map(img, to_px, cams, rings)
    fig.savefig(OUTPUT_PNG)


if __name__ == "__main__":
    main(sys.argv[1])
