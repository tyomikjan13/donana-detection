"""Mapa de las cámaras (deployments.csv) sobre el Parque Nacional de Doñana."""
import io, json, math, sys
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

POLY = sys.argv[1]  # geojson de Nominatim con el parque
Z = 11
H = {"User-Agent": "metodos-uni/1.0"}

def tile(lat, lon):
    n = 2 ** Z
    return ((lon + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)

def inv(x, y):
    n = 2 ** Z
    return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))),
            x / n * 360 - 180)

cams = pd.read_csv("dona_24/deployments.csv").drop_duplicates("locationID")
park = json.load(open(POLY))["features"][0]["geometry"]
rings = [park["coordinates"][0]] if park["type"] == "Polygon" else [p[0] for p in park["coordinates"]]

lats = list(cams.latitude) + [c[1] for r in rings for c in r]
lons = list(cams.longitude) + [c[0] for r in rings for c in r]
x0, y1 = tile(min(lats) - .03, min(lons) - .05)
x1, y0 = tile(max(lats) + .03, max(lons) + .05)
x0, x1, y0, y1 = int(x0), int(x1) + 1, int(y0), int(y1) + 1
img = Image.new("RGB", ((x1 - x0) * 256, (y1 - y0) * 256))
for x in range(x0, x1):
    for y in range(y0, y1):
        t = requests.get(f"https://tile.openstreetmap.org/{Z}/{x}/{y}.png", headers=H).content
        img.paste(Image.open(io.BytesIO(t)), ((x - x0) * 256, (y - y0) * 256))

def px(lat, lon):
    x, y = tile(lat, lon)
    return (x - x0) * 256, (y - y0) * 256

fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
ax.imshow(img)
for r in rings:
    xs, ys = zip(*[px(lat, lon) for lon, lat in r])
    ax.fill(xs, ys, color="#2a9d3f", alpha=.25)
    ax.plot(xs, ys, color="#1b6b2a", lw=1.5)
cx, cy = zip(*[px(a, b) for a, b in zip(cams.latitude, cams.longitude)])
ax.scatter(cx, cy, s=45, c="#d62828", edgecolor="white", lw=.8, zorder=3)
for x, y, n in zip(cx, cy, cams.locationID.str.replace("dona_", "")):
    ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6,
                bbox=dict(boxstyle="round,pad=.15", fc="white", alpha=.7, lw=0))
ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0)
ax.set_title(f"Cámaras trampa en el Parque Nacional de Doñana ({len(cams)} ubicaciones)")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#d62828", label="Cámara"),
                   plt.Rectangle((0, 0), 1, 1, fc="#2a9d3f", alpha=.4, ec="#1b6b2a", label="Parque Nacional")],
          loc="lower left")
ax.set_xticks([]); ax.set_yticks([])
ax.text(.99, .005, "© OpenStreetMap contributors", transform=ax.transAxes, ha="right", fontsize=6)
fig.tight_layout()
fig.savefig("mapa_camaras_donana.png")
