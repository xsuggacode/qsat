"""Radar SMN (Bolívar RMA14, PNG ZH CMAX) -> grilla de lluvia mm/h en la misma malla que RRQPE.
El PNG es una imagen matplotlib: se lee por color contra la barra de escala y se georreferencia
por distancia al centro del radar (círculo de 240 km)."""
import io, re, math
from datetime import datetime, timezone
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import maximum_filter, uniform_filter

SMN_BASE = "https://estaticos.smn.gob.ar/vmsr/radar/"
LISTING_PAGES = ["https://meteo.deroweb.com.ar/imagen_radar-bolivar.php"]   # lista de las últimas imágenes
NAME_RE = re.compile(r"RMA14_240_ZH_CMAX_(\d{8})_(\d{6})Z\.png")

# Calibrado con la imagen de 1341x1336 px (centro por coordenadas de Chivilcoy, Junín y Bolívar)
IMG_SIZE = (1341, 1336)
CX, CY = 555.0, 692.0          # px del radar
PX_PER_KM = 2.1667             # círculo de 240 km = 520 px
RADAR_LAT, RADAR_LON = -36.18, -61.075
USE_KM = 232.0                 # más allá del borde dibujado no se usa
BAR_X, Y30, PX_PER_DBZ = 1216, 675.0, 11.2
MIN_DBZ, MAX_DBZ = 12.0, 57.0  # <12: ruido/clutter; >57: blanco/celeste de la escala = ruido de etiquetas
ZR_A, ZR_B = 200.0, 1.6        # Marshall-Palmer
MAX_MMH = 100.0
MAX_AGE_MIN = 25


def listing():
    """[(datetime UTC, url)] más nuevo primero, leyendo la página que lista las últimas imágenes."""
    out = {}
    for page in LISTING_PAGES:
        try:
            r = requests.get(page, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except Exception as e:
            print(f"radar: no pude leer {page}: {e}", flush=True); continue
        for m in NAME_RE.finditer(r.text):
            t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            out[t] = SMN_BASE + m.group(0)
    return sorted(out.items(), reverse=True)


def palette(im):
    ys = [Y30 - (d - 30) * PX_PER_DBZ for d in np.arange(MIN_DBZ, MAX_DBZ + 0.01, 0.5)]
    pal = np.array([im[int(round(y)), BAR_X] for y in ys], dtype=np.int16)
    return np.arange(MIN_DBZ, MAX_DBZ + 0.01, 0.5), pal


def to_dbz(png_bytes):
    """Devuelve (dbz 2D con NaN donde no hay eco válido, imagen RGB)."""
    im = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    if im.shape[1] != IMG_SIZE[0] or im.shape[0] != IMG_SIZE[1]:
        raise ValueError(f"tamaño de imagen inesperado {im.shape[1]}x{im.shape[0]}")
    vals, pal = palette(im)
    h, w, _ = im.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - CX, yy - CY) / PX_PER_KM
    inside = (dist <= USE_KM) & (xx < 1097) & (yy > 155)          # fuera: cortes verticales/horizontales y escala
    dbz = np.full((h, w), np.nan, np.float32)
    ys, xs = np.nonzero(inside)
    px = im[ys, xs].astype(np.int16)
    best = np.zeros(len(px), np.int32); bd = np.full(len(px), 1 << 30, np.int32)
    for i in range(len(vals)):
        d = np.abs(px - pal[i]).sum(axis=1)
        m = d < bd; bd[m] = d[m]; best[m] = i
    ok = bd <= 45
    dbz[ys[ok], xs[ok]] = vals[best[ok]]
    # Lo que no es ni escala ni fondo oscuro (polígono naranja de avisos, límites grises, nombres blancos)
    # es "desconocido": se completa con los vecinos si hay eco alrededor, y si no queda sin lluvia.
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    poly = (R >= 215) & (G >= 145) & (G <= 195) & (B <= 75)
    poly = maximum_filter(poly.astype(np.uint8), size=3).astype(bool)
    dbz[poly] = np.nan
    bg = (np.maximum(np.maximum(R, G), B) < 70) & ~poly
    valid = ~np.isnan(dbz)
    unknown = inside & ~valid & ~bg
    v0 = np.where(valid, dbz, 0.0)
    near_max = maximum_filter(v0, size=7)
    frac = uniform_filter(valid.astype(np.float32), size=7)
    fill = unknown & (frac >= 0.2)
    dbz = np.where(fill, near_max, dbz)
    dbz = np.where(dbz >= MIN_DBZ, dbz, np.nan).astype(np.float32)
    # manchas sueltas de 1-2 px (clutter) fuera
    m = ~np.isnan(dbz)
    keep = uniform_filter(m.astype(np.float32), size=5) >= 0.12
    dbz = np.where(keep, dbz, np.nan).astype(np.float32)
    return dbz, dist


def dbz_to_mmh(dbz):
    z = 10.0 ** (dbz / 10.0)
    return np.minimum((z / ZR_A) ** (1.0 / ZR_B), MAX_MMH)


def radar_grid(png_bytes, lat0, lon0, step, ny, nx):
    """Rellena la malla (ny,nx; fila 0 = norte) con mm/h máximo por celda. Devuelve (mmh, cubierto)."""
    dbz, _ = to_dbz(png_bytes)
    mmh_img = np.where(np.isnan(dbz), 0.0, dbz_to_mmh(np.nan_to_num(dbz, nan=0.0)))
    h, w = dbz.shape
    ys, xs = np.mgrid[0:h, 0:w]
    east = (xs - CX) / PX_PER_KM
    north = (CY - ys) / PX_PER_KM
    ok = np.hypot(east, north) <= USE_KM
    lat = RADAR_LAT + north / 111.19
    lon = RADAR_LON + east / (111.32 * np.cos(np.radians(RADAR_LAT)))
    r = np.rint((lat0 - lat[ok]) / step).astype(int)
    c = np.rint((lon[ok] - lon0) / step).astype(int)
    v = (r >= 0) & (r < ny) & (c >= 0) & (c < nx)
    mm = np.zeros((ny, nx), np.float32)
    np.maximum.at(mm, (r[v], c[v]), mmh_img[ok][v])
    cov = np.zeros((ny, nx), bool)
    cov[r[v], c[v]] = True
    # huecos de la malla dentro del disco (celdas sin píxel): se cierran con el vecino
    from scipy.ndimage import binary_closing, maximum_filter
    cov = binary_closing(cov, iterations=2)
    mm = np.where(cov, np.maximum(mm, maximum_filter(mm, size=3) * (mm == 0)), 0)
    return mm, cov


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0.0.0 Safari/537.36")
HEADER_VARIANTS = [
    {"User-Agent": UA, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8", "Accept-Language": "es-AR,es;q=0.9"},
    {"User-Agent": UA, "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8", "Accept-Language": "es-AR,es;q=0.9",
     "Referer": "https://meteo.deroweb.com.ar/"},
    {"User-Agent": UA, "Accept": "image/*,*/*;q=0.8", "Referer": "https://www.smn.gob.ar/"},
]


def download(url):
    """Baja el PNG probando cabeceras distintas (el SMN rechaza clientes que no parecen navegador)."""
    last = None
    for i, h in enumerate(HEADER_VARIANTS):
        try:
            r = requests.get(url, headers=h, timeout=60)
            if r.ok and r.content[:4] == b"\x89PNG":
                if i: print(f"radar: anduvo con la variante de cabeceras {i}", flush=True)
                return r.content
            last = f"{r.status_code} {r.headers.get('server', '')} {r.headers.get('content-type', '')} {len(r.content)}B"
        except Exception as e:
            last = str(e)
    raise RuntimeError(last)


def latest(now=None):
    """(tiempo, bytes) de la imagen más reciente que se pueda bajar y no sea vieja; None si no hay."""
    now = now or datetime.now(timezone.utc)
    for t, url in listing()[:3]:
        age = (now - t).total_seconds() / 60
        if age > MAX_AGE_MIN:
            print(f"radar: la más nueva tiene {age:.0f} min, muy vieja", flush=True); return None
        try:
            return t, download(url)
        except Exception as e:
            print(f"radar: falló {url.split('/')[-1]}: {e}", flush=True)
    return None
