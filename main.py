"""QSAT: lee GOES-19 (banda 13, IR 10.3 µm) y entrega al Worker una grilla liviana de
temperatura de brillo (BT) sobre Argentina. No guarda imágenes: todo en memoria.

Uso:
  python main.py                 # descarga lo último de NOAA y lo manda al Worker
  python main.py --file x.nc --out grid.json   # prueba local, sin red
Variables de entorno: QSAT_INGEST_URL, QSAT_KEY
"""
import argparse, base64, io, json, os, re, sys, time, zlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import requests
from scipy.ndimage import minimum_filter

from geos import GeosProj

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPF"
CHANNEL = "C13"
# Grilla de salida (fila 0 = norte). 0.05° ≈ 5.5 km. Cubre toda Argentina continental + Tierra del Fuego.
LAT_N, LAT_S, LON_W, LON_E, STEP = -21.0, -56.0, -74.0, -53.0, 0.05
BT_OFFSET = 150.0          # q = round(BT_K - 150) -> 1 K de resolución, rango 150..405 K. 0 = sin dato
POOL = 3                   # ventana de mínimo (coge el tope más frío de ~6 km)
MAX_AGE_MIN = 40           # si el último archivo es más viejo, no se envía


def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def list_keys(prefix):
    r = requests.get(BUCKET, params={"list-type": "2", "prefix": prefix}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"s": root.tag.split("}")[0].strip("{")}
    return [k.text for k in root.findall(".//s:Contents/s:Key", ns)]


def scan_time(key):
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    y, doy, hh, mm, ss = map(int, m.groups())
    return datetime(y, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def latest_key(now=None):
    now = now or datetime.now(timezone.utc)
    for back in (0, 1, 2):
        t = now - timedelta(hours=back)
        prefix = f"{PRODUCT}/{t.year}/{t.timetuple().tm_yday:03d}/{t.hour:02d}/"
        keys = [k for k in list_keys(prefix) if f"M6{CHANNEL}_" in k or f"M3{CHANNEL}_" in k or f"M4{CHANNEL}_" in k]
        if keys:
            return max(keys, key=scan_time)
    return None


def read_var(f, name):
    """Lee una variable NetCDF4 aplicando _Unsigned, _FillValue, scale_factor y add_offset."""
    ds = f[name]
    a = ds[...]
    at = ds.attrs
    g = lambda n, d=None: (np.asarray(at[n]).ravel()[0] if n in at else d)
    uns = at.get("_Unsigned")
    if uns is not None and (uns.decode() if isinstance(uns, bytes) else str(uns)).lower() == "true" and a.dtype.kind == "i":
        a = a.view(a.dtype.str.replace("i", "u"))
    fill = g("_FillValue")
    if fill is not None:
        fill = np.array(fill).astype(a.dtype)
    bad = (a == fill) if fill is not None else np.zeros(a.shape, bool)
    a = a.astype(np.float32) * np.float32(g("scale_factor", 1.0)) + np.float32(g("add_offset", 0.0))
    a[bad] = np.nan
    return a


def build_grid(f):
    """Devuelve (q uint8 [ny,nx], meta) con BT cuantizada. No guarda nada en disco."""
    proj = GeosProj.from_attrs(f["goes_imager_projection"].attrs)
    xs = read_var(f, "x").astype(np.float64)
    ys = read_var(f, "y").astype(np.float64)

    lats = LAT_N - (np.arange(int(round((LAT_N - LAT_S) / STEP))) + 0.5) * STEP
    lons = LON_W + (np.arange(int(round((LON_E - LON_W) / STEP))) + 0.5) * STEP
    LON, LAT = np.meshgrid(lons, lats)
    x, y, vis = proj.latlon_to_scan(LAT, LON)

    dx, dy = xs[1] - xs[0], ys[1] - ys[0]      # dy es negativo (y baja con el índice)
    ix = np.rint((x - xs[0]) / dx).astype(int)
    iy = np.rint((y - ys[0]) / dy).astype(int)
    ok = vis & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    pad = POOL + 2
    x0, x1 = max(ix[ok].min() - pad, 0), min(ix[ok].max() + pad + 1, xs.size)
    y0, y1 = max(iy[ok].min() - pad, 0), min(iy[ok].max() + pad + 1, ys.size)

    cmi = f["CMI"]
    at = cmi.attrs
    sub = cmi[y0:y1, x0:x1]                    # lectura parcial del dataset
    uns = at.get("_Unsigned")
    if uns is not None and (uns.decode() if isinstance(uns, bytes) else str(uns)).lower() == "true" and sub.dtype.kind == "i":
        sub = sub.view(sub.dtype.str.replace("i", "u"))
    fill = np.asarray(at["_FillValue"]).ravel()[0].astype(sub.dtype) if "_FillValue" in at else None
    bad = (sub == fill) if fill is not None else np.zeros(sub.shape, bool)
    bt = sub.astype(np.float32) * np.float32(np.asarray(at["scale_factor"]).ravel()[0]) + np.float32(np.asarray(at["add_offset"]).ravel()[0])
    bt[bad] = np.nan
    # el mínimo ignora NaN reemplazándolos por un valor alto
    filt = minimum_filter(np.where(np.isnan(bt), 999.0, bt), size=POOL, mode="nearest")
    out = filt[np.clip(iy - y0, 0, filt.shape[0] - 1), np.clip(ix - x0, 0, filt.shape[1] - 1)]
    valid = ok & (out < 400) & (out > 150)
    q = np.where(valid, np.rint(out - BT_OFFSET), 0).clip(0, 255).astype(np.uint8)
    meta = {"lat0": LAT_N, "lon0": LON_W, "step": STEP, "ny": q.shape[0], "nx": q.shape[1], "offset": BT_OFFSET}
    return q, meta


def payload(q, meta, ts_iso):
    return {"v": 1, "sat": "GOES-19", "band": 13, "scan": ts_iso, **meta,
            "data": base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="archivo .nc local (prueba)")
    ap.add_argument("--out", help="escribir JSON en vez de enviar")
    a = ap.parse_args()

    if a.file:
        with open(a.file, "rb") as fh:
            raw = fh.read()
        ts = "1970-01-01T00:00:00Z"
        key = a.file
        m = re.search(r"_s\d{7}", key)
        if m:
            ts = scan_time(key).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        key = latest_key()
        if not key:
            log("no hay archivos recientes"); return 2
        st = scan_time(key)
        age = (datetime.now(timezone.utc) - st).total_seconds() / 60
        log("último:", key, f"edad {age:.0f} min")
        if age > MAX_AGE_MIN:
            log("demasiado viejo, no se envía"); return 3
        r = requests.get(f"{BUCKET}/{key}", timeout=120)
        r.raise_for_status()
        raw = r.content
        ts = st.strftime("%Y-%m-%dT%H:%M:%SZ")

    with h5py.File(io.BytesIO(raw), "r") as f:
        q, meta = build_grid(f)
    del raw
    body = payload(q, meta, ts)
    log(f"grilla {meta['ny']}x{meta['nx']}, cubierta {np.mean(q > 0)*100:.0f}%, {len(body['data'])//1024} KB")

    if a.out:
        json.dump(body, open(a.out, "w"))
        return 0
    url, k = os.environ.get("QSAT_INGEST_URL"), os.environ.get("QSAT_KEY")
    if not url or not k:
        log("faltan QSAT_INGEST_URL / QSAT_KEY"); return 1
    r = requests.post(url, json=body, headers={"x-qsat-key": k}, timeout=30)
    log("Worker:", r.status_code, r.text[:200])
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
