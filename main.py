"""QSAT: lee GOES-19 y entrega al Worker grillas numéricas livianas sobre Argentina. No guarda imágenes.

  1) Banda 13 IR (temperatura de brillo)      -> respaldo "probable lluvia"
  2) Lluvia estimada RRQPE (mm/h, NOAA)       -> lluvia real estimada (producto Enterprise, con el viejo de respaldo)

Uso:
  python main.py                       # descarga lo último y lo manda al Worker
  python main.py --file x.nc --out g.json            # prueba local banda 13
  python main.py --rain-file x.nc --out r.json       # prueba local lluvia
  python main.py --inspect x.nc                      # lista variables/atributos de un .nc
Variables de entorno: QSAT_INGEST_URL, QSAT_KEY
"""
import argparse, base64, io, json, os, re, sys, time, zlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import requests
from scipy.ndimage import minimum_filter, maximum_filter

from geos import GeosProj

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPF"
CHANNEL = "C13"
# Lluvia: producto nuevo (Enterprise) y el viejo (se apaga pronto). El más reciente gana; a igualdad, el nuevo.
RAIN_SOURCES = [
    ("https://noaa-enterprise-rainrate-pds.s3.amazonaws.com", lambda t: f"G19/ABI-L2-FLDK-RRQPE-AWIPS/{t:%Y/%m/%d/%H}/"),
    (BUCKET, lambda t: f"ABI-L2-RRQPEF/{t.year}/{t.timetuple().tm_yday:03d}/{t:%H}/"),
]
RAIN_STEP_MMH = 0.5        # q = round(mm/h / 0.5), 0..255 => hasta 127 mm/h
RAIN_MAX_AGE_MIN = 40
# Grilla de salida (fila 0 = norte). 0.05° ≈ 5.5 km. Cubre toda Argentina continental + Tierra del Fuego.
LAT_N, LAT_S, LON_W, LON_E, STEP = -21.0, -56.0, -74.0, -53.0, 0.05
BT_OFFSET = 150.0          # q = round(BT_K - 150) -> 1 K de resolución, rango 150..405 K. 0 = sin dato
POOL = 3                   # ventana de mínimo (coge el tope más frío de ~6 km)
MAX_AGE_MIN = 40           # si el último archivo es más viejo, no se envía


def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def list_keys(prefix, bucket=BUCKET):
    r = requests.get(bucket, params={"list-type": "2", "prefix": prefix}, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"s": root.tag.split("}")[0].strip("{")}
    return [k.text for k in root.findall(".//s:Contents/s:Key", ns)]


def scan_time(key):
    """Hora de inicio del escaneo. Acepta _sAAAAJJJHHMMSSt (14 dígitos, día juliano) y _sAAAAMMDDHHMMSSt (15, Enterprise)."""
    d = re.search(r"_s(\d{14,15})", key).group(1)
    if len(d) == 15:
        y, mo, da, hh, mm, ss = int(d[0:4]), int(d[4:6]), int(d[6:8]), int(d[8:10]), int(d[10:12]), int(d[12:14])
        return datetime(y, mo, da, hh, mm, ss, tzinfo=timezone.utc)
    y, doy, hh, mm, ss = int(d[0:4]), int(d[4:7]), int(d[7:9]), int(d[9:11]), int(d[11:13])
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


def rain_candidates(now=None):
    """[(bucket, key)] de lluvia, el más reciente primero (a igual hora, el producto nuevo primero)."""
    now = now or datetime.now(timezone.utc)
    out = []
    for si, (bucket, pref) in enumerate(RAIN_SOURCES):
        for back in (0, 1):
            t = now - timedelta(hours=back)
            try:
                keys = [k for k in list_keys(pref(t), bucket) if k.endswith(".nc")]
            except Exception as e:
                log(f"aviso: no pude listar {bucket.split('//')[1].split('.')[0]} {pref(t)}: {e}")
                continue
            out += [(scan_time(k), -si, bucket, k) for k in keys]
            if keys:
                break
    out.sort(reverse=True)
    return [(b, k) for _, _, b, k in out]


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


def read_sub(ds, sl):
    """Igual que read_var pero sobre un recorte (lectura parcial)."""
    at = ds.attrs
    g = lambda n, d=None: (np.asarray(at[n]).ravel()[0] if n in at else d)
    sub = ds[sl]
    uns = at.get("_Unsigned")
    if uns is not None and (uns.decode() if isinstance(uns, bytes) else str(uns)).lower() == "true" and sub.dtype.kind == "i":
        sub = sub.view(sub.dtype.str.replace("i", "u"))
    fill = g("_FillValue")
    bad = (sub == np.array(fill).astype(sub.dtype)) if fill is not None else np.zeros(sub.shape, bool)
    a = sub.astype(np.float32) * np.float32(g("scale_factor", 1.0)) + np.float32(g("add_offset", 0.0))
    a[bad] = np.nan
    return a


def geo_index(f, pad):
    """Para cada celda de la grilla de salida: índices (iy, ix) en el archivo + recorte mínimo a leer."""
    pname = next((n for n in ("goes_imager_projection", "goes_lat_lon_projection") if n in f), None)
    if pname is None:
        pname = next((n for n in f.keys() if "grid_mapping_name" in f[n].attrs), None)
    if pname is None or "x" not in f or "y" not in f:
        raise RuntimeError("el archivo no trae proyección geoestacionaria + x/y. Variables: " + ", ".join(f.keys()))
    proj = GeosProj.from_attrs(f[pname].attrs)
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
    x0, x1 = max(ix[ok].min() - pad, 0), min(ix[ok].max() + pad + 1, xs.size)
    y0, y1 = max(iy[ok].min() - pad, 0), min(iy[ok].max() + pad + 1, ys.size)
    return ix, iy, ok, (y0, y1, x0, x1)


def build_grid(f):
    """Devuelve (q uint8 [ny,nx], meta) con BT cuantizada. No guarda nada en disco."""
    ix, iy, ok, (y0, y1, x0, x1) = geo_index(f, POOL + 2)
    bt = read_sub(f["CMI"], (slice(y0, y1), slice(x0, x1)))
    # el mínimo ignora NaN reemplazándolos por un valor alto
    filt = minimum_filter(np.where(np.isnan(bt), 999.0, bt), size=POOL, mode="nearest")
    out = filt[np.clip(iy - y0, 0, filt.shape[0] - 1), np.clip(ix - x0, 0, filt.shape[1] - 1)]
    valid = ok & (out < 400) & (out > 150)
    q = np.where(valid, np.rint(out - BT_OFFSET), 0).clip(0, 255).astype(np.uint8)
    meta = {"lat0": LAT_N, "lon0": LON_W, "step": STEP, "ny": q.shape[0], "nx": q.shape[1], "offset": BT_OFFSET}
    return q, meta


def build_rain_grid(f):
    """Lluvia estimada (mm/h) -> q uint8 con paso 0.5 mm/h. Toma el máximo de ~3x3 píxeles (picos de lluvia)."""
    name = "RRQPE" if "RRQPE" in f else None
    if name is None:
        cand = [n for n in f.keys() if f[n].ndim == 2 and n not in ("x", "y") and "DQF" not in n.upper() and f[n].shape[0] > 500]
        if len(cand) == 1:
            name = cand[0]
    if name is None:
        raise RuntimeError("no encuentro la variable de lluvia. Variables: " + ", ".join(f"{n}{f[n].shape}" for n in f.keys()))
    ix, iy, ok, (y0, y1, x0, x1) = geo_index(f, 4)
    r = read_sub(f[name], (slice(y0, y1), slice(x0, x1)))
    r = np.where(np.isfinite(r) & (r > 0) & (r < 500), r, 0.0)
    r = maximum_filter(r, size=3, mode="nearest")
    out = r[np.clip(iy - y0, 0, r.shape[0] - 1), np.clip(ix - x0, 0, r.shape[1] - 1)]
    q = np.where(ok, np.rint(out / RAIN_STEP_MMH), 0).clip(0, 255).astype(np.uint8)
    meta = {"lat0": LAT_N, "lon0": LON_W, "step": STEP, "ny": q.shape[0], "nx": q.shape[1], "scale": RAIN_STEP_MMH}
    return q, meta


def payload(q, meta, ts_iso, kind=None):
    body = {"v": 1, "sat": "GOES-19", "band": 13, "scan": ts_iso, **meta,
            "data": base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()}
    if kind:
        body.update({"kind": kind, "product": "RRQPE"})
        body.pop("band", None)
    return body


def post(body):
    url, k = os.environ.get("QSAT_INGEST_URL"), os.environ.get("QSAT_KEY")
    if not url or not k:
        log("faltan QSAT_INGEST_URL / QSAT_KEY"); return False
    r = requests.post(url, json=body, headers={"x-qsat-key": k}, timeout=30)
    log("Worker:", r.status_code, r.text[:200])
    return r.ok


def sat_rain():
    """(q, meta, scan_dt) del satélite probando candidatos; None si ninguno sirve."""
    for bucket, key in rain_candidates()[:4]:
        st = scan_time(key)
        age = (datetime.now(timezone.utc) - st).total_seconds() / 60
        src = "nuevo" if "enterprise" in bucket else "viejo"
        log(f"lluvia [{src}]:", key.split("/")[-1], f"edad {age:.0f} min")
        if age > RAIN_MAX_AGE_MIN:
            log("lluvia: demasiado vieja"); return None
        try:
            r = requests.get(f"{bucket}/{key}", timeout=120)
            r.raise_for_status()
            with h5py.File(io.BytesIO(r.content), "r") as f:
                q, meta = build_rain_grid(f)
            return q, meta, st
        except Exception as e:
            log(f"lluvia [{src}] falló: {e}")
    log("lluvia: ningún candidato sirvió")
    return None


def run_rain():
    """Lluvia = satélite (RRQPE) + radar SMN Bolívar donde alcanza (el radar manda). True si se envió."""
    sat = sat_rain()
    rad = None
    try:
        import radar
        got = radar.latest()
        if got:
            rt, png = got
            ny, nx = (sat[1]["ny"], sat[1]["nx"]) if sat else (round((LAT_N - LAT_S) / STEP), round((LON_E - LON_W) / STEP))
            mm, cov = radar.radar_grid(png, LAT_N, LON_W, STEP, ny, nx)
            rad = (np.clip(np.rint(mm / RAIN_STEP_MMH), 0, 255).astype(np.uint8), cov, rt)
            log(f"radar SMN {rt:%H:%M}Z: celdas con lluvia {int((mm > 0.5).sum())}, máx {mm.max():.1f} mm/h")
    except Exception as e:
        log(f"radar SMN falló (se sigue con satélite): {e}")
    if not sat and not rad:
        return False
    if sat:
        q, meta, st = sat
    else:
        ny, nx = rad[0].shape
        q = np.zeros((ny, nx), np.uint8)
        meta = {"lat0": LAT_N, "lon0": LON_W, "step": STEP, "ny": ny, "nx": nx, "scale": RAIN_STEP_MMH}
        st = rad[2]
    if rad:
        q = np.where(rad[1], rad[0], q)
        st = rad[2]
    body = payload(q, meta, st.strftime("%Y-%m-%dT%H:%M:%SZ"), kind="rain")
    if rad:
        body["product"] = "RRQPE+RMA14" if sat else "RMA14"
        import radar as _r
        body["radar"] = {"lat": _r.RADAR_LAT, "lon": _r.RADAR_LON, "rKm": _r.USE_KM}
    log(f"lluvia grilla {meta['ny']}x{meta['nx']}, con lluvia {np.mean(q > 0)*100:.2f}%, máx {q.max()*RAIN_STEP_MMH:.1f} mm/h, {len(body['data'])//1024} KB, fuente {body.get('product')}")
    return post(body)


def inspect(path):
    with h5py.File(path, "r") as f:
        def show(n, o):
            if isinstance(o, h5py.Dataset):
                print(n, o.shape, o.dtype, {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in list(o.attrs.items())[:8]})
        f.visititems(show)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="archivo .nc local (prueba banda 13)")
    ap.add_argument("--rain-file", help="archivo .nc local (prueba lluvia)")
    ap.add_argument("--inspect", help="listar contenido de un .nc")
    ap.add_argument("--out", help="escribir JSON en vez de enviar")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.inspect); return 0

    if a.rain_file:
        with h5py.File(a.rain_file, "r") as f:
            q, meta = build_rain_grid(f)
        ts = scan_time(a.rain_file).strftime("%Y-%m-%dT%H:%M:%SZ") if re.search(r"_s\d{14,15}", a.rain_file) else "1970-01-01T00:00:00Z"
        body = payload(q, meta, ts, kind="rain")
        log(f"lluvia grilla {meta['ny']}x{meta['nx']}, con lluvia {np.mean(q > 0)*100:.2f}%, máx {q.max()*RAIN_STEP_MMH:.1f} mm/h")
        if a.out:
            json.dump(body, open(a.out, "w"))
        return 0

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
    ok_ir = post(body)
    try:
        ok_rain = run_rain()
    except Exception as e:
        log("lluvia: error inesperado:", e); ok_rain = False
    if not ok_rain:
        log("AVISO: la lluvia no se envió (QSAT banda 13 sigue funcionando)")
    return 0 if ok_ir else 1


if __name__ == "__main__":
    sys.exit(main())
