"""Proyección geoestacionaria GOES-R (PUG vol. 3): lat/lon <-> ángulos de escaneo (x, y)."""
import numpy as np


class GeosProj:
    def __init__(self, h_sat=35786023.0, r_eq=6378137.0, r_pol=6356752.31414, lon0=-75.2):
        self.H = h_sat + r_eq
        self.req = r_eq
        self.rpol = r_pol
        self.lon0 = np.deg2rad(lon0)
        self.k = (r_eq / r_pol) ** 2          # r_eq² / r_pol²
        self.e2 = 1.0 - 1.0 / self.k          # excentricidad²

    @classmethod
    def from_attrs(cls, attrs):
        g = lambda n: float(np.asarray(attrs[n]).ravel()[0])
        return cls(g("perspective_point_height"), g("semi_major_axis"), g("semi_minor_axis"),
                   g("longitude_of_projection_origin"))

    def latlon_to_scan(self, lat, lon):
        """lat/lon en grados -> (x, y, visible). x/y en radianes."""
        lat = np.deg2rad(np.asarray(lat, dtype=float))
        lon = np.deg2rad(np.asarray(lon, dtype=float))
        phic = np.arctan(np.tan(lat) / self.k)
        rc = self.rpol / np.sqrt(1.0 - self.e2 * np.cos(phic) ** 2)
        sx = self.H - rc * np.cos(phic) * np.cos(lon - self.lon0)
        sy = -rc * np.cos(phic) * np.sin(lon - self.lon0)
        sz = rc * np.sin(phic)
        visible = self.H * (self.H - sx) >= sy * sy + self.k * sz * sz
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz))
        return x, y, visible

    def scan_to_latlon(self, x, y):
        """x/y en radianes -> (lat, lon, valid) en grados. valid=False si el rayo no toca la Tierra."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, sxn, cy, syn = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
        a = sxn ** 2 + cx ** 2 * (cy ** 2 + self.k * syn ** 2)
        b = -2.0 * self.H * cx * cy
        c = self.H ** 2 - self.req ** 2
        disc = b * b - 4.0 * a * c
        valid = disc >= 0
        rs = (-b - np.sqrt(np.where(valid, disc, 0.0))) / (2.0 * a)
        sx = rs * cx * cy
        sy = -rs * sxn
        sz = rs * cx * syn
        lat = np.arctan(self.k * sz / np.sqrt((self.H - sx) ** 2 + sy * sy))
        lon = self.lon0 - np.arctan(sy / (self.H - sx))
        return np.rad2deg(lat), np.rad2deg(lon), valid
