"""geo.py — distance helpers (metres)."""
import numpy as np

R_EARTH = 6_371_008.8
LAT0, LON0 = 40.7831, -73.9712          # Manhattan centre, for local projection
GRID_ROT_DEG = 29.0                      # Manhattan street grid is ~29 deg east of true north

def haversine(lat1, lon1, lat2, lon2):
    """Geodesic (great-circle) distance in metres; broadcasts. Used for SPACING rules (delta)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

def to_xy(lat, lon):
    """Local east/north metres (equirectangular, accurate to <0.1% over Manhattan)."""
    x = np.radians(np.asarray(lon) - LON0) * R_EARTH * np.cos(np.radians(LAT0))
    y = np.radians(np.asarray(lat) - LAT0) * R_EARTH
    return x, y

def grid_walk(lat1, lon1, lat2, lon2):
    """FALLBACK walking distance: L1 (city-block) distance on the rotated Manhattan grid.
    Use when an OSM pedestrian network is not available; document it as an assumption."""
    x1, y1 = to_xy(lat1, lon1); x2, y2 = to_xy(lat2, lon2)
    t = np.radians(GRID_ROT_DEG)
    dx, dy = x2 - x1, y2 - y1
    u = dx * np.cos(t) - dy * np.sin(t)       # cross-town (street) direction
    v = dx * np.sin(t) + dy * np.cos(t)       # up-town (avenue) direction
    return np.abs(u) + np.abs(v)
