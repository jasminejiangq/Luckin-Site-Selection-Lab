"""
synthetic.py — a SYNTHETIC Manhattan-shaped instance with exactly the same
schema as data/processed/, so the whole pipeline can be tested before the
real data are frozen.  Numbers are invented: NEVER report them as results.
Usage:  python src/synthetic.py  -> data/synthetic/
"""
import numpy as np, pandas as pd
from config import ROOT
from geo import grid_walk, R_EARTH, LAT0, LON0, GRID_ROT_DEG
from build_instance import KEEP_M

OUT = ROOT / "data" / "synthetic"
ORIGIN_LAT, ORIGIN_LON = 40.7033, -74.0170       # the Battery

def to_latlon(s, c):
    """s = metres up the island axis, c = metres across it."""
    t = np.radians(GRID_ROT_DEG)
    e = s * np.sin(t) + c * np.cos(t); n = s * np.cos(t) - c * np.sin(t)
    lat = ORIGIN_LAT + np.degrees(n / R_EARTH)
    lon = ORIGIN_LON + np.degrees(e / (R_EARTH * np.cos(np.radians(LAT0))))
    return lat, lon

def width(s):
    return np.interp(s, [0, 1500, 3500, 16000, 19000, 21000], [1200, 2600, 3300, 3000, 1800, 1200])

def inside(s, c):
    return (s >= 0) & (s <= 21000) & (np.abs(c) <= width(s) / 2)

def sample(n, weight_fn, jitter, rng):
    """Rejection-sample n points with density ~ weight_fn(s), jittered, kept inside the island."""
    S, Cc = [], []
    wmax = weight_fn(np.linspace(0, 21000, 400)).max()
    while len(S) < n:
        s = rng.uniform(0, 21000, 4 * n); c = rng.uniform(-1700, 1700, 4 * n)
        s2 = s + rng.normal(0, jitter, 4 * n); c2 = c + rng.normal(0, jitter, 4 * n)
        ok = inside(s, c) & inside(s2, c2) & (rng.uniform(0, 1, 4 * n) < weight_fn(s) / wmax)
        S += list(s2[ok]); Cc += list(c2[ok])
    return np.array(S[:n]), np.array(Cc[:n])

def main(seed=4004):
    rng = np.random.default_rng(seed); OUT.mkdir(parents=True, exist_ok=True)
    jobs_f = lambda s: 40000 * np.exp(-((s - 6500) / 1500) ** 2) + 30000 * np.exp(-((s - 1000) / 800) ** 2) + 1500
    res_f = lambda s: 3000 + 3500 * np.exp(-((s - 12000) / 3000) ** 2) + 1500 * (s > 16000)
    # tracts on a ~440 m lattice
    ss, cc = np.meshgrid(np.arange(200, 21000, 440), np.arange(-1650, 1700, 440))
    ss, cc = ss.ravel(), cc.ravel(); k = inside(ss, cc); ss, cc = ss[k], cc[k]
    n = len(ss); lat, lon = to_latlon(ss, cc)
    residents = np.round(res_f(ss) * rng.lognormal(0, 0.45, n) * (rng.uniform(0, 1, n) > 0.04))
    jobs = np.round(jobs_f(ss) * rng.lognormal(0, 0.8, n) / 3)
    has_st = rng.uniform(0, 1, n) < 0.45
    transit = np.round(has_st * (residents / 3 + jobs / 2) * rng.lognormal(0, 0.6, n))
    tr = pd.DataFrame(dict(tract_id=[f"T{i+1:03d}" for i in range(n)], lat=lat, lon=lon,
                           residents=residents, jobs=jobs, transit=transit))
    tr = tr[(tr.residents > 0) | (tr.jobs > 0) | (tr.transit > 0)]
    act = lambda s: jobs_f(s) / 3 + res_f(s)
    s, c = sample(700, act, 120, rng); la, lo = to_latlon(s, c)
    comp = pd.DataFrame(dict(store_id=[f"C{i+1:04d}" for i in range(len(s))], lat=la, lon=lo))
    s, c = sample(14, jobs_f, 300, rng); la, lo = to_latlon(s, c)
    lk = pd.DataFrame(dict(store_id=[f"L{i+1:02d}" for i in range(len(s))], lat=la, lon=lo))
    s, c = sample(900, act, 200, rng); la, lo = to_latlon(s, c); m = len(s)
    rent = 6 + 7 * np.exp(-((s - 6500) / 2500) ** 2) + 3 * np.exp(-((s - 1000) / 1200) ** 2) + rng.normal(0, 1.2, m)
    rent = np.where(rng.uniform(0, 1, m) < 0.2, np.nan, np.clip(rent, 3.5, None).round(2))
    cand = pd.DataFrame(dict(cand_id=[f"J{i+1:03d}" for i in range(m)], lat=la, lon=lo, rent_psf_month=rent,
                             reusable_fitout=(rng.uniform(0, 1, m) < 0.25).astype(int),
                             eligible=(rng.uniform(0, 1, m) > 0.04).astype(int)))
    stores = pd.concat([cand.rename(columns={"cand_id": "store_id"})[["store_id", "lat", "lon"]].assign(kind="cand"),
                        lk.assign(kind="luckin"), comp.assign(kind="comp")], ignore_index=True)
    d = grid_walk(tr.lat.values[:, None], tr.lon.values[:, None], stores.lat.values[None], stores.lon.values[None])
    ii, kk = np.nonzero(d <= KEEP_M)
    wd = pd.DataFrame(dict(tract_id=tr.tract_id.values[ii], kind=stores.kind.values[kk],
                           store_id=stores.store_id.values[kk], dist_m=d[ii, kk]))
    for name, df in [("tracts", tr), ("candidates", cand), ("luckin", lk), ("competitors", comp), ("walk_dist", wd)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"SYNTHETIC |I|={len(tr)} |J|={len(cand)} |L|={len(lk)} |C|={len(comp)} pairs={len(wd)} -> {OUT}")

if __name__ == "__main__":
    main()
