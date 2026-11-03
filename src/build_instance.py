"""
build_instance.py — STEP 2: turn data/raw/* into the frozen model tables in
data/processed/:
  tracts.csv       tract_id, lat, lon, residents, jobs, transit      (set I)
  candidates.csv   cand_id, lat, lon, rent_psf_month, reusable_fitout, eligible   (set J)
  luckin.csv       store_id, lat, lon                                (set L)
  competitors.csv  store_id, lat, lon                                (set C)
  walk_dist.csv    tract_id, kind, store_id, dist_m   (all pairs <= 2,400 m)
Usage:  python src/build_instance.py [--walk grid|osmnx]
"""
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np, pandas as pd
from config import RAW, PROC
from geo import haversine, grid_walk

KEEP_M = 2400.0     # store pairs up to 1.5 x D_max so D_max can be stress-tested
SUBWAY_FILE = "subway_202608.csv"
MANHATTAN_BOUNDS = (40.68, 40.89, -74.03, -73.90)


def validate_locations(df, id_col, name):
    required = {id_col, "lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"{name} missing required columns: {sorted(required - set(df.columns))}")
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    south, north, west, east = MANHATTAN_BOUNDS
    valid = (
        df["lat"].between(south, north)
        & df["lon"].between(west, east)
        & df[id_col].notna()
    )
    invalid = int((~valid).sum())
    if invalid:
        raise ValueError(f"{name} contains {invalid} invalid IDs or coordinates")
    if df[id_col].astype(str).duplicated().any():
        raise ValueError(f"{name} contains duplicate {id_col} values")
    return df

def build_tracts():
    pts = pd.read_csv(RAW / "tract_points.csv", dtype={"tract_id": str})
    res = pd.read_csv(RAW / "residents.csv", dtype={"tract_id": str})
    job = pd.read_csv(RAW / "jobs.csv", dtype={"tract_id": str})
    sub = pd.read_csv(RAW / SUBWAY_FILE)
    if pts["tract_id"].duplicated().any() or res["tract_id"].duplicated().any():
        raise ValueError("Duplicate tract IDs in Gazetteer or ACS data")
    t = pts.merge(res, on="tract_id", how="left").merge(job, on="tract_id", how="left")
    t["jobs"] = t.jobs.fillna(0)          # LODES lists only blocks with jobs -> true zero
    if t.residents.isna().any():
        missing = t.loc[t.residents.isna(), "tract_id"].tolist()
        raise ValueError(f"ACS estimates missing for Manhattan tracts: {missing[:20]}")
    if (t[["residents", "jobs"]] < 0).any().any():
        raise ValueError("Negative resident or job counts found")
    sub = validate_locations(sub, "station_complex_id", "MTA station complexes")
    if sub["avg_daily_entries"].isna().any() or (sub["avg_daily_entries"] < 0).any():
        raise ValueError("MTA monthly station totals contain missing or negative values")
    # each station complex assigned ONCE, to the nearest tract representative point
    d = haversine(sub.lat.values[:, None], sub.lon.values[:, None], t.lat.values[None], t.lon.values[None])
    sub["tract_id"] = t.tract_id.values[d.argmin(1)]
    t = t.merge(sub.groupby("tract_id", as_index=False).avg_daily_entries.sum()
                .rename(columns={"avg_daily_entries": "transit"}), on="tract_id", how="left")
    t["transit"] = t.transit.fillna(0)
    t = t[(t.residents > 0) | (t.jobs > 0) | (t.transit > 0)]     # positive activity in >=1 input
    return t[["tract_id", "lat", "lon", "residents", "jobs", "transit"]]

def build_candidates():
    f = RAW / "candidates_team.csv"
    src = f if f.exists() else RAW / "storefronts_vacant_manhattan.csv"
    c = pd.read_csv(src, low_memory=False)
    if "cand_id" not in c: c["cand_id"] = [f"J{k+1:03d}" for k in range(len(c))]
    c = validate_locations(c, "cand_id", "Candidate sites")
    for col, default in [("rent_psf_month", np.nan), ("reusable_fitout", 0), ("eligible", 1)]:
        if col not in c: c[col] = default
    c["eligible"] = pd.to_numeric(c["eligible"], errors="coerce")
    if c["eligible"].isna().any() or not c["eligible"].isin([0, 1]).all():
        raise ValueError("Candidate eligible values must be 0 or 1")
    if c.empty:
        raise ValueError("Candidate site table is empty")
    print(f"candidates from {src.name}: {len(c)}")
    columns = ["cand_id", "lat", "lon", "rent_psf_month", "reusable_fitout", "eligible"]
    columns.extend(
        col for col in ("property_street_address_or", "zip_code", "borough_block_lot", "vacancy_as_of")
        if col in c.columns
    )
    return c[columns]

def build_stores():
    f = RAW / "luckin.csv"
    if not f.exists():
        raise FileNotFoundError(
            "Create data/raw/luckin.csv (store_id,lat,lon) from a verified store source"
        )
    lk = validate_locations(pd.read_csv(f), "store_id", "Luckin locations")
    cp = pd.read_csv(RAW / "competitors_dohmh.csv").rename(
        columns={"latitude": "lat", "longitude": "lon"}
    )
    cp["store_id"] = "C" + cp.camis.astype(str)
    cp = validate_locations(cp, "store_id", "Competitor locations")
    return lk, cp[["store_id", "lat", "lon"]]

def distances(t, stores, method):
    rows = []
    if method == "osmnx":
        import osmnx as ox, networkx as nx           # pip install osmnx scikit-learn
        G = ox.graph_from_place("Manhattan, New York, USA", network_type="walk")
        G = nx.Graph(G)                              # undirected for pedestrians
        tn = ox.distance.nearest_nodes(G, t.lon.values, t.lat.values)
        sn = ox.distance.nearest_nodes(G, stores.lon.values, stores.lat.values)
        node2stores = {}
        for k, n in enumerate(sn): node2stores.setdefault(n, []).append(k)
        for i, n in enumerate(tn):
            dist = nx.single_source_dijkstra_path_length(G, n, cutoff=KEEP_M, weight="length")
            for node, dm in dist.items():
                for k in node2stores.get(node, []):
                    rows.append((t.tract_id.iat[i], stores.kind.iat[k], stores.store_id.iat[k], dm))
        # stores with no route within KEEP_M are simply absent -> NOT zero distance
    else:
        d = grid_walk(t.lat.values[:, None], t.lon.values[:, None], stores.lat.values[None], stores.lon.values[None])
        ii, kk = np.nonzero(d <= KEEP_M)
        rows = list(zip(t.tract_id.values[ii], stores.kind.values[kk], stores.store_id.values[kk], d[ii, kk]))
    return pd.DataFrame(rows, columns=["tract_id", "kind", "store_id", "dist_m"])

def main(method, out=PROC):
    out.mkdir(parents=True, exist_ok=True)
    t = build_tracts(); c = build_candidates(); lk, cp = build_stores()
    if t.empty or lk.empty or cp.empty:
        raise ValueError("Tracts, incumbent stores, and competitors must all be nonempty")
    stores = pd.concat([c.rename(columns={"cand_id": "store_id"})[["store_id", "lat", "lon"]].assign(kind="cand"),
                        lk.assign(kind="luckin"), cp.assign(kind="comp")], ignore_index=True)
    w = distances(t, stores, method)
    if w.empty or w["dist_m"].isna().any() or (w["dist_m"] < 0).any():
        raise ValueError("No valid tract-to-store distances were generated")
    if w.duplicated(["tract_id", "kind", "store_id"]).any():
        raise ValueError("Duplicate tract/store distance rows were generated")
    for name, df in [("tracts", t), ("candidates", c), ("luckin", lk), ("competitors", cp), ("walk_dist", w)]:
        df.to_csv(out / f"{name}.csv", index=False)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "walk_method": method,
        "walk_distance_note": (
            "Approximate Manhattan-grid L1 distances; not pedestrian network routes."
            if method == "grid"
            else "OSM pedestrian-network distances."
        ),
        "keep_distance_m": KEEP_M,
        "rows": {
            "tracts": len(t),
            "candidates": len(c),
            "luckin": len(lk),
            "competitors": len(cp),
            "walk_dist": len(w),
        },
        "files": {
            f"{name}.csv": {
                "bytes": (out / f"{name}.csv").stat().st_size,
                "sha256": hashlib.sha256((out / f"{name}.csv").read_bytes()).hexdigest(),
            }
            for name in ("tracts", "candidates", "luckin", "competitors", "walk_dist")
        },
        "source_manifest": "data/raw/source_manifest.json",
    }
    (out / "processed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"|I|={len(t)} |J|={len(c)} |L|={len(lk)} |C|={len(cp)} pairs<=2.4km={len(w)} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--walk", default="grid", choices=["grid", "osmnx"])
    main(ap.parse_args().walk)
