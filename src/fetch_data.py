"""Download and clean the public inputs for the Manhattan site-selection model."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import BASE, RAW

STATE = "36"
COUNTY = "061"
ACS_YEAR = 2024
LODES_YEAR = 2023
MTA_MONTH = "2026-08"
VACANCY_YEAR = "2025"
NYC = "https://data.cityofnewyork.us/resource"
NYS = "https://data.ny.gov/resource"
MANHATTAN_BOUNDS = (40.68, 40.89, -74.03, -73.90)
TIMEOUT = (20, 300)
RAW.mkdir(parents=True, exist_ok=True)

CHAINS = (
    "STARBUCKS",
    "DUNKIN",
    "BLUESTONE",
    "JOE COFFEE",
    "BLUE BOTTLE",
    "PRET",
    "GREGORYS",
    "TIM HORTONS",
    "COFFEE PROJECT",
    "MAMAN",
    "VARIETY COFFEE",
    "BLANK STREET",
    "COTTI",
    "LUCKIN",
)


def _download(url: str, destination: Path) -> dict:
    """Download a source file atomically and return its provenance."""
    digest = hashlib.sha256()
    byte_count = 0
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temp.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
    if byte_count == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded an empty file from {url}")
    temp_path.replace(destination)
    return {
        "url": url,
        "file": destination.name,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _socrata_rows(dataset_id: str, params: dict, base: str = NYC) -> pd.DataFrame:
    """Read every matching SODA row with stable offset pagination."""
    rows = []
    offset = 0
    page_size = 50_000
    while True:
        query = dict(params)
        query.update({"$limit": page_size, "$offset": offset})
        response = requests.get(
            f"{base}/{dataset_id}.json", params=query, timeout=TIMEOUT
        )
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError(f"Unexpected Socrata response for {dataset_id}: {page}")
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    if not rows:
        raise ValueError(f"No records returned for Socrata dataset {dataset_id}")
    return pd.DataFrame.from_records(rows)


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = RAW / name
    frame.to_csv(path, index=False)
    return path


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _coordinate_mask(frame: pd.DataFrame, lat: str, lon: str) -> pd.Series:
    south, north, west, east = MANHATTAN_BOUNDS
    return (
        frame[lat].between(south, north)
        & frame[lon].between(west, east)
    )


def _read_tract_points(sources: dict) -> pd.DataFrame:
    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2024_Gazetteer/2024_Gaz_tracts_national.zip"
    )
    info = _download(url, RAW / "gazetteer_2024_tracts_national.zip")
    sources["census_gazetteer"] = info
    with zipfile.ZipFile(RAW / info["file"]) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Unexpected Gazetteer ZIP contents: {archive.namelist()}")
        frame = pd.read_csv(
            io.BytesIO(archive.read(members[0])),
            sep="\t",
            dtype={"GEOID": "string"},
        )
    frame.columns = frame.columns.str.strip()
    required = {"GEOID", "INTPTLAT", "INTPTLONG", "ALAND"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Gazetteer schema changed; expected {required}, got {list(frame.columns)}")
    frame = frame.loc[frame["GEOID"].str.startswith(STATE + COUNTY)].copy()
    frame = frame.rename(
        columns={"GEOID": "tract_id", "INTPTLAT": "lat", "INTPTLONG": "lon"}
    )
    frame = frame[["tract_id", "lat", "lon", "ALAND"]]
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    if frame.empty or frame["tract_id"].duplicated().any() or frame[["lat", "lon"]].isna().any().any():
        raise ValueError("Invalid or duplicate Manhattan tract points in Census Gazetteer")
    _write_csv(frame, "tract_points.csv")
    return frame


def _read_residents(sources: dict) -> pd.DataFrame:
    url = (
        "https://www2.census.gov/programs-surveys/acs/summary_file/2024/"
        "table-based-SF/data/5YRData/acsdt5y2024-b01003.dat"
    )
    info = _download(url, RAW / "acs_2024_b01003.dat")
    sources["acs_2024_b01003"] = info
    data = pd.read_csv(
        RAW / info["file"],
        sep="|",
        dtype={"GEO_ID": "string"},
        usecols=["GEO_ID", "B01003_E001", "B01003_M001"],
    )
    tract_prefix = f"1400000US{STATE}{COUNTY}"
    data = data.loc[data["GEO_ID"].str.startswith(tract_prefix)].copy()
    data["tract_id"] = data["GEO_ID"].str[-11:]
    data["residents"] = pd.to_numeric(data["B01003_E001"], errors="coerce")
    data["residents_moe"] = pd.to_numeric(data["B01003_M001"], errors="coerce")
    result = data[["tract_id", "residents", "residents_moe"]]
    if (
        result.empty
        or result["tract_id"].duplicated().any()
        or result["residents"].isna().any()
        or (result["residents"] < 0).any()
    ):
        raise ValueError("ACS B01003 has missing, invalid, or duplicate Manhattan tract estimates")
    _write_csv(result, "residents.csv")
    return result


def _read_jobs(sources: dict) -> pd.DataFrame:
    url = (
        f"https://lehd.ces.census.gov/data/lodes/LODES8/ny/wac/"
        f"ny_wac_S000_JT01_{LODES_YEAR}.csv.gz"
    )
    info = _download(url, RAW / f"ny_wac_S000_JT01_{LODES_YEAR}.csv.gz")
    sources["lodes_wac_jt01"] = info
    blocks = pd.read_csv(
        RAW / info["file"],
        compression="gzip",
        dtype={"w_geocode": "string"},
        usecols=["w_geocode", "C000"],
    )
    blocks = blocks.loc[blocks["w_geocode"].str.startswith(STATE + COUNTY)].copy()
    if blocks.empty or blocks["w_geocode"].str.len().ne(15).any():
        raise ValueError("LODES returned no valid New York County block records")
    blocks["C000"] = pd.to_numeric(blocks["C000"], errors="coerce")
    if blocks["C000"].isna().any() or (blocks["C000"] < 0).any():
        raise ValueError("LODES C000 contains missing or negative job counts")
    blocks["tract_id"] = blocks["w_geocode"].str[:11]
    result = (
        blocks.groupby("tract_id", as_index=False)["C000"]
        .sum()
        .rename(columns={"C000": "jobs"})
    )
    _write_csv(result, "jobs.csv")
    return result


def _read_subway(sources: dict) -> pd.DataFrame:
    year, month = map(int, MTA_MONTH.split("-"))
    start = f"{MTA_MONTH}-01T00:00:00"
    if month == 12:
        end = f"{year + 1}-01-01T00:00:00"
    else:
        end = f"{year}-{month + 1:02d}-01T00:00:00"
    days_in_month = (pd.Timestamp(end) - pd.Timestamp(start)).days
    daily_frames = []
    first_day = pd.Timestamp(start)
    last_day = pd.Timestamp(end)
    for day in pd.date_range(first_day, last_day - pd.Timedelta(days=1), freq="D"):
        day_start = day.strftime("%Y-%m-%dT00:00:00")
        day_end = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
        where = (
            "borough='Manhattan' AND transit_mode='subway' AND "
            f"transit_timestamp >= '{day_start}' AND transit_timestamp < '{day_end}'"
        )
        params = {
            "$select": (
                "station_complex_id,station_complex,avg(latitude) as latitude,"
                "avg(longitude) as longitude,sum(ridership) as rides"
            ),
            "$where": where,
            "$group": "station_complex_id,station_complex",
            "$order": "station_complex_id",
        }
        frame = _socrata_rows("5wq4-mkjj", params, base=NYS)
        frame["service_date"] = day
        daily_frames.append(frame)
        print(f"MTA daily station totals: {day.date()} ({len(frame)} complexes)")
    daily = pd.concat(daily_frames, ignore_index=True)
    for column in ("latitude", "longitude", "rides"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["service_date"] = pd.to_datetime(daily["service_date"], errors="coerce")
    if daily[["latitude", "longitude", "rides", "service_date"]].isna().any().any():
        raise ValueError("MTA daily aggregate contains missing dates, coordinates, or ridership")
    if (daily["rides"] < 0).any() or not _coordinate_mask(daily, "latitude", "longitude").all():
        raise ValueError("MTA subway result contains invalid counts or non-Manhattan coordinates")
    _write_csv(daily, f"mta_station_daily_{MTA_MONTH}.csv")

    daily["service_day"] = daily["service_date"].dt.date
    monthly = (
        daily.groupby("station_complex_id", as_index=False)
        .agg(
            station=("station_complex", "first"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            rides=("rides", "mean"),
            observed_days=("service_day", "nunique"),
        )
        .rename(columns={"rides": "avg_daily_entries"})
    )
    if monthly.empty or (monthly["observed_days"] < days_in_month - 3).any():
        low_coverage = monthly.loc[
            monthly["observed_days"] < days_in_month - 3,
            ["station_complex_id", "observed_days"],
        ].to_dict("records")
        raise ValueError(f"Incomplete MTA monthly station coverage: {low_coverage[:10]}")
    _write_csv(monthly, f"subway_{MTA_MONTH.replace('-', '')}.csv")
    sources["mta_subway_ridership"] = {
        "dataset_id": "5wq4-mkjj",
        "url": f"{NYS}/5wq4-mkjj.json",
        "where": (
            "borough='Manhattan' AND transit_mode='subway' AND "
            f"transit_timestamp in [{start}, {end})"
        ),
        "aggregation": (
            "31 daily Socrata queries; sum ridership by station complex per day, "
            "then mean over observed dates"
        ),
        "daily_file": f"mta_station_daily_{MTA_MONTH}.csv",
        "daily_rows": len(daily),
        "stations": len(monthly),
        "min_days_per_station": int(monthly["observed_days"].min()),
        "month_days": days_in_month,
    }
    return monthly


def _read_cafes(sources: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    chain_filter = " OR ".join(f"upper(dba) like '%{chain}%'" for chain in CHAINS)
    where = f"boro='Manhattan' AND (cuisine_description='Coffee/Tea' OR {chain_filter})"
    params = {
        "$select": (
            "camis,dba,boro,building,street,zipcode,cuisine_description,"
            "latitude,longitude,inspection_date,record_date"
        ),
        "$where": where,
        "$order": "camis,inspection_date,record_date",
    }
    data = _socrata_rows("43nn-pn8j", params)
    _write_csv(data, "dohmh_cafes_raw.csv")
    sources["dohmh_restaurant_inspections"] = {
        "dataset_id": "43nn-pn8j",
        "url": f"{NYC}/43nn-pn8j.json",
        "where": where,
        "raw_rows": len(data),
    }
    for column in ("latitude", "longitude"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data[["latitude", "longitude"]].notna().all(axis=1)
    valid &= _coordinate_mask(data, "latitude", "longitude")
    clean = data.loc[valid].copy()
    clean["inspection_date"] = pd.to_datetime(clean["inspection_date"], errors="coerce")
    clean["record_date"] = pd.to_datetime(clean["record_date"], errors="coerce")
    clean = clean.sort_values(
        ["record_date", "inspection_date"], na_position="first"
    ).drop_duplicates("camis", keep="last")
    clean["dba"] = clean["dba"].fillna("").astype(str)
    luckin_mask = clean["dba"].str.contains("LUCKIN", case=False, na=False)
    luckin_source = clean.loc[luckin_mask].copy()
    competitors = clean.loc[~luckin_mask].copy()
    if luckin_source.empty or competitors.empty:
        raise ValueError("DOHMH did not return both Luckin and competitor coffee businesses")
    _write_csv(competitors, "competitors_dohmh.csv")
    _write_csv(luckin_source, "luckin_from_dohmh.csv")

    # DOHMH is the only accessible public source of Luckin locations in this run.
    # Preserve the permit IDs and source file separately; this list is not proof
    # that each establishment is still trading.
    luckin = luckin_source.rename(columns={"camis": "store_id"})[
        ["store_id", "latitude", "longitude"]
    ].rename(columns={"latitude": "lat", "longitude": "lon"})
    _write_csv(luckin, "luckin.csv")
    sources["store_status_note"] = (
        "Luckin locations are unique DOHMH inspection/establishment records; current operating "
        "status was not independently confirmed from the chain locator."
    )
    return competitors, luckin


def _read_vacancies(sources: dict) -> pd.DataFrame:
    where = f"borough='MANHATTAN' AND reporting_year='{VACANCY_YEAR}'"
    params = {
        "$select": (
            "filing_due_date,reporting_year,borough_block_lot,property_street_address_or,"
            "borough,zip_code,vacant_on_12_31,construction_reported,"
            "vacant_6_30_or_date_sold,primary_business_activity,"
            "expir_dt_of_most_recent_lease,property_number,property_street,unit,"
            "latitude,longitude,census_tract,bin,bbl,nta,nbhd"
        ),
        "$where": where,
        "$order": "borough_block_lot,property_street_address_or,unit",
    }
    source = _socrata_rows("92iy-9c3n", params)
    _write_csv(source, f"storefronts_manhattan_{VACANCY_YEAR}_raw.csv")
    sources["nyc_dof_storefronts"] = {
        "dataset_id": "92iy-9c3n",
        "url": f"{NYC}/92iy-9c3n.json",
        "where": where,
        "raw_rows": len(source),
        "source_description": (
            "DOF storefront registrations; includes both vacant and non-vacant storefronts."
        ),
    }
    expected = {"vacant_on_12_31", "vacant_6_30_or_date_sold", "construction_reported"}
    if not expected.issubset(source.columns):
        raise ValueError(f"DOF vacancy schema changed; expected {expected}")

    year_end = source["vacant_on_12_31"].fillna("").str.upper().str.strip()
    later = source["vacant_6_30_or_date_sold"].fillna("").str.upper().str.strip()
    construction = source["construction_reported"].fillna("").str.upper().str.strip()
    vacant = later.eq("YES") | (later.eq("") & year_end.eq("YES"))
    not_under_construction = ~construction.eq("YES")
    clean = source.loc[vacant & not_under_construction].copy()
    clean["lat"] = pd.to_numeric(clean["latitude"], errors="coerce")
    clean["lon"] = pd.to_numeric(clean["longitude"], errors="coerce")
    clean = clean.loc[clean[["lat", "lon"]].notna().all(axis=1)].copy()
    clean = clean.loc[_coordinate_mask(clean, "lat", "lon")].copy()
    clean["vacancy_as_of"] = clean["vacant_6_30_or_date_sold"].map(
        lambda value: (
            "2026-06-30_or_sale_date"
            if str(value).strip().upper() == "YES"
            else f"{VACANCY_YEAR}-12-31_no_later_status_reported"
        )
    )
    clean = clean.sort_values(
        ["borough_block_lot", "property_street_address_or", "unit", "lat", "lon"],
        na_position="last",
    )
    # The source provides building/storefront coordinates, not separate door
    # coordinates. Collapse exact geocoding duplicates before using them as sites.
    clean["coordinate_key"] = clean["lat"].round(6).astype(str) + "," + clean["lon"].round(6).astype(str)
    clean = clean.drop_duplicates("coordinate_key", keep="first").drop(columns="coordinate_key")
    clean = clean.reset_index(drop=True)
    clean.insert(0, "cand_id", [f"VAC{k:05d}" for k in range(1, len(clean) + 1)])
    clean["rent_psf_month"] = math.nan
    clean["reusable_fitout"] = 0
    clean["eligible"] = 1
    if clean.empty or clean["cand_id"].duplicated().any():
        raise ValueError("No valid, unique Manhattan vacant storefront candidates were produced")
    _write_csv(clean, "storefronts_vacant_manhattan.csv")
    sources["nyc_dof_storefronts"].update(
        {
            "clean_candidates": len(clean),
            "vacancy_rule": (
                "vacant_6_30_or_date_sold=YES, or blank with vacant_on_12_31=YES; "
                "exclude construction_reported=YES; valid Manhattan coordinates; "
                "collapse rounded 6-decimal coordinate duplicates"
            ),
        }
    )
    return clean


def main() -> None:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    sources = {}
    tracts = _read_tract_points(sources)
    residents = _read_residents(sources)
    jobs = _read_jobs(sources)
    subway = _read_subway(sources)
    competitors, luckin = _read_cafes(sources)
    candidates = _read_vacancies(sources)

    tract_ids = set(tracts["tract_id"])
    resident_ids = set(residents["tract_id"])
    if tract_ids != resident_ids:
        raise ValueError(
            "ACS/Gazetteer tract mismatch: "
            f"missing estimates={sorted(tract_ids - resident_ids)[:10]}, "
            f"unexpected estimates={sorted(resident_ids - tract_ids)[:10]}"
        )
    if not set(jobs["tract_id"]).issubset(tract_ids):
        raise ValueError("LODES includes tract IDs outside Manhattan Gazetteer coverage")

    manifest = {
        "retrieved_at_utc": retrieved_at,
        "study_area": "New York County (Manhattan), NY",
        "data_vintages": {
            "acs": f"{ACS_YEAR} ACS 5-year estimates (2020-2024), table B01003",
            "lodes": f"LODES 8 WAC JT01, {LODES_YEAR}",
            "mta": f"NY MTA Subway Hourly Ridership, {MTA_MONTH}",
            "vacancies": (
                f"NYC DOF reporting year {VACANCY_YEAR}; later available 2026 status "
                "is used where reported"
            ),
        },
        "sources": sources,
        "files": {
            path.name: _fingerprint(path)
            for path in sorted(RAW.iterdir())
            if path.is_file() and path.name != "source_manifest.json"
        },
        "cleaned_row_counts": {
            "tract_points": len(tracts),
            "residents": len(residents),
            "tracts_with_lodes_jobs": len(jobs),
            "subway_station_complexes": len(subway),
            "competitor_establishments": len(competitors),
            "luckin_establishments": len(luckin),
            "vacant_storefront_candidates": len(candidates),
        },
        "quality_findings": {
            "candidate_rent_quotes_missing": int(candidates["rent_psf_month"].isna().sum()),
            "candidate_rent_fallback_assumption_psf_month": BASE.rent_benchmark_psf_month,
            "candidate_fitout_flags_assumed_zero": int(
                candidates["reusable_fitout"].eq(0).sum()
            ),
            "luckin_operating_status_unverified": len(luckin),
            "competitor_operating_status_unverified": len(competitors),
        },
        "limitations": [
            "DOHMH inspection records do not independently establish that an establishment is currently open.",
            "Luckin locations are provisional DOHMH records; the official chain locator was not reachable during retrieval.",
            "DOF coordinates are storefront/building-level geocodes, not verified entrance coordinates.",
            "DOF source does not provide asking rent or reusable fit-out status; the model's $9/sf/month rent fallback and fit-out flag are assumptions, not measurements.",
            "LODES workplace jobs and subway entries are demand proxies, not observed coffee purchases.",
        ],
    }
    manifest_path = RAW / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["cleaned_row_counts"], indent=2))
    print(f"Raw source files and manifest saved under {RAW}")


if __name__ == "__main__":
    main()
