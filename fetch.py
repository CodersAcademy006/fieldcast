"""Pull research-grade iNaturalist observations near Pune and daily weather from Open-Meteo.
Output: data/obs.parquet (one row per observation) and data/weather.parquet (one row per day)."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

LAT, LNG, RADIUS_KM = 18.52, 73.86, 30
START, END = "2019-01-01", "2026-10-04"
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def observations(taxon_id, name):
    ckpt = OUT / f"ckpt_{name}.jsonl"  # one line per page, so a crashed run resumes at the last full page
    rows = []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                rows += [tuple(r) for r in json.loads(line)]
            except ValueError:
                pass  # half-written last line from a killed process
    id_above = max((r[0] for r in rows), default=0)
    while True:
        r = requests.get("https://api.inaturalist.org/v1/observations", timeout=60, params={
            "lat": LAT, "lng": LNG, "radius": RADIUS_KM, "quality_grade": "research", "taxon_id": taxon_id,
            "d1": START, "d2": END, "per_page": 200, "order_by": "id", "order": "asc", "id_above": id_above})
        r.raise_for_status()
        res = r.json()["results"]
        if not res:
            break
        page = [(o["id"], o["observed_on"], o["taxon"]["name"] if o.get("taxon") else None,
                 o["taxon"].get("preferred_common_name") if o.get("taxon") else None, name) for o in res if o.get("observed_on")]
        with ckpt.open("a") as f:
            f.write(json.dumps(page) + "\n")
        rows += page
        id_above = res[-1]["id"]
        print(f"{name}: {len(rows)}", flush=True)
        time.sleep(1)  # iNaturalist asks for at most ~1 request per second
    return rows


def weather():
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
        "latitude": LAT, "longitude": LNG, "start_date": START, "end_date": END, "timezone": "Asia/Kolkata",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max,shortwave_radiation_sum"})
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    # 47224 = Papilionoidea (butterflies), 3 = Aves
    rows = observations(47224, "butterfly") + observations(3, "bird")
    obs = pd.DataFrame(rows, columns=["id", "date", "sci", "common", "group"])
    obs["date"] = pd.to_datetime(obs["date"])
    obs.to_parquet(OUT / "obs.parquet", index=False)
    weather().to_parquet(OUT / "weather.parquet", index=False)
    print(obs.groupby("group").size())
