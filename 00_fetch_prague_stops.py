"""
00_fetch_prague_stops.py
========================

Завантажує GTFS-фід Праги (PID — Pražská integrovaná doprava),
парсить stops.txt, фільтрує міські зупинки і зберігає у форматі GeoJSON,
сумісному з рештою пайплайну (data/prague_stops.geojson).

Джерело: http://opendata.iprpraha.cz/DPP/JR/jrdata.zip
Ліцензія: CC-BY (відкриті дані, дозволено використовувати).

Запуск:
    python 00_fetch_prague_stops.py

Результат:
    data/prague_stops.geojson
"""

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

# Джерело GTFS Prague Integrated Transport
GTFS_URL = "http://opendata.iprpraha.cz/DPP/JR/jrdata.zip"

OUTPUT_PATH = Path(__file__).parent / "data" / "prague_stops.geojson"

# Фільтри для відсікання нерелевантних точок
# location_type: 0 = стандартна зупинка, 1 = станція, 2 = вхід/вихід, 3 = node, 4 = boarding area
ALLOWED_LOCATION_TYPES = {"0", ""}  # тільки реальні зупинки

# Зони PID: P (центр), 0, B, 1, 2, 3, 4 ... (передмістя)
# Беремо тільки центральну зону "P" і прилеглу "0" — це власне Прага
ALLOWED_ZONES = {"P", "0"}

# Ліміт зупинок (Прага має ~7000 зупинок, нам не треба всі для експерименту)
MAX_STOPS = 4500


def download_gtfs(url: str) -> bytes:
    """Завантажує zip-архів GTFS у пам'ять."""
    print(f"[00] Завантаження GTFS-фіду: {url}")
    print("     (~30-40 МБ, може зайняти 30-60 секунд)")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Research Project)"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"     Завантажено: {len(data) / 1024 / 1024:.1f} МБ")
    return data


def extract_stops(zip_bytes: bytes) -> list[dict]:
    """Витягає stops.txt із zip-архіву і парсить його."""
    print("[00] Розпаковка stops.txt ...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Шукаємо stops.txt (може бути в підпапці)
        stops_name = next((n for n in names if n.endswith("stops.txt")), None)
        if stops_name is None:
            raise RuntimeError(f"stops.txt не знайдено в архіві. Файли: {names[:10]}")
        
        with zf.open(stops_name) as f:
            text = f.read().decode("utf-8-sig")  # GTFS часто з BOM

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    print(f"     Усього записів у stops.txt: {len(rows):,}")
    return rows


def filter_and_dedupe(rows: list[dict]) -> list[dict]:
    """
    Фільтрує stops:
      - тільки реальні зупинки (location_type 0)
      - дедуплікація за округленими координатами
    """
    print("[00] Фільтрація і дедуплікація ...")

    filtered = []
    seen_coords = set()

    for r in rows:
        loc_type = r.get("location_type", "").strip()
        if loc_type not in ALLOWED_LOCATION_TYPES:
            continue

        try:
            lat = float(r["stop_lat"])
            lon = float(r["stop_lon"])
        except (ValueError, KeyError):
            continue

        # Фільтр меж Праги (приблизно): 49.94-50.20 lat, 14.20-14.70 lon
        # Виключаємо передмістя/міжміські з'єднання
        if not (49.94 <= lat <= 50.20 and 14.20 <= lon <= 14.70):
            continue

        # Дедупа за округленими координатами (5 знаків ≈ 1 м)
        key = (round(lat, 5), round(lon, 5))
        if key in seen_coords:
            continue
        seen_coords.add(key)

        filtered.append({
            "id": r.get("stop_id", "").strip(),
            "name": r.get("stop_name", "").strip().strip('"'),
            "lat": lat,
            "lon": lon,
            "zone": "",
        })

    print(f"     Після фільтра + дедупи: {len(filtered):,} зупинок")

    if len(filtered) > MAX_STOPS:
        import random
        random.seed(42)
        filtered = random.sample(filtered, MAX_STOPS)
        print(f"     Обмежено до {MAX_STOPS} зупинок (sample)")

    return filtered


def to_geojson(stops: list[dict]) -> dict:
    """Формує GeoJSON FeatureCollection."""
    features = []
    for s in stops:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [s["lon"], s["lat"]],  # GeoJSON: [lon, lat]
            },
            "properties": {
                "stop_id": s["id"],
                "name": s["name"],
                "zone": s["zone"],
            },
        })
    return {"type": "FeatureCollection", "features": features}


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    zip_bytes = download_gtfs(GTFS_URL)
    rows = extract_stops(zip_bytes)
    stops = filter_and_dedupe(rows)
    
    # Діагностика розкиду
    if len(stops) >= 2:
        lats = [s["lat"] for s in stops]
        lons = [s["lon"] for s in stops]
        print(f"     Lat діапазон: {min(lats):.4f} - {max(lats):.4f}")
        print(f"     Lon діапазон: {min(lons):.4f} - {max(lons):.4f}")
        # Прага: lat ~50.0-50.2, lon ~14.2-14.7
        # Розмір приблизно 30x30 км
    
    geojson = to_geojson(stops)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    print(f"[00] Збережено: {OUTPUT_PATH}")
    print(f"     Зупинок: {len(stops)}")


if __name__ == "__main__":
    main()
