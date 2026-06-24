# Input data

The two input datasets are **not** committed to this repository (one is ~90 MB,
both are openly licensed and easy to re-fetch). Place them here before running
the pipeline.

```
data/
├── data.csv             # NY Bus Breakdown & Delays  (trains the BBN)
└── prague_stops.geojson # Prague public-transport stops (routing graph)
```

## 1. `data.csv` — NY Bus Breakdown and Delays

* Source: NYC Open Data, dataset `ez4e-fazm`
  (<https://data.cityofnewyork.us/Transportation/Bus-Breakdown-and-Delays/ez4e-fazm>)
* License: public domain (NYC Open Data terms of use)
* ~379k records of school-bus breakdowns/delays, 2015-2019.

Download the full CSV from the portal ("Export → CSV") and save it as
`data/data.csv`. The pipeline only reads a handful of columns
(`Reason`, `Breakdown_or_Running_Late`, `Occurred_On`,
`Number_Of_Students_On_The_Bus`, `Boro`), so any recent export works.

## 2. `prague_stops.geojson` — Prague stops

* Source: Prague Integrated Transport (PID) GTFS feed
  (<http://opendata.iprpraha.cz/DPP/JR/jrdata.zip>)
* License: CC-BY (Prague open data)

You do **not** have to download this by hand — `00_fetch_prague_stops.py`
fetches the GTFS feed, extracts `stops.txt`, filters to the central zone and
writes `data/prague_stops.geojson` in the format the rest of the pipeline
expects:

```bash
python 00_fetch_prague_stops.py
```

Any other GeoJSON `FeatureCollection` of `Point` stops works too, as long as
each feature has `geometry.coordinates: [lon, lat]` and a `properties.stop_id`
(or `id` / `atco_code` / `naptan_code`).
