# RCDx — Road Condition Data Exchange (draft v0.1)

**A simple, open format for any vehicle fleet to report road-condition
observations to a city.**

Modeled directly on [WZDx](https://github.com/usdot-jpo-ode/wzdx), the USDOT
Work Zone Data Exchange: GeoJSON, voluntary, no license fees, no vendor
lock-in. If you can publish a JSON file at a URL, you can participate.

> Status: draft proposal by [Warren Stading](https://github.com/warstad11),
> written after building and testing pothole detectors and finding out how
> badly they transfer between vehicles. Feedback and pull requests welcome —
> this is meant to be argued with, not adopted as-is.

## Why another spec?

Several fleets already detect potholes. Waymo shares detections with cities
through Waze; commercial trucking fleets run their own systems; cities have
311. But each feed is a different shape, so nobody can combine them — and
**combining them is the entire point.**

My own research is the argument. I trained a pothole detector on one vehicle,
froze it, and drove different roads: precision fell to roughly 0.10, and the
camera half of the system detected essentially nothing, because dashcam footage
didn't look like its training data
([full results](../results/transfer/FINDINGS.md)). A detector tuned to one
vehicle's cameras and suspension does not transfer to another vehicle.

The consequence for cities: **every single fleet's data has systematic blind
spots** — the streets it doesn't serve, the conditions its sensors handle
poorly, the defects its suspension doesn't feel. No one fleet is a census of
your roads. Aggregation isn't a nice-to-have; it's the only way to get coverage.

## Three design rules (each one learned the hard way)

**1. Provenance is required, not optional.**
Because detection doesn't transfer across vehicles, a report is uninterpretable
without knowing what produced it. Every detection carries its modalities,
vehicle class, and detector version. This is also how a city discovers that
"we have no reports from East Austin" means *nobody drove there*, not *the roads
are fine*.

**2. A confidence number must say what it means.**
In my testing, precision got *worse* as confidence rose on unfamiliar roads —
the most confident alerts were the most wrong. So a bare `0.9` from one fleet is
not comparable to `0.9` from another. RCDx requires `confidence_basis`, and asks
publishers to state the precision they actually observe at that threshold.

**3. Publish where you looked and found nothing.**
This is the field most detection feeds omit and the one cities most need. Without
a coverage feed you cannot distinguish "this street is fine" from "nothing with a
sensor has driven this street in six months." My study used randomly sampled
"probe" clips for exactly this reason — to measure what the detectors missed.
Coverage is how a city measures its own blind spots, and it's the honest basis
for equity analysis.

## The format

Two GeoJSON feeds. Both are plain GeoJSON — existing tools already read them.

### Detections feed

```json
{
  "type": "FeatureCollection",
  "rcdx": {
    "version": "0.1",
    "feed_type": "detections",
    "publisher": { "name": "Example Fleet", "contact": "data@example.com" },
    "published_at": "2026-08-23T19:00:00Z",
    "update_frequency_sec": 3600
  },
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-97.7431, 30.2672] },
      "properties": {
        "id": "ex-2026-08-23-000417",
        "condition_type": "pothole",
        "detected_at": "2026-08-23T18:42:11Z",
        "confidence": 0.82,
        "confidence_basis": "calibrated_precision",
        "observed_precision_at_threshold": 0.71,
        "source": {
          "modalities": ["camera", "imu"],
          "vehicle_class": "robotaxi",
          "detector_id": "example-detector-v3.2"
        },
        "verification": { "status": "unverified", "confirmations": 0 },
        "severity": "moderate"
      }
    }
  ]
}
```

### Coverage feed

```json
{
  "type": "FeatureCollection",
  "rcdx": {
    "version": "0.1",
    "feed_type": "coverage",
    "publisher": { "name": "Example Fleet", "contact": "data@example.com" },
    "published_at": "2026-08-23T19:00:00Z"
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "LineString",
        "coordinates": [[-97.7445, 30.2668], [-97.7419, 30.2681]]
      },
      "properties": {
        "segment_id": "osm:way/12345678",
        "window_start": "2026-08-01T00:00:00Z",
        "window_end": "2026-08-23T00:00:00Z",
        "passes": 42,
        "source": {
          "modalities": ["camera", "imu"],
          "vehicle_class": "robotaxi",
          "detector_id": "example-detector-v3.2"
        }
      }
    }
  ]
}
```

## Field reference

**`condition_type`** — `pothole`, `crack`, `rough_surface`, `debris`,
`standing_water`, `missing_marking`, `other`. Keep potholes distinct from
"rough road": in my benchmarks, a simple shake-detector rated speed bumps
*higher* than potholes, so lumping them together makes the data useless for
repair prioritization.

**`confidence_basis`** — one of:
- `calibrated_precision` — confidence is a calibrated probability the report is
  a true positive (best)
- `uncalibrated_score` — a raw model score, not comparable across publishers
- `human_verified` — a person confirmed it

**`observed_precision_at_threshold`** *(recommended)* — of reports published at
this confidence or above, the fraction that were real when someone checked. This
single number is what lets a city weight one fleet's feed against another's.

**`verification.status`** — `unverified`, `verified`, `rejected`, `repaired`.
Lets crowd confirmation (as Waze does) and city work orders close the loop.

**`vehicle_class`** — `robotaxi`, `passenger`, `truck`, `transit`, `municipal`,
`micromobility`. Different suspensions feel different defects; this is how a
city sees which vehicle types are over- or under-represented.

## Privacy requirements

RCDx feeds carry **road conditions, not people or trips.**

- No imagery, no audio, no raw sensor recordings.
- No vehicle, device, driver, or passenger identifiers — `detector_id` describes
  a *model version*, never a specific vehicle.
- No trip reconstruction: coverage is aggregated per segment per time window,
  never as a continuous path.
- Coverage windows should be long enough (recommended ≥ 24 h) that a single
  vehicle's route cannot be reconstructed.

I applied the same standard to my own released data — stripping GPS, timestamps,
device IDs, and blurring faces and plates before publishing
([how](../dataset/README.md)) — so I know it's achievable.

## Validate a feed

```bash
python tools/validate_feed.py path/to/feed.json
```

## Open questions

Genuinely unresolved, and I'd like input from people who run city road
operations:

1. Is per-segment coverage the right granularity, or is a coarse grid better?
2. Should severity be a controlled vocabulary or a physical measure (depth in cm)?
3. What's the minimum viable feed a small fleet could publish without a data team?
