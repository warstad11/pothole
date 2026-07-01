"""Build the privacy-scrubbed public dataset into GITHUB/dataset/.

READS ONLY from data/iphone/ and results/transfer/ — never modifies them.
WRITES ONLY into GITHUB/dataset/ (plus a PRIVATE id->folder map at the
repo root that must NOT be published).

Deterministic, 100%-reliable scrubbing (Tiers 1-2 from the privacy
assessment):
  - sensor.csv: keep only seconds_elapsed + accel + gyro (drop GPS/speed)
  - neutral drive IDs (drive_01..); real street-name folders never leave
    your machine (private map kept out of the release)
  - device UUID, exact timestamps, timezone removed
  - labels.csv built from the blinded human review (iPhone clips only;
    nuScenes clips excluded — that data is Motional's, not ours to ship)
  - home-segment trim: clips within HOME_TRIM_S of a drive's start/end are
    excluded (your driveway/house is most likely visible there)

The video blurring (Tier 3 — faces/plates/signs in the frames) is a
SEPARATE step: tools/blur_clips.py reads clip_manifest.json written here.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IPHONE = ROOT / "data" / "iphone"
REVIEW = ROOT / "results" / "transfer" / "review_items.json"
LABELS = ROOT / "results" / "transfer" / "labels.jsonl"
OUT = ROOT / "GITHUB" / "dataset"
PRIVATE_MAP = ROOT / "release_private_mapping.DO_NOT_PUBLISH.json"
# clip manifest holds ORIGINAL street-name paths -> stays OFF the release.
CLIP_MANIFEST = ROOT / "release_clip_manifest.DO_NOT_PUBLISH.json"

HOME_TRIM_S = 20.0          # exclude clips this close to a drive's ends
KEEP_COLS = ["seconds_elapsed", "accel_x", "accel_y", "accel_z",
             "gyro_x", "gyro_y", "gyro_z"]


def phone_model(orig_dir):
    meta = orig_dir / "Metadata.csv"
    if not meta.exists():
        return "unknown"
    rows = list(csv.DictReader(open(meta)))
    return rows[0].get("device name", "unknown") if rows else "unknown"


def scrub_sensor(orig_dir, out_csv):
    src = orig_dir / "sensor.csv"
    rows = list(csv.DictReader(open(src)))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in KEEP_COLS})
    return len(rows)


def scrub_published_transfer(drive_id):
    """Overwrite the GITHUB copy of review_items.json with a PII-free
    version (reads the untouched project-tree original). Removes the 837
    GPS coords, neutralizes iPhone drive names, drops clip/image/video
    paths and absolute video filenames. Keeps everything the metrics tool
    needs (platform, kind, per-pipeline provenance)."""
    pub = ROOT / "GITHUB" / "results" / "transfer" / "review_items.json"
    if not pub.exists():
        return 0
    data = json.loads(REVIEW.read_text())
    for it in data["items"]:
        if it.get("platform") == "iphone":
            it["drive"] = drive_id.get(it["drive"], "drive_xx")
        it.pop("gps", None)
        it.pop("clip_url", None)
        it.pop("image_url", None)
    for d in data.get("manifest", {}).get("drives", []):
        if d.get("platform") == "iphone":
            d["drive"] = drive_id.get(d["drive"], "drive_xx")
        d.pop("video", None)          # filename is a unix-ms timestamp
        d.pop("gps_available", None)
    # review_sampling.per_drive is a dict KEYED by drive name
    samp = data.get("manifest", {}).get("review_sampling", {})
    if "per_drive" in samp:
        samp["per_drive"] = {drive_id.get(k, k): v
                             for k, v in samp["per_drive"].items()}
    # final guard: refuse to write if any original street name survives
    blob = json.dumps(data, indent=1)
    for realname in drive_id:
        if realname in blob:
            raise RuntimeError(f"street name '{realname}' survived scrub — aborting")
    pub.write_text(blob)
    return len(data["items"])


def main():
    review = json.loads(REVIEW.read_text())
    items = {it["id"]: it for it in review["items"]}
    intervals = {d["drive"]: d["interval"] for d in review["manifest"]["drives"]}

    labels = {}
    for line in LABELS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["item_id"]] = r["label"]

    # iPhone drives only, stable neutral IDs by sorted real name
    iphone_drives = sorted({it["drive"] for it in items.values()
                            if it["platform"] == "iphone"})
    drive_id = {name: f"drive_{i+1:02d}" for i, name in enumerate(iphone_drives)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drives").mkdir(exist_ok=True)
    private = {"WARNING": "PRIVATE — contains home street names. Never commit/publish.",
               "mapping": {v: k for k, v in drive_id.items()}}
    PRIVATE_MAP.write_text(json.dumps(private, indent=1))

    # ---- per-drive sensor scrub ----
    drive_info = {}
    for name in iphone_drives:
        did = drive_id[name]
        orig = IPHONE / name
        ddir = OUT / "drives" / did
        ddir.mkdir(parents=True, exist_ok=True)
        n = scrub_sensor(orig, ddir / "sensor.csv")
        lo, hi = intervals.get(name, [0, 0])
        info = {"drive_id": did, "phone_model": phone_model(orig),
                "sample_rate_hz": 100, "n_samples": n,
                "duration_s": round(n / 100, 1),
                "columns": KEEP_COLS,
                "removed_for_privacy": ["latitude", "longitude", "speed",
                                        "absolute_timestamp", "device_id",
                                        "timezone", "recording_datetime"]}
        (ddir / "info.json").write_text(json.dumps(info, indent=1))
        drive_info[name] = (did, lo, hi)

    # ---- labels + clip manifest (iPhone only, home-trimmed) ----
    label_rows, clip_manifest = [], []
    clip_counter = {d: 0 for d in drive_id.values()}
    excluded_home = 0
    for iid, lab in sorted(labels.items()):
        it = items.get(iid)
        if not it or it["platform"] != "iphone":
            continue
        name = it["drive"]
        did, lo, hi = drive_info[name]
        t = it["time"]
        home = (t - lo < HOME_TRIM_S) or (hi - t < HOME_TRIM_S)
        if home:
            excluded_home += 1
            continue
        clip_counter[did] += 1
        cid = f"{did}_clip_{clip_counter[did]:03d}"
        orig_clip = IPHONE / name / "clips_transfer" / f"{iid}.mp4"
        if not orig_clip.exists():
            continue
        out_clip = OUT / "clips" / did / f"{cid}.mp4"
        label_rows.append({"clip_id": cid, "drive_id": did,
                           "t_center_s": round(t, 2), "label": lab})
        clip_manifest.append({"clip_id": cid, "drive_id": did,
                              "src": str(orig_clip), "dst": str(out_clip)})

    (OUT / "clips").mkdir(exist_ok=True)
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "drive_id", "t_center_s", "label"])
        w.writeheader()
        w.writerows(label_rows)
    CLIP_MANIFEST.write_text(json.dumps(clip_manifest, indent=1))

    n_pub = scrub_published_transfer(drive_id)

    from collections import Counter
    by_label = Counter(r["label"] for r in label_rows)
    print(f"scrubbed published review_items.json ({n_pub} items: GPS + "
          f"street names removed)")
    print(f"drives: {len(iphone_drives)} | sensor.csv scrubbed")
    print(f"clips selected for release: {len(clip_manifest)} "
          f"(excluded {excluded_home} home-proximal clips)")
    print(f"labels: {dict(by_label)}")
    print(f"private map -> {PRIVATE_MAP.name} (keep OFF GitHub)")
    print(f"next: python tools/blur_clips.py  (blurs the {len(clip_manifest)} clips)")


if __name__ == "__main__":
    main()
