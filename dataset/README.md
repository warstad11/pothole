# Real-World Pothole Drives — Dataset

Real dashcam + motion-sensor recordings from everyday driving, collected
for the [Pothole Detection Lab](../README.md) project. Six drives, with
the accelerometer/gyroscope streams and 232 short, human-labeled video
clips.

> ## ⚠️ Please read the privacy note
> The **sensor data** (`drives/`) is privacy-scrubbed and carries no
> location information. The **video clips** (`clips/`) have been auto-blurred
> for faces, license plates, and text and were reviewed before release, but
> **automatic blurring is not perfect** — distant signs and recognizable
> scenery can remain (see "Residual risk" below and [TERMS.md](TERMS.md)).
> If you reuse or re-share this data, do not attempt to identify any person
> or location.

## What's inside

```
dataset/
  drives/
    drive_01/ … drive_06/
      sensor.csv     # seconds_elapsed, accel_x/y/z, gyro_x/y/z @ 100 Hz
      info.json      # phone model, duration, what was removed
  clips/
    drive_01/ …      # 4-second blurred clips, one per labeled event
  labels.csv         # clip_id, drive_id, t_center_s, label
```

`labels.csv` labels each clip as **pothole**, **not_pothole**, or
**unsure**, from a blinded human review (see the project's
[transfer study](../results/transfer/FINDINGS.md)).

| Label | Count | Note |
|---|---|---|
| not_pothole | 141 | cracks, bumps, manholes, smooth road, etc. |
| unsure | 79 | couldn't tell from the clip |
| pothole | 12 | confirmed potholes |

Potholes are **rare** in everyday driving — that imbalance is real and is
itself a finding. Use F1 or precision/recall, not accuracy.

## How it was collected

A phone was mounted at the windshield and recorded video plus motion data
(Sensor Logger app) during normal drives around Austin, Texas. The sensor
stream is resampled to 100 Hz. Each clip is a 4-second window (2 s before
to 2 s after) around a moment a detector flagged.

## How privacy was protected

**Removed completely (reliable):**
- GPS coordinates, speed, and bearing — gone from every file.
- Exact timestamps, time zone, and the phone's unique device ID.
- Street names — drives are renamed `drive_01`…`drive_06`; the real names
  never leave the recorder's computer.
- The start and end of each drive (≈20 s) — trimmed, because they are most
  likely to show the recorder's own home.

**Blurred in the video (best-effort, not perfect):**
- Faces, license plates, and on-screen text (street signs, house numbers,
  business names) are detected automatically and pixelated every frame.
- **Residual risk:** automatic detection misses small, distant, blurry, or
  angled text, and it cannot hide a recognizable building or landmark. A
  determined person with local knowledge might still recognize a location.
  This is why the clips need human review before publishing, and why the
  terms below forbid re-identification.

See [TERMS.md](TERMS.md) before using or sharing this data.

## License

Sensor data and labels: **CC BY-NC 4.0** (research/education, non-commercial,
with attribution). Video clips: same, **plus the re-identification ban in
[TERMS.md](TERMS.md)**. Reproduce the scrub with
`tools/build_release_dataset.py` and `tools/blur_clips.py`.
