"""Validate an RCDx road-condition feed (see spec/README.md).

Checks structure, RCDx semantics, and privacy requirements.

Usage:
    python tools/validate_feed.py spec/example-detections.json
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "spec" / "rcdx-v0.1.schema.json"

# Field names that must never appear: RCDx carries road conditions, not people.
BANNED = {
    "image", "image_url", "photo", "video", "video_url", "audio", "frame",
    "vehicle_id", "vin", "plate", "license_plate", "driver", "driver_id",
    "passenger", "trip_id", "route_id", "user_id", "device_id", "imei",
}
MIN_COVERAGE_WINDOW = timedelta(hours=24)


def iso(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def walk_keys(obj, path=""):
    """Yield every (key, path) pair anywhere in the document."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, f"{path}.{k}"
            yield from walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_keys(v, f"{path}[{i}]")


def validate(path):
    errors, warnings = [], []
    try:
        feed = json.loads(Path(path).read_text())
    except Exception as e:
        return [f"not valid JSON: {e}"], []

    # --- optional JSON Schema pass ---
    try:
        import jsonschema
        try:
            jsonschema.validate(feed, json.loads(SCHEMA.read_text()))
        except jsonschema.ValidationError as e:
            errors.append(f"schema: {e.message} (at {'/'.join(str(p) for p in e.path)})")
    except ImportError:
        warnings.append("jsonschema not installed — structural check skipped "
                        "(pip install jsonschema for full validation)")

    meta = feed.get("rcdx", {})
    feed_type = meta.get("feed_type")
    if feed_type not in ("detections", "coverage"):
        errors.append("rcdx.feed_type must be 'detections' or 'coverage'")

    # --- privacy: no personal or trip-identifying fields, anywhere ---
    for key, where in walk_keys(feed):
        if key.lower() in BANNED:
            errors.append(f"PRIVACY: field '{key}' is not permitted (at {where})")

    # --- per-feature semantics ---
    uncalibrated = no_precision = 0
    for i, f in enumerate(feed.get("features", [])):
        p = f.get("properties", {})
        at = f"features[{i}]"
        if "source" not in p:
            errors.append(f"{at}: 'source' is required — a report without "
                          f"provenance can't be interpreted across fleets")

        if feed_type == "detections":
            for req in ("condition_type", "detected_at"):
                if req not in p:
                    errors.append(f"{at}: detections feed requires '{req}'")
            if "confidence" in p and "confidence_basis" not in p:
                errors.append(f"{at}: 'confidence' given without "
                              f"'confidence_basis' — the number is not "
                              f"comparable across publishers")
            if p.get("confidence_basis") == "uncalibrated_score":
                uncalibrated += 1
            if "confidence" in p and "observed_precision_at_threshold" not in p:
                no_precision += 1

        elif feed_type == "coverage":
            for req in ("segment_id", "window_start", "window_end", "passes"):
                if req not in p:
                    errors.append(f"{at}: coverage feed requires '{req}'")
            a, b = iso(p.get("window_start")), iso(p.get("window_end"))
            if a and b:
                if b <= a:
                    errors.append(f"{at}: window_end must be after window_start")
                elif b - a < MIN_COVERAGE_WINDOW:
                    warnings.append(f"{at}: coverage window is under 24 h — "
                                    f"short windows can let a single vehicle's "
                                    f"route be reconstructed")

    if uncalibrated:
        warnings.append(f"{uncalibrated} detection(s) use 'uncalibrated_score'. "
                        f"These scores are not comparable to other publishers' "
                        f"— cities can't weight them against each other.")
    if no_precision:
        warnings.append(f"{no_precision} detection(s) omit "
                        f"'observed_precision_at_threshold'. Without it a city "
                        f"can't tell how often your reports are right.")
    if feed_type == "detections":
        warnings.append("Reminder: publish a coverage feed too. Detections alone "
                        "can't distinguish 'this street is fine' from 'nobody "
                        "drove here'.")
    return errors, warnings


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ok = True
    for path in sys.argv[1:]:
        errors, warnings = validate(path)
        print(f"\n=== {path} ===")
        for e in errors:
            print(f"  ERROR   {e}")
        for w in warnings:
            print(f"  warning {w}")
        if errors:
            ok = False
        else:
            print("  VALID RCDx v0.1 feed ✓")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
