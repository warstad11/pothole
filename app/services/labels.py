"""
Centralized label inference from file paths, annotation files, and data columns.

All label decisions go through this module so that every pipeline (sensor,
image, hybrid) agrees on what "positive" means.

Task definition: label 1 means POTHOLE. Other road anomalies — cracks
(fatigue/alligator/slippage), spalls, faulting, shoving, bumps, construction
joints — are NOT potholes and map to label 0. This matches the cleaned
sensor datasets (see tools/relabel_sensor_data.py) and is the definition a
pothole-detection paper must use; treating every vibration-producing anomaly
as a pothole reduces the task to generic anomaly detection.
"""

import re
from pathlib import Path
from typing import Optional

# Only these mean "pothole".
POSITIVE_KEYWORDS = frozenset([
    'pothole', 'positive',
])

# Known not-a-pothole markers: normal road AND non-pothole anomalies.
# 'aligator' covers the misspelling used by the Data 3 source files.
NEGATIVE_KEYWORDS = frozenset([
    'normal', 'negative', 'undamaged', 'plain', 'regular',
    'joint', 'crack', 'cracking', 'alligator', 'aligator',
    'bump', 'shoving', 'spall', 'fault', 'faulting',
    'slippage', 'worn',
])

# "no_pothole", "not-pothole", "nonpothole", "not_a_pothole", "pothole_free"
_NEGATED_POSITIVE = re.compile(
    r'(?:\b|_)(?:no|non|not)(?:[\s_-]+(?:a|an|any|real))?[\s_-]*pothole'
    r'|pothole[\s_-]*(?:free|less)'
)


def _classify_string(source: str) -> Optional[int]:
    """Classify one lowercase string. Negation guard, then positive, then negative."""
    if _NEGATED_POSITIVE.search(source):
        return 0
    if any(kw in source for kw in POSITIVE_KEYWORDS):
        return 1
    if any(kw in source for kw in NEGATIVE_KEYWORDS):
        return 0
    return None


def infer_label_from_path(file_path: Path) -> Optional[int]:
    """Infer a binary label (1 = pothole, 0 = not pothole) from a file path.

    Checks the filename first, then the parent directory name.
    Returns ``None`` when no determination can be made.
    """
    for source in (file_path.name.lower(), file_path.parent.name.lower()):
        label = _classify_string(source)
        if label is not None:
            return label
    return None


def infer_label_from_yolo_txt(image_path: Path) -> Optional[int]:
    """Infer a binary label from an adjacent YOLO-format label file.

    Searches common YOLO directory layouts:
      - ``../labels/<stem>.txt``  (standard split layout)
      - ``./labels/<stem>.txt``
      - ``./<stem>.txt``
      - images→labels path substitution

    Returns 1 if the label file exists and has content, 0 if the file
    exists but is empty, or ``None`` if no label file is found.
    """
    stem_txt = image_path.with_suffix(".txt").name

    candidates = [
        image_path.parent.parent / "labels" / stem_txt,
        image_path.parent / "labels" / stem_txt,
        image_path.with_suffix(".txt"),
    ]

    # Handle images→labels substitution (dataset/images/val/x.jpg -> dataset/labels/val/x.txt)
    if "images" in image_path.parts:
        try:
            parts = list(image_path.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            candidates.append(Path(*parts).with_suffix(".txt"))
        except (ValueError, TypeError):
            pass

    for lbl_path in candidates:
        if lbl_path.exists():
            try:
                content = lbl_path.read_text().strip()
                return 1 if content else 0
            except OSError:
                continue

    return None


def parse_label_value(value) -> Optional[int]:
    """Parse a single raw label cell (string/number) into 0/1, or None if
    unparseable. Empty/NaN cells are None, NOT 0 — a dataset shipped with an
    empty label column (e.g. Data 2) must fall through to path inference
    rather than silently becoming all-negative."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN without numpy import
        return None
    s = str(value).strip().lower()
    if s in ('', 'nan', 'none'):
        return None
    if s in ('pothole', '1', '1.0', 'true', 'yes'):
        return 1
    if s in ('0', '0.0', 'false', 'no', 'normal', 'negative'):
        return 0
    try:
        v = float(s)
        if v != v:  # NaN
            return None
        return 1 if v >= 0.5 else 0
    except ValueError:
        return None


def infer_label(file_path: Path, df=None, label_col: str = 'label') -> Optional[int]:
    """Unified single-label inference with a clear priority order.

    Priority:
      1. YOLO label file (for image paths)
      2. Filename / directory keywords
      3. DataFrame column (first row)

    Returns ``None`` when no determination can be made — callers must decide
    whether to skip the sample or fail. Never silently defaults to 0: an
    unlabeled sample treated as negative is label noise.
    """
    # YOLO txt lookup only applies to images — a sensor CSV with a same-stem
    # sibling .txt must not be labeled by it.
    if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
        yolo_label = infer_label_from_yolo_txt(file_path)
        if yolo_label is not None:
            return yolo_label

    path_label = infer_label_from_path(file_path)
    if path_label is not None:
        return path_label

    if df is not None and label_col in df.columns and len(df) > 0:
        return parse_label_value(df[label_col].iloc[0])

    return None


def sensor_labels_for_df(file_path: Path, df, label_col: str = 'label'):
    """Resolve per-row labels for a sensor recording.

    Priority — the opposite of :func:`infer_label`, deliberately:
      1. An embedded label column (per-row, time-localized — most precise;
         the sensor datasets were cleaned so these are trustworthy).
      2. Path keywords, broadcast to every row.
      3. ``None`` — caller must skip the file (do NOT guess).

    Returns a pandas Series aligned to ``df.index``, or None.
    """
    import pandas as pd

    if label_col in df.columns:
        parsed = df[label_col].map(parse_label_value)
        if parsed.notna().any():
            # Unparseable rows in an otherwise-labeled file are dropped by
            # callers via NaN, not coerced to 0.
            return parsed.astype('float')

    path_label = infer_label_from_path(file_path)
    if path_label is not None:
        return pd.Series(float(path_label), index=df.index)

    return None
