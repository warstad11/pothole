"""
Shared, persistent train/val/test splits.

Why this exists
---------------
Reported metrics are only valid if the test partition is (a) disjoint from
everything used to fit or select models and (b) identical across every stage
that claims to be evaluated on "the same data". Before this module, each
wrapper did its own ad-hoc 80/20 split, selected checkpoints on the val set,
and then reported metrics from that same val set; the hybrid fusion stage
re-split randomly, so its "validation" rows overlapped the component models'
training data.

The contract
------------
- Splits are at the FILE level (one sensor CSV / image = one unit; a file is
  a recording session, so windows from one file never straddle partitions).
- The split is stratified by file-level label when labels are available,
  shuffled deterministically by seed, and persisted to
  ``<dataset>/split_manifest.json``.
- Every pipeline stage (sensor, image, fusion) loads the same manifest, so
  the test partition is untouched by component training, checkpoint
  selection, AND fusion training.
- ``train`` is for fitting, ``val`` for checkpoint/hyperparameter selection,
  ``test`` for reported metrics only.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

MANIFEST_NAME = "split_manifest.json"
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
PARTITIONS = ("train", "val", "test")


def _stratified_assignment(
    stems_by_label: Dict[Optional[int], List[str]],
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, List[str]]:
    """Split each label group by the ratios so rare classes appear in every partition."""
    rng = random.Random(seed)
    out: Dict[str, List[str]] = {p: [] for p in PARTITIONS}
    for label in sorted(stems_by_label, key=lambda x: (x is None, x)):
        stems = sorted(stems_by_label[label])
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        # Guarantee at least one sample per partition when the group is big enough
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        out["train"].extend(stems[:n_train])
        out["val"].extend(stems[n_train:n_train + n_val])
        out["test"].extend(stems[n_train + n_val:])
    for p in PARTITIONS:
        out[p].sort()
    return out


def get_or_create_split(
    dataset_path: Path,
    files: Sequence[Path],
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    label_fn: Optional[Callable[[Path], Optional[int]]] = None,
) -> Dict:
    """Load the dataset's split manifest, creating it on first use.

    files: the recording/sample files discovered by the caller.
    label_fn: optional file -> 0/1/None used to stratify on first creation.

    Returns the manifest dict with keys: seed, ratios, train, val, test
    (each a sorted list of file stems).
    """
    dataset_path = Path(dataset_path)
    manifest_path = dataset_path / MANIFEST_NAME
    stems = sorted({f.stem for f in files})

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        known = set(manifest["train"]) | set(manifest["val"]) | set(manifest["test"])
        unknown = [s for s in stems if s not in known]
        if unknown:
            # Files added after the manifest was frozen are excluded rather than
            # silently re-rolled into a new split (which would invalidate any
            # model already trained against this manifest).
            print(f"[splits] WARNING: {len(unknown)} file(s) not in {manifest_path.name}; "
                  f"they will be EXCLUDED. Delete the manifest to re-split. "
                  f"e.g. {unknown[:3]}")
        missing = [s for s in known if s not in set(stems)]
        if missing:
            print(f"[splits] WARNING: {len(missing)} manifest stem(s) have no file on disk "
                  f"(dataset shrank since the split was frozen). e.g. {missing[:3]}")
        if manifest.get("seed") != seed:
            print(f"[splits] NOTE: requested seed {seed} ignored — using the frozen "
                  f"manifest's seed {manifest.get('seed')}.")
        return manifest

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    stems_by_label: Dict[Optional[int], List[str]] = {}
    stem_to_file = {f.stem: f for f in files}
    for s in stems:
        label = label_fn(stem_to_file[s]) if label_fn else None
        stems_by_label.setdefault(label, []).append(s)

    assignment = _stratified_assignment(stems_by_label, ratios, seed)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": list(ratios),
        "strategy": "file-level, stratified by file label" if label_fn else "file-level",
        "counts": {p: len(assignment[p]) for p in PARTITIONS},
        **assignment,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[splits] Created {manifest_path}: " +
          ", ".join(f"{p}={len(assignment[p])}" for p in PARTITIONS))
    return manifest


def partition_files(files: Sequence[Path], manifest: Dict) -> Dict[str, List[Path]]:
    """Bucket files into train/val/test according to the manifest (by stem)."""
    stem_partition = {}
    for p in PARTITIONS:
        for s in manifest[p]:
            stem_partition[s] = p
    out: Dict[str, List[Path]] = {p: [] for p in PARTITIONS}
    for f in sorted(files):
        p = stem_partition.get(f.stem)
        if p is not None:
            out[p].append(f)
    return out


def stem_partition_map(manifest: Dict) -> Dict[str, str]:
    """stem -> partition lookup dict (build once, O(1) per query)."""
    out = {}
    for p in PARTITIONS:
        for s in manifest[p]:
            out[s] = p
    return out


def partition_of_stem(stem: str, manifest: Dict) -> Optional[str]:
    """Which partition a stem belongs to, or None if not in the manifest.
    For bulk queries build stem_partition_map() once instead."""
    return stem_partition_map(manifest).get(stem)
