from typing import List, Dict, Tuple
import numpy as np

class EventGenerator:
    @staticmethod
    def generate_events(times: np.array, scores: np.array, threshold: float = 0.75,
                        v_scores: np.array = None, s_scores: np.array = None,
                        gap_tolerance: float = None) -> List[Dict]:
        """
        Segment-based event extraction.

        Merges contiguous supra-threshold runs into segments (small gaps — up to
        roughly one sample interval — are bridged so a single missing/dipping
        sample does not split one physical pothole into two events). Each
        segment yields exactly ONE event located at its peak-score timestamp:

            {'time': peak_t, 't': peak_t, 'score': peak_score,
             't_start': segment_start, 't_end': segment_end,
             'v_score': ..., 's_score': ...}

        'time'/'score'/'v_score'/'s_score' keys are preserved for backward
        compatibility with existing consumers (e.g. worker.py).

        gap_tolerance: max time gap (seconds) between consecutive supra-threshold
        samples that still counts as the same segment. Defaults to ~2x the
        median sample interval (i.e. one intervening sample is tolerated).
        """
        times = np.asarray(times, dtype=float)
        scores = np.asarray(scores, dtype=float)
        n = len(times)
        if n == 0:
            return []

        # Ensure time-sorted order (keep auxiliary scores aligned)
        order = np.argsort(times)
        times = times[order]
        scores = scores[order]
        v_arr = np.asarray(v_scores, dtype=float)[order] if v_scores is not None else None
        s_arr = np.asarray(s_scores, dtype=float)[order] if s_scores is not None else None

        if gap_tolerance is None:
            if n > 1:
                dt = float(np.median(np.diff(times)))
                gap_tolerance = 2.05 * dt  # bridge a single missing/sub-threshold sample
            else:
                gap_tolerance = 0.0

        supra = np.nonzero(scores >= threshold)[0]
        if len(supra) == 0:
            return []

        # Split supra-threshold indices into segments wherever the time gap
        # between consecutive supra samples exceeds the tolerance.
        segments = []
        current = [supra[0]]
        for idx in supra[1:]:
            if times[idx] - times[current[-1]] <= gap_tolerance:
                current.append(idx)
            else:
                segments.append(current)
                current = [idx]
        segments.append(current)

        events = []
        for seg in segments:
            seg = np.asarray(seg)
            peak = int(seg[np.argmax(scores[seg])])
            evt = {
                'time': float(times[peak]),
                't': float(times[peak]),
                'score': float(scores[peak]),
                't_start': float(times[seg[0]]),
                't_end': float(times[seg[-1]]),
            }
            if v_arr is not None: evt['v_score'] = float(v_arr[peak])
            if s_arr is not None: evt['s_score'] = float(s_arr[peak])
            events.append(evt)
        return events

    @staticmethod
    def apply_nms(events: List[Dict], time_window: float = 1.0) -> List[Dict]:
        """Merge events within *time_window*, keeping the highest-scored one.

        Kept for backward compatibility: generate_events() now performs
        segment merging itself, so this is typically a no-op afterwards.

        Uses a greedy suppression approach (sort by score descending) so that
        non-adjacent events that still fall within the window are correctly
        suppressed.
        """
        if not events:
            return []

        remaining = sorted(events, key=lambda e: e['score'], reverse=True)
        suppressed = [False] * len(remaining)
        merged = []

        for i in range(len(remaining)):
            if suppressed[i]:
                continue
            merged.append(remaining[i])
            for j in range(i + 1, len(remaining)):
                if suppressed[j]:
                    continue
                if abs(remaining[j]['time'] - remaining[i]['time']) < time_window:
                    suppressed[j] = True

        merged.sort(key=lambda e: e['time'])
        return merged
