from typing import List, Dict
import numpy as np


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection + 1e-6)


def _calculate_ap(precisions: List[float], recalls: List[float]) -> float:
    """Average Precision via all-point interpolation (VOC-2010+ definition).

    Computes the monotone (non-increasing) precision envelope, then
    integrates precision over recall continuously:
    ``sum((r[i] - r[i-1]) * p_envelope[i])``. This replaces the older
    11-point VOC-2007 interpolation, which quantizes recall and biases AP.
    Note COCO uses 101-point interpolation — numbers computed here are not
    directly comparable to COCO-evaluated results.
    """
    if not precisions or not recalls:
        return 0.0

    # Recalls come from a cumulative TP count over score-sorted detections,
    # so they must already be non-decreasing. Out-of-order recalls indicate
    # a caller bug — raise (a plain assert vanishes under `python -O`).
    if any(recalls[i] > recalls[i + 1] for i in range(len(recalls) - 1)):
        raise ValueError("_calculate_ap: recalls must be non-decreasing")

    # Monotone precision envelope (right-to-left running max)
    precisions_interp = list(precisions)
    for i in range(len(precisions_interp) - 2, -1, -1):
        precisions_interp[i] = max(precisions_interp[i], precisions_interp[i + 1])

    # Continuous integration of precision over recall
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions_interp, recalls):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def calculate_map(preds: List[Dict], targets: List[Dict],
                  iou_threshold: float = 0.5) -> float:
    """Mean Average Precision at a given IoU threshold.

    Args:
        preds: per-image dicts with ``boxes`` (N,4), ``scores`` (N,), ``labels`` (N,).
        targets: per-image dicts with ``boxes`` (M,4), ``labels`` (M,).
        iou_threshold: IoU threshold for TP.

    Returns:
        mAP (0..1).

    Note:
        Classes that appear in predictions but have no ground-truth boxes
        anywhere in ``targets`` are skipped (not counted as AP=0), per the
        VOC convention — AP is undefined when total_gt == 0 for a class.
    """
    if not preds or not targets:
        return 0.0

    all_classes = set()
    for t in targets:
        lbls = t.get('labels', [])
        if len(lbls) > 0:
            all_classes.update(int(c) for c in lbls)

    if not all_classes:
        return 0.0

    aps = []
    for cls_id in sorted(all_classes):
        det_scores = []
        det_matched = []
        total_gt = 0

        for pred, target in zip(preds, targets):
            p_boxes = np.asarray(pred.get('boxes', [])).reshape(-1, 4)
            p_scores = np.asarray(pred.get('scores', [])).ravel()
            p_labels = np.asarray(pred.get('labels', [])).ravel()

            t_boxes = np.asarray(target.get('boxes', [])).reshape(-1, 4)
            t_labels = np.asarray(target.get('labels', [])).ravel()

            gt_mask = t_labels == cls_id
            gt_boxes = t_boxes[gt_mask] if gt_mask.any() else np.empty((0, 4))
            total_gt += len(gt_boxes)

            det_mask = p_labels == cls_id
            d_boxes = p_boxes[det_mask] if det_mask.any() else np.empty((0, 4))
            d_scores = p_scores[det_mask] if det_mask.any() else np.array([])

            if len(d_scores) == 0:
                continue

            order = np.argsort(-d_scores)
            d_boxes = d_boxes[order]
            d_scores = d_scores[order]

            matched_gt = set()
            for di in range(len(d_boxes)):
                best_iou = 0.0
                best_gt = -1
                for gi in range(len(gt_boxes)):
                    if gi in matched_gt:
                        continue
                    iou = calculate_iou(d_boxes[di], gt_boxes[gi])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                det_scores.append(float(d_scores[di]))
                if best_iou >= iou_threshold and best_gt >= 0:
                    det_matched.append(1)
                    matched_gt.add(best_gt)
                else:
                    det_matched.append(0)

        if total_gt == 0:
            continue
        if len(det_scores) == 0:
            aps.append(0.0)
            continue

        order = np.argsort(-np.array(det_scores))
        det_matched = np.array(det_matched)[order]

        tp_cumsum = np.cumsum(det_matched)
        fp_cumsum = np.cumsum(1 - det_matched)

        precisions = (tp_cumsum / (tp_cumsum + fp_cumsum)).tolist()
        recalls = (tp_cumsum / total_gt).tolist()
        aps.append(_calculate_ap(precisions, recalls))

    return float(np.mean(aps)) if aps else 0.0
