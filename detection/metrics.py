"""Detection metrics: AP50 (per class, all-point interpolation) and recall."""
import numpy as np


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(gt_b, gt_c, pr_b, pr_s, pr_c, thr=0.5):
    """Greedy matching by confidence. Returns (true positive per prediction, detected ground truth)."""
    order = np.argsort(-pr_s)
    tp = np.zeros(len(pr_b), bool)
    found = np.zeros(len(gt_b), bool)
    ious = iou_matrix(pr_b, gt_b)
    for i in order:
        cand = np.where((gt_c == pr_c[i]) & (~found))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            if ious[i, j] >= thr:
                tp[i] = True
                found[j] = True
    return tp, found


def evaluate(samples, small_area=32 ** 2):
    """samples: list of dicts with gt_b, gt_c, pr_b, pr_s, pr_c and an optional 'mask' (subset of GT)."""
    per_cls = {}
    n_gt, n_found, n_small, n_small_found = 0, 0, 0, 0
    for s in samples:
        tp, found = match(s["gt_b"], s["gt_c"], s["pr_b"], s["pr_s"], s["pr_c"])
        areas = (s["gt_b"][:, 2] - s["gt_b"][:, 0]) * (s["gt_b"][:, 3] - s["gt_b"][:, 1])
        n_gt += len(found); n_found += found.sum()
        sm = areas < small_area
        n_small += sm.sum(); n_small_found += found[sm].sum()
        for c in np.unique(np.concatenate([s["gt_c"], s["pr_c"]])):
            d = per_cls.setdefault(int(c), {"scores": [], "tp": [], "n_gt": 0})
            m = s["pr_c"] == c
            d["scores"] += list(s["pr_s"][m]); d["tp"] += list(tp[m]); d["n_gt"] += int((s["gt_c"] == c).sum())
    aps = []
    for d in per_cls.values():
        if d["n_gt"] == 0:
            continue
        o = np.argsort(-np.array(d["scores"]))
        tp = np.array(d["tp"], float)[o]
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = ctp / d["n_gt"]; prec = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        aps.append(float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])))
    return {"AP50": float(np.mean(aps)) if aps else 0.0, "recall": n_found / max(n_gt, 1),
            "small_object_recall": (n_small_found / n_small) if n_small else None, "objects": int(n_gt),
            "small_objects": int(n_small)}


def recall_subset(samples):
    """Recall only over the objects flagged in s['mask'] (for example, the occluded ones)."""
    tot, hit = 0, 0
    for s in samples:
        _, found = match(s["gt_b"], s["gt_c"], s["pr_b"], s["pr_s"], s["pr_c"])
        m = s["mask"]
        tot += m.sum(); hit += found[m].sum()
    return hit / max(tot, 1)
