"""Detection lab for occlusion and small objects.

Compares:
  1. YOLOv8n on the full frame vs YOLOv8n with tiled inference (SAHI method) on scenes with small objects.
  2. YOLOv8n vs RT-DETR-l under synthetic partial occlusion (40-60 % of the object covered).
  3. CPU latency p50/p95: PyTorch vs ONNX FP32 vs ONNX INT8 (static quantization), and its accuracy cost.
"""
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

from common.utils import Tracker, cache_path, download, log, results_dir, save_json, timer  # noqa: E402
from detection.data import coco128, occlude, small_object_scenes  # noqa: E402
from detection.metrics import evaluate, recall_subset  # noqa: E402

OUT = results_dir("detection")
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
CONF = float(os.getenv("DET_CONF", "0.25"))
N_SCENES = int(os.getenv("DET_SCENES", "16"))
N_OCC = int(os.getenv("DET_OCCLUSION_IMAGES", "64"))
RTDETR_ON = os.getenv("DET_RTDETR", "1") == "1"


def load_models():
    from ultralytics import RTDETR, YOLO

    yolo = YOLO(str(download(ASSETS + "yolov8n.pt", cache_path("models", "yolov8n.pt"))))
    rtdetr = None
    if RTDETR_ON:
        rtdetr = RTDETR(str(download(ASSETS + "rtdetr-l.pt", cache_path("models", "rtdetr-l.pt"))))
    return yolo, rtdetr


def predict(model, imgs_rgb, imgsz=640):
    """Ultralytics expects BGR. Returns a list of (boxes xyxy, scores, classes)."""
    res = model.predict([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs_rgb], imgsz=imgsz, conf=CONF, verbose=False)
    return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)) for r in res]


# ---------------------------------------------------------------- Tiled inference (SAHI method)
def tile_origins(size, tile, overlap):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(size - tile, 0) + 1, step))
    if xs[-1] + tile < size:
        xs.append(size - tile)
    return xs


def sliced_predict(model, img, tile=640, overlap=0.2, include_full=True):
    """Implementation of the SAHI method (Akyon et al., 2022): overlapping native-resolution windows
    plus one full-frame pass, merged with per-class NMS."""
    h, w = img.shape[:2]
    tiles, offs = [], []
    for y in tile_origins(h, tile, overlap):
        for x in tile_origins(w, tile, overlap):
            tiles.append(img[y:y + tile, x:x + tile]); offs.append((x, y))
    B, S, C = [], [], []
    for (b, s, c), (x, y) in zip(predict(model, tiles, imgsz=tile), offs):
        B.append(b + np.array([x, y, x, y], np.float32)); S.append(s); C.append(c)
    if include_full:
        b, s, c = predict(model, [img])[0]
        B.append(b); S.append(s); C.append(c)
    B, S, C = np.concatenate(B), np.concatenate(S), np.concatenate(C)
    if len(B) == 0:
        return B, S, C
    keep = torchvision.ops.batched_nms(torch.from_numpy(B).float(), torch.from_numpy(S).float(),
                                       torch.from_numpy(C), 0.5).numpy()
    return B[keep], S[keep], C[keep]


def as_samples(gts, preds, masks=None):
    out = []
    for i, ((_, gb, gc, *_), (pb, ps, pc)) in enumerate(zip(gts, preds)):
        d = {"gt_b": gb, "gt_c": gc, "pr_b": pb, "pr_s": ps, "pr_c": pc}
        if masks is not None:
            d["mask"] = masks[i]
        out.append(d)
    return out


# ---------------------------------------------------------------- ONNX FP32 / INT8
def letterbox(img_rgb, size=640):
    h, w = img_rgb.shape[:2]
    r = size / max(h, w)
    im = cv2.resize(img_rgb, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - im.shape[0]) // 2, (size - im.shape[1]) // 2
    canvas[top:top + im.shape[0], left:left + im.shape[1]] = im
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, left, top


class OnnxDetector:
    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def raw(self, x):
        return self.sess.run(None, {self.inp: x})[0]

    def __call__(self, img_rgb):
        from ultralytics.utils import ops

        x, r, left, top = letterbox(img_rgb)
        out = self.raw(x)
        det = ops.non_max_suppression(torch.from_numpy(out), conf_thres=CONF, iou_thres=0.7)[0].numpy()
        b = det[:, :4].copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - left) / r
        b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
        return b, det[:, 4], det[:, 5].astype(int)


def export_and_quantize(yolo, calib_imgs):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    fp32 = cache_path("models", "yolov8n.onnx")
    if not fp32.exists():
        path = yolo.export(format="onnx", imgsz=640, simplify=False, dynamic=False, opset=13)
        os.replace(path, fp32)
    int8 = cache_path("models", "yolov8n_int8.onnx")
    if not int8.exists():
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"images": letterbox(i)[0]} for i in calib_imgs])

            def get_next(self):
                return next(self.it, None)

        quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                        op_types_to_quantize=["Conv"])  # the decoding head stays in FP32
    return fp32, int8


def latency(fn, img, runs=25, warmup=3):
    for _ in range(warmup):
        fn(img)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(img); ts.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}


def main():
    tracker = Tracker("detection", "sahi-rtdetr-int8")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    items = coco128()
    yolo, rtdetr = load_models()
    results = {}

    # ---------------------------------------------------------- 1 / Small objects: full frame vs SAHI
    scenes = small_object_scenes(items, n_scenes=N_SCENES)
    with timer("Scenes with small objects"):
        full = [predict(yolo, [s[0]])[0] for s in scenes]
        sliced = [sliced_predict(yolo, s[0]) for s in scenes]
    results["small_objects"] = {
        "YOLOv8n full frame": evaluate(as_samples(scenes, full)),
        "YOLOv8n + SAHI tiles (proposed)": evaluate(as_samples(scenes, sliced)),
    }
    for k, v in results["small_objects"].items():
        log(f"  {k}: {v}")

    # ---------------------------------------------------------- 2 / Occlusion: YOLOv8n vs RT-DETR-l
    subset = [it for it in items if len(it[1])][:N_OCC]
    occ = occlude(subset)
    masks = [o[3] for o in occ]
    models = {"YOLOv8n": yolo}
    if rtdetr is not None:
        models["RT-DETR-l (proposed)"] = rtdetr
    results["occlusion"] = {}
    with timer("Occlusion set"):
        for name, m in models.items():
            clean_pred = [predict(m, [im])[0] for im, *_ in subset]
            occ_pred = [predict(m, [o[0]])[0] for o in occ]
            results["occlusion"][name] = {
                "recall_unoccluded_objects": recall_subset(as_samples(subset, clean_pred, masks)),
                "recall_occluded_objects": recall_subset(as_samples(occ, occ_pred, masks)),
                "AP50_occluded_images": evaluate(as_samples(occ, occ_pred))["AP50"],
                "occluded_objects": int(sum(m_.sum() for m_ in masks)),
            }
            log(f"  {name}: {results['occlusion'][name]}")

    # ---------------------------------------------------------- 3 / Latency and quantization
    sample = subset[0][0]
    results["cpu_latency"] = {"YOLOv8n PyTorch": latency(lambda im: predict(yolo, [im]), sample)}
    if rtdetr is not None:
        results["cpu_latency"]["RT-DETR-l PyTorch"] = latency(lambda im: predict(rtdetr, [im]), sample, runs=10)
    results["quantization"] = {}
    try:
        with timer("ONNX export and INT8 quantization"):
            fp32, int8 = export_and_quantize(yolo, [it[0] for it in items[:32]])
        eval_set = subset[:32]
        for name, path in (("YOLOv8n ONNX FP32", fp32), ("YOLOv8n ONNX INT8 (proposed)", int8)):
            det = OnnxDetector(path)
            results["cpu_latency"][name] = latency(det, sample)
            preds = [det(im) for im, *_ in eval_set]
            results["quantization"][name] = {"AP50": evaluate(as_samples(eval_set, preds))["AP50"],
                                             "size_MB": os.path.getsize(path) / 1e6}
    except Exception as e:  # noqa: BLE001
        log(f"Quantization skipped: {e}")
    for k, v in results["cpu_latency"].items():
        log(f"  Latency {k}: {v}")
    for k, v in results["quantization"].items():
        log(f"  {k}: {v}")

    results["assumed_latency_budget_ms"] = 50
    results["note"] = ("Latencies were measured on the local CPU. In production they must be measured on the "
                       "target hardware. Knowledge distillation and retraining with augmentation need a GPU "
                       "(see train_demo.py).")
    save_json(results, OUT / "metrics.json")
    figures(scenes, full, sliced, occ, models, results)
    tracker.params({"conf": CONF, "scenes": N_SCENES, "occlusion_images": N_OCC})
    for sec in ("small_objects", "occlusion"):
        for k, v in results[sec].items():
            key = "".join(ch for ch in k if ch.isalnum())[:25]
            tracker.metrics({f"{sec}_{key}_{m}": x for m, x in v.items() if isinstance(x, (int, float))})
    tracker.end()
    log(f"Done. Results in {OUT}")


def draw(img, b, c=(255, 60, 60)):
    im = img.copy()
    for x1, y1, x2, y2 in b.astype(int):
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
    return im


def figures(scenes, full, sliced, occ, models, results):
    img = scenes[0][0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.8))
    ax[0].imshow(draw(img, full[0][0])); ax[0].set_title(f"Full frame / {len(full[0][0])} detections")
    ax[1].imshow(draw(img, sliced[0][0])); ax[1].set_title(f"SAHI tiles / {len(sliced[0][0])} detections")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "sahi_comparison.png", dpi=100); plt.close()

    im, b, c, m = next(o for o in occ if o[3].any())
    fig, axs = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    axs = np.atleast_1d(axs)
    for a, (name, mdl) in zip(axs, models.items()):
        pb, _, _ = predict(mdl, [im])[0]
        vis = draw(im, b[m], (60, 200, 60))
        a.imshow(draw(vis, pb)); a.set_title(f"{name}\ngreen = occluded object / red = detection"); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "occlusion_example.png", dpi=100); plt.close()

    lat = results["cpu_latency"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    names = list(lat)
    ax.barh(names, [lat[n]["p95_ms"] for n in names], color="#9673a6")
    ax.axvline(results["assumed_latency_budget_ms"], color="red", ls="--", label="Assumed budget (50 ms)")
    ax.set_xlabel("CPU latency p95 (ms)"); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "latency.png", dpi=100); plt.close()


if __name__ == "__main__":
    main()
