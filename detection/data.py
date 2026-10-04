"""Detection datasets built from COCO128 (real labels).

- Small-object scenes: 9 downscaled images placed on a large canvas (objects only a few pixels wide).
- Occlusion set: 40 % to 60 % of some objects is covered by a synthetic "robotic arm".
"""
import zipfile

import cv2
import numpy as np

from common.utils import cache_path, download

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def coco128():
    """Returns a list of (RGB image, boxes[N,4] xyxy in px, classes[N])."""
    root = cache_path("datasets", "coco128")
    if not (root / "images").exists():
        z = download(COCO128_URL, cache_path("datasets", "coco128.zip"))
        with zipfile.ZipFile(z) as f:
            f.extractall(root.parent)
    items = []
    for ip in sorted((root / "images" / "train2017").glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lp = root / "labels" / "train2017" / (ip.stem + ".txt")
        boxes, cls = [], []
        if lp.exists():
            for line in lp.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                    cls.append(c)
        items.append((img, np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, np.int64)))
    return items


def conveyor_background(h, w, rng):
    bg = np.full((h, w, 3), 70, np.float32)
    bg += rng.normal(0, 6, (h, w, 1))
    for x in range(0, w, 60):  # conveyor rollers
        bg[:, x:x + 4] -= 25
    return np.clip(bg, 0, 255).astype(np.uint8)


def small_object_scenes(items, n_scenes=24, canvas=1440, grid=3, seed=1):
    rng = np.random.default_rng(seed)
    cell = canvas // grid
    scenes = []
    for s in range(n_scenes):
        bg = conveyor_background(canvas, canvas, rng)
        boxes, cls = [], []
        for gy in range(grid):
            for gx in range(grid):
                img, b, c = items[rng.integers(len(items))]
                h, w = img.shape[:2]
                sc = (cell - 40) / max(h, w)
                im = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                ox = gx * cell + rng.integers(0, cell - im.shape[1] + 1)
                oy = gy * cell + rng.integers(0, cell - im.shape[0] + 1)
                bg[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
                if len(b):
                    boxes.append(b * sc + np.array([ox, oy, ox, oy], np.float32))
                    cls.append(c)
        scenes.append((bg, np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
                       np.concatenate(cls) if cls else np.zeros(0, np.int64)))
    return scenes


def occlude(items, frac=(0.4, 0.6), min_side=40, max_per_img=2, seed=2):
    """Returns a list of (occluded image, boxes, classes, boolean mask of occluded objects)."""
    rng = np.random.default_rng(seed)
    out = []
    for img, b, c in items:
        im = img.copy()
        occl = np.zeros(len(b), bool)
        cand = [i for i in range(len(b)) if (b[i, 2] - b[i, 0]) >= min_side and (b[i, 3] - b[i, 1]) >= min_side]
        rng.shuffle(cand)
        for i in cand[:max_per_img]:
            x1, y1, x2, y2 = b[i].astype(int)
            bw, bh = x2 - x1, y2 - y1
            f = rng.uniform(*frac)
            side = rng.integers(4)  # the "arm" enters from one side of the object
            if side == 0:
                r = (x1, y1, x1 + int(bw * f), y2)
            elif side == 1:
                r = (x2 - int(bw * f), y1, x2, y2)
            elif side == 2:
                r = (x1, y1, x2, y1 + int(bh * f))
            else:
                r = (x1, y2 - int(bh * f), x2, y2)
            grad = np.linspace(110, 170, max(r[2] - r[0], 1), dtype=np.float32)[None, :, None]
            patch = np.broadcast_to(grad, (max(r[3] - r[1], 1), max(r[2] - r[0], 1), 3)).copy()
            patch += rng.normal(0, 4, patch.shape)
            im[r[1]:r[3], r[0]:r[2]] = np.clip(patch, 0, 255).astype(np.uint8)[: r[3] - r[1], : r[2] - r[0]]
            occl[i] = True
        out.append((im, b, c, occl))
    return out
