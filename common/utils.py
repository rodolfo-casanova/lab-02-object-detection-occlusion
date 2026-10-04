"""Shared utilities: paths, results, and optional MLflow / MinIO integrations."""
import json
import os
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ROOT / "cache"))


def results_dir(name: str) -> Path:
    p = RESULTS_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(*parts) -> Path:
    p = CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def download(url: str, dest: Path) -> Path:
    """Download a file once and keep it in the cache."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cv-labs"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


_ENV_PREFIXES = ("BATCH_", "USE_", "DET_", "TRK_", "BIO_", "AR_", "WHISPER_", "VERIFY_", "OLLAMA_", "EDGE_", "DEVICE_", "BACKEND_")


def run_metadata(lab: str) -> dict:
    """Context that makes a metrics file self-explanatory: parameters, versions, and machine."""
    import datetime
    import platform
    from importlib import metadata

    versions = {}
    for pkg in ("numpy", "opencv-python-headless", "scikit-image", "torch", "torchvision", "ultralytics",
                "onnxruntime", "motmetrics", "insightface", "grad-cam", "faster-whisper"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001
            pass
    return {
        "lab": lab,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "parameters": {k: v for k, v in sorted(os.environ.items()) if k.startswith(_ENV_PREFIXES)},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "libraries": versions,
    }


def save_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "metrics.json" and isinstance(obj, dict):
        obj = {**obj, "_meta": run_metadata(path.parent.name)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    log(f"{name}: {time.perf_counter() - t0:.1f} s")


# ---------------------------------------------------------------- MLflow (optional)
class Tracker:
    """Logs parameters and metrics to MLflow when available. Otherwise it does nothing."""

    def __init__(self, experiment: str, run_name: str):
        self.active = False
        uri = os.getenv("MLFLOW_TRACKING_URI")
        if not uri:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self.run = mlflow.start_run(run_name=run_name)
            self.mlflow = mlflow
            self.active = True
            log(f"MLflow enabled: {uri} / experiment '{experiment}'")
        except Exception as e:  # noqa: BLE001
            log(f"MLflow unavailable ({e}). Continuing without tracking.")

    def params(self, d: dict):
        if self.active:
            try:
                self.mlflow.log_params({k: str(v)[:250] for k, v in d.items()})
            except Exception as e:  # noqa: BLE001
                log(f"MLflow params: {e}")

    def metrics(self, d: dict, step=None):
        if self.active:
            try:
                clean = {k.replace("@", "_at_").replace("%", "pct"): float(v) for k, v in d.items() if v is not None}
                self.mlflow.log_metrics(clean, step=step)
            except Exception as e:  # noqa: BLE001
                log(f"MLflow metrics: {e}")

    def end(self):
        if self.active:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------- MinIO (optional)
class ObjectStore:
    """S3-compatible object storage (MinIO). Does nothing when it is not available."""

    def __init__(self):
        self.client = None
        endpoint = os.getenv("MINIO_ENDPOINT")
        if not endpoint:
            return
        try:
            import boto3
            from botocore.config import Config

            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                region_name="us-east-1",
                config=Config(connect_timeout=5, retries={"max_attempts": 2}),
            )
            self.client.list_buckets()
            log(f"MinIO enabled: {endpoint}")
        except Exception as e:  # noqa: BLE001
            log(f"MinIO unavailable ({e}). Using local disk only.")
            self.client = None

    def ensure_bucket(self, bucket: str):
        if not self.client:
            return
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:  # noqa: BLE001
            self.client.create_bucket(Bucket=bucket)

    def put_bytes(self, bucket: str, key: str, data: bytes, content_type="image/png"):
        if not self.client:
            return
        try:
            self.client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        except Exception as e:  # noqa: BLE001
            log(f"MinIO put {bucket}/{key}: {e}")
