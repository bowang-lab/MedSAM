#!/usr/bin/env python3
# src.scripts.train_yolo.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou as uy_box_iou

from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.yolo_splits import create_yolo_dataset

# =========================
# Dataset / tuning configuration (algorithmic, not env-specific)
# =========================

VISUALIZE_ONE = False

SPACE = {
    # optimizer / schedule
    "lr0": (1e-4, 5e-3),
    "lrf": (0.1, 1.0),
    "weight_decay": (0.0, 5e-4),
    "momentum": (0.85, 0.98),
    "warmup_epochs": (0.0, 2.0),
    # loss balance
    "box": (5, 15),
    "cls": (0.2, 2.0),
    # light augmentations (fundus-safe)
    "degrees": (0.0, 7.5),
    "translate": (0.0, 0.10),
    "scale": (0.0, 0.20),
    "shear": (0.0, 3.0),
    "hsv_h": (0.0, 0.02),
    "hsv_s": (0.0, 0.30),
    "hsv_v": (0.0, 0.30),
    "flipud": (0.0, 0.10),
    "fliplr": (0.0, 0.10),
    # avoid aggressive mixing for medical images
    "mosaic": (0.0, 0.10),
    "mixup": (0.0, 0.05),
    "copy_paste": (0.0, 0.0),
}

# =========================
# Reproducibility
# =========================
# Import centralized seed function from utils
from src.utils import set_global_seed


# =========================
# Dataset build helpers
# =========================
def infer_data_yaml_path(out_dir: Path) -> Path:
    return out_dir / "data.yaml"


def scan_filter(data_root: Path, out_dir: Path, exclude_datasets: Optional[List[str]] = None):
    print("[INFO] Scanning datasets…")
    image_factory = ImageFactory(root=data_root, auto_scan=True)

    image_factory.filter_empty_masks()

    if exclude_datasets:
        print(f"[INFO] Excluding datasets matching: {exclude_datasets}")
        image_factory.filter_datasets(exclude=exclude_datasets)


    images = image_factory.make_images()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images:
        raise RuntimeError(
            "No images after filtering. Check dataset layout and mask availability."
        )

    print(f"[INFO] Found {len(images)} paired samples across datasets.")
    if VISUALIZE_ONE:
        try:
            images[0].visualize(show=True)
        except Exception as e:
            print(f"[WARN] Visualization failed: {e!r}")

    return images


def create_yolo_ds(
    images,
    out_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    """
    Create the YOLO dataset structure at out_dir using the provided split ratios.
    """
    print("[INFO] Creating YOLO dataset structure…")
    create_yolo_dataset(
        images,
        train=train_ratio,
        val=val_ratio,
        test=test_ratio,
        out_dir=out_dir,
    )
    data_yaml = infer_data_yaml_path(out_dir)
    if not data_yaml.exists():
        raise FileNotFoundError(f"`data.yaml` not found at {data_yaml}.")
    return data_yaml


# =========================
# Ultralytics metrics → JSON-safe
# =========================
def _jsonify(o: Any) -> Any:
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if hasattr(o, "tolist"):
        try:
            return _jsonify(o.tolist())
        except Exception:
            pass
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonify(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if "torch" in type(o).__module__:
        try:
            return str(o)
        except Exception:
            return None
    if hasattr(o, "__dict__"):
        return {k: _jsonify(v) for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def _extract_ultralytics_metrics(val_obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    box = getattr(val_obj, "box", None)
    if box is not None:
        out["box"] = {
            "mp": _jsonify(getattr(box, "mp", np.nan)),
            "mr": _jsonify(getattr(box, "mr", np.nan)),
            "map50": _jsonify(getattr(box, "map50", np.nan)),
            "map": _jsonify(getattr(box, "map", np.nan)),
            "maps": _jsonify(getattr(box, "maps", [])),
        }

    speed = getattr(val_obj, "speed", None)
    if speed is not None:
        out["speed"] = _jsonify(speed)

    cm = getattr(val_obj, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix"):
        out["confusion_matrix_shape"] = list(np.shape(getattr(cm, "matrix", [])))

    results_dict = getattr(val_obj, "results_dict", None)
    if results_dict:
        out["results_dict"] = _jsonify(results_dict)

    out["raw_summary"] = _jsonify(val_obj)
    return out


# =========================
# Test evaluator utilities
# =========================
def _resolve_test_images(yolo_ds: Path, data: Dict[str, Any]) -> List[Path]:
    entry = data.get("test")
    if not entry:
        return []

    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()

    if p.suffix.lower() == ".txt":
        return [Path(s.strip()) for s in p.read_text().splitlines() if s.strip()]

    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in exts)

    raise ValueError(f"Unsupported test entry in data.yaml: {entry}")


def _label_path_from_image(img: Path) -> Path:
    parts = list(img.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        if lbl.exists():
            return lbl
    except ValueError:
        pass
    return img.with_suffix(".txt")


def _load_gt_xyxy(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lbl = _label_path_from_image(img_path)
    if not lbl.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    W, H = PILImage.open(img_path).size
    rows = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    xyxy, cls = [], []
    for r in rows:
        parts = r.split()
        c = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        Xc, Yc = xc * W, yc * H
        Wb, Hb = w * W, h * H
        x1, y1 = Xc - Wb / 2.0, Yc - Hb / 2.0
        x2, y2 = Xc + Wb / 2.0, Yc + Hb / 2.0
        xyxy.append([x1, y1, x2, y2])
        cls.append(c)

    return np.asarray(xyxy, np.float32), np.asarray(cls, np.int64)


def _predict_xyxy_conf(
    model: YOLO,
    img_path: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (xyxy, cls, conf) for all predictions passing thresholds.
    """
    res = model.predict(
        source=str(img_path),
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    if res.boxes is None or len(res.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    b = res.boxes
    xyxy = b.xyxy.cpu().numpy().astype(np.float32)
    cls = b.cls.cpu().numpy().astype(np.int64)
    cf = b.conf.cpu().numpy().astype(np.float32)
    return xyxy, cls, cf


def _keep_single_top_conf_per_class(
    xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, nc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce predictions to at most one (the highest-confidence) per class.
    """
    if xyxy.size == 0:
        return xyxy, cls, conf

    keep_idx: List[int] = []
    for c in range(nc):
        inds = np.where(cls == c)[0]
        if inds.size == 0:
            continue
        best = inds[np.argmax(conf[inds])]
        keep_idx.append(int(best))

    if not keep_idx:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return xyxy[keep_idx], cls[keep_idx], conf[keep_idx]


def evaluate_test_boxes_hungarian_strict(
    model: YOLO,
    yolo_ds: Path,
    *,
    device: str = "cuda:0",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.70,
    save_jsonl: Path | None = None,
    single_top_per_class: bool = False,
) -> Dict[str, Any]:
    """
    Custom box-level evaluation with Hungarian matching:

      - If single_top_per_class=False:
          Use all predictions returned by YOLO above (conf, iou) thresholds.

      - If single_top_per_class=True:
          For each image and class, keep only the highest-confidence predicted
          box (0 or 1 per class, per image).

      - For each class:
          Hungarian matching on IoU.
          Dice = 2*IoU/(1+IoU) for matched pairs.
          Denominator per class += max(#GT, #Pred) → unmatched count as zero.
    """
    data_yaml = Path(yolo_ds) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    data = yaml.safe_load(data_yaml.read_text())
    test_imgs = _resolve_test_images(yolo_ds, data)
    if not test_imgs:
        raise RuntimeError("No test images resolved from data.yaml 'test' entry.")

    # class count from model
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        nc = len(names)
        class_names = [names[i] for i in range(nc)]
    elif isinstance(names, list):
        nc = len(names)
        class_names = names
    else:
        nc = int(max(getattr(model, "nc", 2), 2))
        class_names = [str(i) for i in range(nc)]

    dice_sum = np.zeros(nc, dtype=np.float64)
    match_cnt = np.zeros(nc, dtype=np.int64)

    jsonl_fp = None
    if save_jsonl is not None:
        Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = open(save_jsonl, "w")

    for img_path in test_imgs:
        gt_b, gt_c = _load_gt_xyxy(img_path)

        # All predictions for the image
        pr_b_all, pr_c_all, pr_cf_all = _predict_xyxy_conf(
            model,
            img_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

        if single_top_per_class:
            pr_b, pr_c, pr_cf = _keep_single_top_conf_per_class(
                pr_b_all, pr_c_all, pr_cf_all, nc
            )
        else:
            pr_b, pr_c, pr_cf = pr_b_all, pr_c_all, pr_cf_all

        for c in range(nc):
            gi = np.where(gt_c == c)[0]
            pi = np.where(pr_c == c)[0]
            n_g, n_p = len(gi), len(pi)
            den = max(n_g, n_p)  # denominator counts unmatched as zero
            sum_dice = 0.0

            if n_g > 0 and n_p > 0:
                g = torch.from_numpy(gt_b[gi]).float()
                p = torch.from_numpy(pr_b[pi]).float()
                iou_mat = uy_box_iou(g, p).cpu().numpy()

                try:
                    import scipy.optimize as spo

                    row_ind, col_ind = spo.linear_sum_assignment(-iou_mat)
                    pairs = [(r, c2, iou_mat[r, c2]) for r, c2 in zip(row_ind, col_ind)]
                except Exception:
                    # Greedy fallback
                    pairs = []
                    used_g, used_p = set(), set()
                    flat = [
                        (r, c2, iou_mat[r, c2])
                        for r in range(n_g)
                        for c2 in range(n_p)
                    ]
                    flat.sort(key=lambda x: x[2], reverse=True)
                    for r, c2, v in flat:
                        if r not in used_g and c2 not in used_p:
                            used_g.add(r)
                            used_p.add(c2)
                            pairs.append((r, c2, v))

                for _, _, u in pairs:
                    d = (2.0 * float(u)) / (1.0 + float(u)) if u > 0.0 else 0.0
                    sum_dice += d

            dice_sum[c] += sum_dice
            match_cnt[c] += den

            if jsonl_fp is not None:
                rec = {
                    "image": str(img_path),
                    "class": int(c),
                    "n_gt": int(n_g),
                    "n_pred": int(n_p),
                    "sum_dice": float(sum_dice),
                    "den": int(den),
                }
                jsonl_fp.write(json.dumps(rec) + "\n")

    if jsonl_fp is not None:
        jsonl_fp.close()

    eps = 1e-12
    per_class = (dice_sum / np.maximum(match_cnt, eps)).tolist()
    macro = float(np.mean(per_class)) if nc else 0.0

    return {
        "per_class_dice": per_class,
        "macro_dice": macro,
        "match_count": match_cnt.tolist(),
        "class_names": class_names,
        "conf_used": conf,
        "iou_used": iou,
        "single_top_per_class": single_top_per_class,
    }


# =========================
# Modular class API
# =========================
class YOLORunner:
    """
    Modular runner for the YOLO glaucoma pipeline.

    Typical usage from Python:

        from src.scripts.train_yolo import YOLORunner

        runner = YOLORunner(
            data_root=Path("/path/to/datasets"),
            out_dir=Path("./yolo_out"),
            run_dir=Path("./runs"),
            cfg=Path("./src/configs/train_custom.yaml"),
            model="yolo12x.pt",
            device="mps",
            epochs=100,
            batch=32,
            imgsz=640,
            conf=0.01,
            iou=0.70,
            seed=42,
            run_name="run1",
            single_top_per_class=False,  # or True to restrict to one pred per class
            papila=0.0,
        )

        data_yaml = runner.ensure_data()
        weights = runner.train()
        runner.test(weights, single_top_per_class=True)
    """

    def __init__(
        self,
        *,
        data_root: Path,
        out_dir: Path,
        run_dir: Path,
        cfg: Path,
        model: str,
        device: str,
        epochs: int,
        batch: int,
        imgsz: int,
        conf: float,
        iou: float,
        yolo_ds: Optional[Path] = None,
        run_name: Optional[str] = None,
        exclude_datasets: Optional[List[str]] = None,
        resume: bool = False,
        seed: int = 42,
        single_top_per_class: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        self.data_root = data_root
        self.out_dir = out_dir
        self.run_dir = run_dir              # base runs directory, e.g. ./runs
        self.cfg = cfg
        self.model_name = model
        self.device = device
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.conf = float(conf)
        self.iou = float(iou)
        self.exclude_datasets = exclude_datasets
        self.yolo_ds = yolo_ds
        self.run_name = run_name or "run"
        self.resume = resume
        self.seed = seed
        self.single_top_per_class = single_top_per_class
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)

        self._data_yaml: Optional[Path] = None
        self._data_yaml_excluded: Optional[Path] = None
        self._base_model: Optional[YOLO] = None  # shared between train/tune

        # Ensure base runs dir and dataset out_dir exist
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -------- run directories --------
    @property
    def run_root(self) -> Path:
        """Root directory for this logical run, e.g. ./runs/run1."""
        return self.run_dir / self.run_name

    @property
    def train_dir(self) -> Path:
        """Directory for training artifacts, e.g. ./runs/run1/Train."""
        return self.run_root / "Train"

    @property
    def test_dir(self) -> Path:
        """Directory for test artifacts, e.g. ./runs/run1/Test."""
        return self.run_root / "Test"

    # -------- internal helpers --------
    def _get_base_model(self) -> YOLO:
        """
        Lazily create the base YOLO model used for training/tuning.
        """
        if self._base_model is None:
            set_global_seed(self.seed)
            self._base_model = YOLO(self.model_name)
        return self._base_model

    # -------- public API --------
    def ensure_data(
        self,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
    ) -> Path:
        """
        Ensure a YOLO-style dataset exists and return the path to data.yaml.

        Behaviour:
          * If yolo_ds is provided, use that directory (expects data.yaml).
          * Otherwise, scan/filter raw datasets and create YOLO splits under out_dir.

        If ratios are not provided, defaults to self.train_ratio/val_ratio/test_ratio.
        """
        if self._data_yaml is not None:
            return self._data_yaml

        # Use instance defaults if not overridden
        tr = self.train_ratio if train_ratio is None else float(train_ratio)
        vr = self.val_ratio if val_ratio is None else float(val_ratio)
        ter = self.test_ratio if test_ratio is None else float(test_ratio)

        if self.yolo_ds is not None:
            data_yaml = self.yolo_ds / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(
                    f"[ERR] Provided yolo_ds but {data_yaml} does not exist."
                )
            print(f"[INFO] Using pre-existing YOLO dataset: {self.yolo_ds}")
            self._data_yaml = data_yaml
            return data_yaml

        # Build from raw datasets
        print("[INFO] Creating YOLO dataset (no pre-existing yolo_ds provided)…")
        images = scan_filter(
            self.data_root,
            self.out_dir,
            exclude_datasets=self.exclude_datasets,
        )
        self._data_yaml = create_yolo_ds(
            images,
            self.out_dir,
            train_ratio=tr,
            val_ratio=vr,
            test_ratio=ter,
        )

        self.yolo_ds = self.out_dir  # for later evaluation
        return self._data_yaml

    def train(self) -> Optional[Path]:
        """
        Run YOLO training. Returns the path to the trained weights (best.pt or last.pt),
        or None if weights could not be resolved.

        Training artifacts live under:
            {run_root}/Train/
        i.e.:
            ./runs/{run_name}/Train/Train/weights/best.pt
        """
        data_yaml = self.ensure_data()
        model = self._get_base_model()
        project_dir = self.train_dir
        project_dir.mkdir(parents=True, exist_ok=True)
        run_name = "Train"

        print(f"[INFO] Starting training… project={project_dir}, name={run_name}")
        model.train(
            cfg=str(self.cfg),
            project=str(project_dir),
            name=run_name,
            data=str(data_yaml),
            device=self.device,
            epochs=self.epochs,
            exist_ok=False,
            resume=self.resume,
            freeze=5,
            imgsz=self.imgsz,
            batch=self.batch,
            amp=False
        )

        weights_dir = project_dir / run_name / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        trained_weights_path = (
            best_pt if best_pt.exists()
            else (last_pt if last_pt.exists() else None)
        )
        if trained_weights_path is None:
            print(
                f"[WARN] Could not locate trained weights at {weights_dir} "
                "(best.pt/last.pt)."
            )
        else:
            print(f"[OK] Trained weights: {trained_weights_path}")
        return trained_weights_path

    def tune(self) -> Dict[str, Any]:
        """
        Run hyperparameter tuning with the configured SPACE.
        Returns the best hyperparameters dictionary from Ultralytics.

        Tuning artifacts live under:
            {run_root}/Tune/
        """
        data_yaml = self.ensure_data()
        model = self._get_base_model()
        project_dir = self.run_root / "Tune"
        project_dir.mkdir(parents=True, exist_ok=True)
        run_name = "Tune"

        print(f"[INFO] Starting tuning… project={project_dir}, name={run_name}")
        best = model.tune(
            data=str(data_yaml),
            project=str(project_dir),
            name=run_name,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
            iterations=15,
            resume=self.resume,
            epochs=self.epochs,
            optimizer="AdamW",
            plots=True,
            save=True,
            val=True,
            space=SPACE,
        )
        print("[OK] Best hyperparameters saved at:", best)
        return best

    def test(
        self,
        weights: Path,
        *,
        single_top_per_class: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run test-time evaluation using:
          1) Ultralytics built-in metrics on the 'test' split.
          2) Custom Hungarian-based strict Dice evaluation.

        If single_top_per_class is:
          - True: keep only the highest-confidence prediction per class per image.
          - False: use all predictions.
          - None: fall back to self.single_top_per_class.

        All test artifacts are written under:
            {run_root}/Test/

        Specifically:
          * {run_root}/Test/test_metrics.json
          * {run_root}/Test/test_dice_records.jsonl
          * {run_root}/Test/test_box_summary_combined.json

        Returns a dictionary with both metric summaries.
        """
        data_yaml = self.ensure_data()
        if not weights.exists():
            raise FileNotFoundError(f"[ERR] test weights not found: {weights}")

        test_dir = self.test_dir
        test_dir.mkdir(parents=True, exist_ok=True)

        # Decide behaviour for custom evaluator
        if single_top_per_class is None:
            single_top_per_class_flag = self.single_top_per_class
        else:
            single_top_per_class_flag = bool(single_top_per_class)

        print(f"[INFO] Running final evaluation on the TEST set… weights={weights}")
        print(f"[INFO] Custom Dice evaluator: single_top_per_class={single_top_per_class_flag}")

        model = YOLO(str(weights))

        # 1) Ultralytics built-in evaluation (use same thresholds)
        val_obj = model.val(
            data=str(data_yaml),
            split="test",
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
            conf=self.conf,
            iou=self.iou,
            save_json=True,
            plots=True,
            project=str(test_dir),
            name="UltralyticsVal",
        )
        ul_metrics = _extract_ultralytics_metrics(val_obj)
        out_json = test_dir / "test_metrics.json"
        with open(out_json, "w") as f:
            json.dump(ul_metrics, f, indent=2)
        print(f"[OK] Saved test metrics to: {out_json}")

        # 2) Strict Dice with unmatched counted as zero
        yolo_ds_dir = self.yolo_ds if self.yolo_ds is not None else self.out_dir
        summary = evaluate_test_boxes_hungarian_strict(
            model=model,
            yolo_ds=yolo_ds_dir,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            save_jsonl=test_dir / "test_dice_records.jsonl",
            single_top_per_class=single_top_per_class_flag,
        )

        agg_json = test_dir / "test_box_summary_combined.json"
        combined = {
            "ultralytics": ul_metrics,
            "box_metrics_summary": summary,
        }
        with open(agg_json, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"[OK] Saved combined summary to: {agg_json}")

        return combined


# =========================
# YOLO Finetuning from Parquet
# =========================

import shutil
from src.imgpipe.image import Image
from src.utils import ultralytics_device_arg


def get_next_run_name(run_root: Path, base: str = "ss") -> str:
    """Generate unique run name like ss1, ss2, etc."""
    i = 1
    while (run_root / f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


def ensure_yolo_dirs(ds_root: Path) -> None:
    """Create YOLO dataset directory structure."""
    for sub in (
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test",
    ):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)


def write_yolo_data_yaml(ds_root: Path) -> Path:
    """Write YOLO data.yaml config file."""
    data = {
        "path": str(ds_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "disc", 1: "cup"},
    }
    p = ds_root / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def fmt_yolo_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    """Format a YOLO label line with clamped values."""
    xc = float(min(1.0, max(0.0, xc)))
    yc = float(min(1.0, max(0.0, yc)))
    w = float(min(1.0, max(0.0, w)))
    h = float(min(1.0, max(0.0, h)))
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def materialize_image_file(img: Image, dst_img_path: Path) -> None:
    """Copy or extract image file to destination."""
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    ref = img.image_ref
    if ref is not None and getattr(ref, "packed", None) is not None:
        dst_img_path.write_bytes(ref.packed)
        return
    src = Path(img.image_path)
    if not src.exists():
        return
    shutil.copy2(src, dst_img_path)


def build_yolo_dataset_from_processed_parquet(
    images: list[Image],
    out_dir: Path,
) -> Path:
    """
    Materialize YOLO dataset from processed images.
    Assumes img.split and gt_*_box are already set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_yolo_dirs(out_dir)

    counts = {"train": 0, "val": 0, "test": 0}

    print(f"[INFO] Materializing {len(images)} images to {out_dir}...")

    for img in images:
        split = img.split
        if split not in counts:
            continue

        ext = (img.image_path.suffix or "").lower()
        if not ext:
            ext = getattr(img.image_ref, "ext", None) or ".png"
        if not ext.startswith("."):
            ext = "." + ext

        dst_img = out_dir / "images" / split / f"{img.uid}{ext}"
        dst_lbl = out_dir / "labels" / split / f"{img.uid}.txt"

        materialize_image_file(img, dst_img)

        lines = []
        if img.gt_disc_box:
            xc, yc, w, h = img.gt_disc_box.as_tuple()
            lines.append(fmt_yolo_line(0, xc, yc, w, h))
        if img.gt_cup_box:
            xc, yc, w, h = img.gt_cup_box.as_tuple()
            lines.append(fmt_yolo_line(1, xc, yc, w, h))

        if lines:
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1

    data_yaml = write_yolo_data_yaml(out_dir)
    print(f"[INFO] Materialization complete. Counts: {json.dumps(counts)}")
    return data_yaml


def finetune_yolo_from_parquet(
    images_parquet: Path,
    out_yolo_ds: Path,
    runs_root: Path,
    init_weights: Path,
    cfg: Path,
    device: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int = 8,
    seed: int = 42,
    skip_materialization: bool = False,
    run_name_base: str = "ss",
) -> Optional[Path]:
    """
    Finetune YOLO from a pre-processed parquet file.
    
    This function handles the full workflow:
    1. Load images from parquet
    2. Materialize YOLO dataset structure
    3. Initialize YOLORunner and train
    
    Args:
        images_parquet: Path to parquet with splits and GT boxes set
        out_yolo_ds: Where to materialize the YOLO dataset
        runs_root: Where to save training runs
        init_weights: Pretrained weights to finetune from
        cfg: YOLO training config file
        device: Device string (e.g., "0,1,2,3")
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        workers: Number of data workers
        seed: Random seed
        skip_materialization: If True, use existing dataset if found
        run_name_base: Base name for run directories (e.g., "ss" -> ss1, ss2, ...)
        
    Returns:
        Path to best weights, or None if training failed
    """
    set_global_seed(seed)
    
    target_device = device if device else ultralytics_device_arg()
    
    # 1. Prepare Dataset
    data_yaml = out_yolo_ds / "data.yaml"
    
    if skip_materialization and data_yaml.exists():
        print(f"[INFO] Skipping materialization. Using existing dataset: {data_yaml}")
    else:
        if skip_materialization:
            print(f"[WARN] --skip-materialization set but {data_yaml} not found. Proceeding with creation.")
        
        print(f"[INFO] Loading Parquet: {images_parquet}")
        images = Image.load_parquet(images_parquet)
        
        build_yolo_dataset_from_processed_parquet(images, out_dir=out_yolo_ds)
    
    # 2. Determine Run Name
    runs_root.mkdir(parents=True, exist_ok=True)
    run_name = get_next_run_name(runs_root, base=run_name_base)
    
    print(f"[INFO] Initializing YOLORunner for run: {run_name}")
    print(f"[INFO] Fine-tuning from: {init_weights}")
    
    # 3. Instantiate YOLORunner
    runner = YOLORunner(
        data_root=out_yolo_ds,  # Required by init but unused when yolo_ds is passed
        out_dir=out_yolo_ds,
        run_dir=runs_root,
        run_name=run_name,
        cfg=cfg,
        model=str(init_weights),
        device=target_device,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        conf=0.001,  # Test-time defaults
        iou=0.7,
        seed=seed,
        yolo_ds=out_yolo_ds,
    )
    
    # 4. Run Training
    best_weights = runner.train()
    
    print(f"[OK] Finetuning finished.")
    if best_weights:
        print(f"[OK] Best weights saved at: {best_weights}")
    else:
        print("[WARN] Could not resolve path to best weights.")
    
    return best_weights