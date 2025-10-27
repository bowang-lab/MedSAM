import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.imgpipe.bounding_box import BoundingBox
from src.imgpipe.enums import LabelType, Structure
from src.imgpipe.image import Image

# -------------------------------------------------------------------
# Hard-coded defaults (edit these to your real files)
# You can also override at runtime with env vars:
#   IMG_PATH, DISC_MASK_PATH, CUP_MASK_PATH
# -------------------------------------------------------------------
DEFAULT_IMG_PATH = "/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/SMDG-19/full-fundus/full-fundus/REFUGE1-val-99.png"
DEFAULT_DISC_MASK_PATH = "/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/SMDG-19/optic-disc/optic-disc/REFUGE1-val-99.png"
DEFAULT_CUP_MASK_PATH = "/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/SMDG-19/optic-cup/optic-cup/REFUGE1-val-99.png"


@pytest.fixture
def real_paths():
    # prefer env var overrides, else fall back to hard-coded defaults
    img = os.getenv("IMG_PATH", DEFAULT_IMG_PATH)
    disc = os.getenv("DISC_MASK_PATH", DEFAULT_DISC_MASK_PATH)
    cup = os.getenv("CUP_MASK_PATH", DEFAULT_CUP_MASK_PATH)

    img_p, disc_p, cup_p = Path(img), Path(disc), Path(cup)
    if not img_p.exists() or not disc_p.exists() or not cup_p.exists():
        pytest.skip(
            "real image/mask paths not found. "
            "Edit DEFAULT_* paths in the test file or set env vars "
            "IMG_PATH, DISC_MASK_PATH, CUP_MASK_PATH."
        )
    return img_p, disc_p, cup_p


# Pillow is used only for real-paths I/O/visualization
PIL_AVAILABLE = False
try:
    from PIL import Image as PILImage
    from PIL import ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ====================== existing tests (unchanged) ======================

def test_image_from_path_without_io(tmp_path):
    # Avoid disk I/O for size read by giving width/height explicitly
    img = tmp_path / "im.jpg"
    img.write_bytes(b"")  # path exists but no need to read
    s = Image.from_path(
        image_path=img,
        dataset="DS1",
        subject_id="SUBJ001",
        uid="abc123",
        split="train",
        width=640,
        height=480,
    )
    assert s.uid == "abc123"
    assert s.dataset == "DS1"
    assert s.width == 640 and s.height == 480
    assert s.split == "train"


def test_setters_and_yolo_lines():
    s = Image(
        uid="u1",
        dataset="DSX",
        subject_id="S1",
        image_path=Path("/dev/null"),
        width=100,
        height=80,
        split="val",
    )
    # Provide boxes directly (so ensure_boxes_from_masks won't need masks)
    s.set_box(Structure.DISC, LabelType.GT, BoundingBox(10, 20, 30, 60))
    s.set_box(Structure.CUP, LabelType.GT, BoundingBox(15, 25, 25, 45))

    lines = list(s.yolo_lines_2class(use_gt=True))
    # Two lines: class 0 (disc) then class 1 (cup)
    assert len(lines) == 2
    c0, c1 = lines
    assert c0.startswith("0 ")
    assert c1.startswith("1 ")
    # basic sanity on normalized ranges
    for line in (c0, c1):
        parts = [float(x) for x in line.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in parts)


def test_metrics_iou():
    s = Image(
        uid="u2",
        dataset="DSX",
        subject_id="S2",
        image_path=Path("/dev/null"),
        width=100,
        height=80,
    )
    s.set_box(Structure.DISC, LabelType.GT, BoundingBox(0, 0, 10, 10))
    s.set_box(Structure.DISC, LabelType.PRED, BoundingBox(5, 5, 15, 15))
    iou = s.disc_iou()
    assert iou is not None and 0.0 < iou < 1.0


def test_serde_roundtrip_json(tmp_path):
    s = Image(
        uid="u3",
        dataset="DS3",
        subject_id="S3",
        image_path=tmp_path / "img.png",
        width=200,
        height=100,
        split="test",
    )
    s.yolo_label_path = tmp_path / "img.txt"
    s.extras["foo"] = 42

    js = s.to_json()
    # sanity: it’s JSON
    json.loads(js)

    s2 = Image.from_json(js)
    assert s2.uid == s.uid
    assert s2.dataset == s.dataset
    assert s2.image_path == s.image_path
    assert s2.yolo_label_path == s.yolo_label_path
    assert s2.extras == s.extras


def test_set_split_validation():
    s = Image(
        uid="u4",
        dataset="DS4",
        subject_id="S4",
        image_path=Path("/dev/null"),
        width=10,
        height=10,
    )
    s.set_split("train")
    s.set_split(None)  # allowed
    with pytest.raises(ValueError):
        s.set_split("invalid")


# ====================== new test using real paths ======================

@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow is required for real-paths I/O and visualization")
def test_real_image_and_masks_visualize(real_paths):
    """Load your provided real image and disc/cup masks, compute BoundingBoxes, set on Image, and visualize."""
    img_path, disc_mask_path, cup_mask_path = real_paths

    # Load sizes from the real image (no synthetic data)
    im = PILImage.open(img_path).convert("RGB")
    W, H = im.size

    # Load masks and binarize (treat any nonzero as foreground)
    disc_arr = np.array(PILImage.open(disc_mask_path).convert("L")) > 0
    cup_arr = np.array(PILImage.open(cup_mask_path).convert("L")) > 0

    # Ensure mask shapes match the image size to avoid unintended resizing
    assert disc_arr.shape == (H, W), f"disc mask size {disc_arr.shape} != image size {(H, W)}"
    assert cup_arr.shape == (H, W), f"cup mask size {cup_arr.shape} != image size {(H, W)}"

    # Compute boxes from masks
    disc_box = BoundingBox.from_mask(disc_arr)
    cup_box = BoundingBox.from_mask(cup_arr)
    assert disc_box is not None and cup_box is not None

    # Create Image instance tied to your real file and set GT boxes
    s = Image.from_path(
        image_path=img_path,
        dataset="RealDS",
        subject_id="SUBJ_REAL",
        uid="uid_real",
        split="train",
        width=W,
        height=H,
    )
    s.set_box(Structure.DISC, LabelType.GT, disc_box)
    s.set_box(Structure.CUP, LabelType.GT, cup_box)

    # Ensure YOLO lines serialize correctly and are normalized
    yolo_lines = list(s.yolo_lines_2class(use_gt=True))
    assert len(yolo_lines) == 2
    for ln in yolo_lines:
        parts = [float(x) for x in ln.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in parts)

    # Visualize: draw both boxes on the real image and save to ./viz/
    draw = ImageDraw.Draw(im)
    # Convert half-open [x2,y2) to inclusive for PIL rectangles
    draw.rectangle([disc_box.x1, disc_box.y1, disc_box.x2 - 1, disc_box.y2 - 1], outline="red", width=3)
    draw.rectangle([cup_box.x1, cup_box.y1, cup_box.x2 - 1, cup_box.y2 - 1], outline="blue", width=3)

    viz_dir = Path("./viz")
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / "viz_real.png"
    print(f"[viz] wrote: {viz_path}")
    im.save(viz_path)

    # basic sanity: file written and same dimensions
    assert viz_path.exists()
    assert PILImage.open(viz_path).size == (W, H)