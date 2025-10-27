import numpy as np
import pytest

from src.imgpipe.bounding_box import BoundingBox


def test_from_mask_none():
    m = np.zeros((10, 10), dtype=np.uint8)
    assert BoundingBox.from_mask(m) is None


def test_from_mask_basic():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:6, 3:8] = 1  # y:2..5, x:3..7
    box = BoundingBox.from_mask(m)
    assert box is not None
    assert box.x1 == 3
    assert box.y1 == 2
    assert box.x2 == 8  # exclusive max +1
    assert box.y2 == 6
    assert box.area() == pytest.approx(5 * 4)


def test_iou():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(5, 5, 15, 15)
    inter = 5 * 5
    union = a.area() + b.area() - inter
    assert a.iou(b) == pytest.approx(inter / union)


def test_yolo_norm_roundtrip():
    img_w, img_h = 100, 80
    b = BoundingBox(10, 20, 30, 60)
    xc, yc, w, h = b.to_yolo_norm(img_w, img_h)
    bb = BoundingBox.from_yolo_norm(xc, yc, w, h, img_w, img_h)
    assert bb.x1 == pytest.approx(b.x1)
    assert bb.y1 == pytest.approx(b.y1)
    assert bb.x2 == pytest.approx(b.x2)
    assert bb.y2 == pytest.approx(b.y2)


def test_clipped():
    b = BoundingBox(-10, -5, 110, 90).clipped(100, 80)
    assert b.x1 == 0 and b.y1 == 0 and b.x2 == 100 and b.y2 == 80