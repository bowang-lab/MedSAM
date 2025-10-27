import math
import uuid

import numpy as np
import pytest

from src.utils import (
    gen_uid,
    ensure_bool_mask,
    xyxy_to_xc_yc_wh,
    xc_yc_wh_to_xyxy,
    read_image_size,
)


def test_gen_uid_length_and_uniqueness():
    a = gen_uid(12)
    b = gen_uid(12)
    assert isinstance(a, str) and isinstance(b, str)
    assert len(a) == 12 and len(b) == 12
    assert a != b


def test_ensure_bool_mask():
    arr = np.array([[0, 2], [3, 0]], dtype=np.uint8)
    out = ensure_bool_mask(arr)
    assert out.dtype == np.bool_
    assert out.shape == arr.shape
    assert out.tolist() == [[False, True], [True, False]]


def test_xyxy_yolo_roundtrip():
    x1, y1, x2, y2 = 10, 20, 30, 60
    xc, yc, w, h = xyxy_to_xc_yc_wh(x1, y1, x2, y2)
    xx1, yy1, xx2, yy2 = xc_yc_wh_to_xyxy(xc, yc, w, h)
    assert xx1 == pytest.approx(x1)
    assert yy1 == pytest.approx(y1)
    assert xx2 == pytest.approx(x2)
    assert yy2 == pytest.approx(y2)


@pytest.mark.skipif(
    not any(__import__(m, globals(), locals(), [], 0) or True for m in ["PIL.Image", "imageio.v3"] if __import__(m.split('.')[0])),
    reason="PIL or imageio required to test read_image_size"
)
def test_read_image_size(tmp_path):
    # Try PIL first; fall back to imageio if PIL not present
    W, H = 123, 77
    p = tmp_path / "tiny.png"
    try:
        from PIL import Image
        Image.new("RGB", (W, H)).save(p)
    except Exception:
        import imageio.v3 as iio  # type: ignore
        import numpy as np
        iio.imwrite(p, (np.zeros((H, W, 3)) + 255).astype(np.uint8))

    rw, rh = read_image_size(p)
    assert rw == W and rh == H