import numpy as np
import pytest

from src.imgpipe.binary_mask_ref import BinaryMaskRef


def test_binary_mask_ref_array_only():
    arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    ref = BinaryMaskRef(array=arr)
    loaded = ref.load()
    assert loaded.dtype == np.bool_
    assert loaded.tolist() == [[False, True], [True, False]]
    d = ref.to_dict()
    assert d["path"] is None and d["has_array"] is True


def test_binary_mask_ref_missing_both_raises():
    ref = BinaryMaskRef()
    with pytest.raises(ValueError):
        _ = ref.load()


@pytest.mark.skipif(
    not any(__import__(m, globals(), locals(), [], 0) or True for m in ["PIL.Image", "imageio.v3"] if __import__(m.split('.')[0])),
    reason="PIL or imageio required to test path-based load"
)
def test_binary_mask_ref_path_load(tmp_path):
    p = tmp_path / "mask.png"
    try:
        from PIL import Image
        import numpy as np
        m = np.zeros((10, 10), dtype=np.uint8)
        m[2:8, 3:7] = 255
        Image.fromarray(m, mode="L").save(p)
    except Exception:
        import imageio.v3 as iio  # type: ignore
        import numpy as np
        m = np.zeros((10, 10), dtype=np.uint8)
        m[2:8, 3:7] = 255
        iio.imwrite(p, m)

    ref = BinaryMaskRef(path=p)
    arr = ref.load()
    assert arr.dtype == np.bool_
    assert arr.sum() == (6 * 4)