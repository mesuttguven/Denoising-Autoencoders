"""
Binary -> RGB image, matching the original pipeline.

Original: three consecutive bytes -> one pixel; square image of side
int(sqrt(n_pixels)) + 1, remaining pixels black; saved as PNG; later read
with Keras load_img(target_size=(128,128)), whose default interpolation is
NEAREST.  Nearest-neighbour means each output pixel is ONE source pixel, so
for a 1 MB file roughly 95% of the bytes never reach the network.  We keep
that as the default for comparability and expose `resample` so an
area-averaging variant can be run as an ablation.
"""
import numpy as np
from PIL import Image

_RESAMPLE = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
             "area": Image.BOX}


def bytes_to_rgb(b: bytes, size: int = 128, resample: str = "nearest") -> np.ndarray:
    a = np.frombuffer(b, dtype=np.uint8)
    n_pix = len(a) // 3
    side = int(np.sqrt(n_pix)) + 1
    canvas = np.zeros((side * side, 3), dtype=np.uint8)
    if n_pix:
        canvas[:n_pix] = a[: n_pix * 3].reshape(-1, 3)
    img = Image.fromarray(canvas.reshape(side, side, 3), "RGB")
    img = img.resize((size, size), _RESAMPLE[resample])
    return np.asarray(img, dtype=np.uint8)


def byte_entropy(b: bytes) -> float:
    if not b:
        return 0.0
    c = np.bincount(np.frombuffer(b, np.uint8), minlength=256)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())
