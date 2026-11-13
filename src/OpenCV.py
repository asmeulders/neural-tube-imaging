import cv2
import numpy as np

def to_uint8(img):
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)   # normalize to [0, 1]
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def canny(img: np.ndarray, lower=50, upper=150, blur=True):
    img8 = to_uint8(img)
    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
    if blur:
        img8 = cv2.GaussianBlur(img8, (5, 5), 0)
    return cv2.Canny(img8, lower, upper)