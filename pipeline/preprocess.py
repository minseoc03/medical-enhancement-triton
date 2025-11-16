from __future__ import annotations
import numpy as np
from typing import Tuple, Literal, Optional, Union, Dict
from kernels.clahe_triton import clahe_triton_atomic

# Optinal torch import
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

import cv2

ArrayLike = Union[np.ndarray, "torch.Tensor"]
# -----------------------------
# Utilities
# -----------------------------
def _ensure_numpy_01(x: ArrayLike) -> np.ndarray:
    """
    Accepts numpy [H,W] or torch [B,1,H,W]/[1,H,W] in any dtype.
    Returns numpy float32 array in [0,1], shape [H,W].
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.ndim == 4:
            x = x[0,0]
        elif x.ndim == 3:
            x = x[0]
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image, get shape {x.shape}")
    x = x.astype(np.float32)

    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax - vmin > 1.5:
        x = (x - vmin) / (vmax - vmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    return x

def _to_uint8(x01 : np.ndarray) -> np.ndarray:
    return np.clip(x01 * 255.0 + 0.5, 0, 255).astype(np.uint8)

def _from_uint8(u8 : np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255.0)

def intensity_clip(x01: ArrayLike, low_q: float = 0.0, high_q: float = 1.0) -> np.ndarray:
    """
    Percentile-based clipping in [0,1]. Use e.g., (0.005, 0.995) to trim outliers.
    """
    x = _ensure_numpy_01(x01)
    if not (0.0 <= low_q < high_q <= 1.0):
        raise ValueError("low_q/high_q must satisfy 0<=low<high<=1")
    lo, hi = np.quantile(x, [low_q, high_q])
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return x

# -----------------------------
# Baseline CPU operators
# -----------------------------
def clahe(
    x01: ArrayLike,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    OpenCV CLAHE baseline. Input/Output in [0,1] float32.
    """
    x = _ensure_numpy_01(x01)
    u8 = _to_uint8(x)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y = op.apply(u8)
    return _from_uint8(y)


def gaussian_denoise(
    x01: ArrayLike,
    ksize: int = 3,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Gaussian blur denoiser. ksize must be odd.
    """
    x = _ensure_numpy_01(x01)
    u8 = _to_uint8(x)
    y = cv2.GaussianBlur(u8, (ksize, ksize), sigma)
    return _from_uint8(y)


def bilateral_denoise(
    x01: ArrayLike,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 3.0,
) -> np.ndarray:
    """
    Edge-preserving denoise. More expensive than Gaussian.
    d: pixel neighborhood diameter (odd). If <=0, computed from sigmaSpace.
    """
    x = _ensure_numpy_01(x01)
    u8 = _to_uint8(x)
    y = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return _from_uint8(y)


def gamma_correction(x01: ArrayLike, gamma: float = 1.0) -> np.ndarray:
    """
    Intensity transform: y = x^(gamma). gamma<1 brightens, >1 darkens mids.
    """
    x = _ensure_numpy_01(x01)
    gamma = max(gamma, 1e-6)
    return np.power(x, gamma, dtype=np.float32)


def unsharp_mask(
    x01: ArrayLike,
    ksize: int = 3,
    sigma: float = 0.0,
    amount: float = 1.0,
) -> np.ndarray:
    """
    Unsharp sharpening: y = x + amount * (x - blur(x))
    """
    x = _ensure_numpy_01(x01)
    u8 = _to_uint8(x)
    blur = cv2.GaussianBlur(u8, (ksize, ksize), sigma)
    # x + amount * (x - blur)
    y = cv2.addWeighted(u8, 1.0 + amount, blur, -amount, 0)
    return _from_uint8(y)


# -----------------------------
# Composite pipeline
# -----------------------------
def enhance_pipeline(
    x01: ArrayLike,
    *,
    denoise: Literal["none", "gaussian", "bilateral"] = "gaussian",
    denoise_params: Optional[Dict] = None,
    do_clahe: bool = True,
    do_triton_clahe: bool = False,
    clahe_params: Optional[Dict] = None,
    do_unsharp: bool = True,
    unsharp_params: Optional[Dict] = None,
    do_clip: bool = True,
    clip_params: Optional[Dict] = None,
    do_gamma: bool = False,
    gamma: float = 1.0,
    return_numpy: bool = True,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Default sequence: denoise -> CLAHE -> unsharp -> intensity_clip -> gamma(optional)
    All steps operate in [0,1]; returns same format as requested (numpy or torch).
    """
    x = _ensure_numpy_01(x01)

    # Denoise
    denoise_params = denoise_params or {}
    if denoise == "gaussian":
        x = gaussian_denoise(x, **denoise_params)
    elif denoise == "bilateral":
        x = bilateral_denoise(x, **denoise_params)
    elif denoise == "none":
        pass
    else:
        raise ValueError("denoise must be one of {'none','gaussian','bilateral'}")

    # CLAHE (local contrast)
    if do_clahe:
        clahe_params = clahe_params or {}
        x = clahe(x, **clahe_params)
    if do_triton_clahe:
        x = clahe_triton_atomic(x)

    # Unsharp (edge emphasis)
    if do_unsharp:
        unsharp_params = unsharp_params or {}
        x = unsharp_mask(x, **unsharp_params)

    # Intensity clipping (robust range)
    if do_clip:
        clip_params = clip_params or {}
        x = intensity_clip(x, **clip_params)

    # Gamma (optional)
    if do_gamma and abs(gamma - 1.0) > 1e-6:
        x = gamma_correction(x, gamma=gamma)

    if return_numpy:
        return x
    else:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available but return_numpy=False requested.")
        return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]