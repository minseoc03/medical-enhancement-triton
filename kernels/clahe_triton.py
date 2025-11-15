from __future__ import annotations
import torch
import triton
import triton.language as tl

# ---- Kernel Hyperparameters ----
DEFAULT_TILE = 16
DEFAULT_BINS = 256

@triton.jit
def _clahe_kernel(
    in_ptr, out_ptr,
    H : tl.constexpr, W : tl.constexpr, stride,
    tile_h : tl.constexpr, tile_w : tl.constexpr,
    n_bins : tl.constexpr, clip_limit : tl.constexpr
):
    #tile index
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    h0 = pid_h * tile_h
    w0 = pid_w * tile_w

    #tile coordinate
    hs = h0 + tl.arange(0, tile_h)[:, None]
    ws = w0 + tl.arange(0, tile_w)[None, :]
    mask = (hs < H) & (ws < W)

    #tile loading
    tile = tl.load(in_ptr + hs * stride + ws, mask = mask, other = 0.0)

    # ---- Histogram ----
    q = tl.minimum((tile * (n_bins - 1)).to(tl.int32), n_bins - 1)

    q_flat = tl.reshape(q, (tile_h * tile_w, ))
    m_flat = tl.reshape(mask, (tile_h * tile_w, ))

    bins = tl.arange(0, n_bins) # (n_bins)
    bins2 = bins[:, None] # (n_bins, 1)
    q2 = q_flat[None, :] # (1, N)
    m2 = m_flat[None, :] # (1, N)

    hist = tl.sum(((q2 == bins2) & m2).to(tl.int32), axis=1)  # [n_bins]

    # ---- Clipping ----
    over = tl.maximum(hist - clip_limit, 0)
    clipped = hist - over
    redist = tl.sum(clipped, axis = 0) // n_bins
    clipped = clipped + redist

    cdf = tl.cumsum(clipped, axis=0)

    # ---- Normalization ---
    area = tile_h * tile_w
    norm = 1.0 / area
    cdf2 = cdf[:, None].to(tl.float32)          # [n_bins, 1]
    weights = ((q2 == bins2) & m2).to(tl.float32)  # [n_bins, N]
    out_flat = tl.sum(weights * cdf2, axis=0) * norm  # [N]
    out_tile = tl.reshape(out_flat, (tile_h, tile_w))

    tl.store(out_ptr + hs * stride + ws, out_tile, mask=mask)

def clahe_triton(x01: torch.Tensor,
                 clip_limit: float = 2.0,
                 tile: int = DEFAULT_TILE,
                 n_bins: int = DEFAULT_BINS) -> torch.Tensor:
    """
    x01: [B,1,H,W] float32 in [0,1], CUDA tensor
    Returns: same shape, CLAHE applied per-image (batch naive loop)
    """
    assert x01.is_cuda and x01.dtype == torch.float32 and x01.ndim == 4 and x01.size(1) == 1
    B, _, H, W = x01.shape
    y = torch.empty_like(x01)

    grid = (triton.cdiv(H, tile), triton.cdiv(W, tile))

    for b in range(B):
        inp = x01[b, 0]
        out = y[b, 0]
        stride = inp.stride(0)

        _clahe_kernel[grid](
            inp, out,
            H, W, stride,
            tile, tile,
            n_bins, int(clip_limit * (tile * tile / n_bins))
        )

    return y
