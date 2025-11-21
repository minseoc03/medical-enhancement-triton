from __future__ import annotations
import torch
import triton
import triton.language as tl

# ---- Kernel Hyperparameters ----
DEFAULT_TILE = 16
DEFAULT_BINS = 256

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['H', 'W'],
)
@triton.jit
def _clahe_kernel(
    in_ptr,          # float32[H, W]
    out_ptr,         # float32[H, W]
    hist_ptr,        # int32[num_tiles, n_bins] (flat)
    H, W,            # image height, width
    stride,          # row stride (usually = W)
    grid_w,          # num of tiles
    tile_h: tl.constexpr,
    tile_w: tl.constexpr,
    n_bins: tl.constexpr,
    clip_limit,      # per-tile clip limit (int)
):
    # ----- tile index -----
    pid_h = tl.program_id(0)   # [0, grid_h)
    pid_w = tl.program_id(1)   # [0, grid_w)

    h0 = pid_h * tile_h
    w0 = pid_w * tile_w

    # 1D tile index of current tile (selecting rows from histogram buffer)
    tile_id = pid_h * grid_w + pid_w

    # ----- tile coordinate -----
    hs = h0 + tl.arange(0, tile_h)[:, None]   # [tile_h, 1]
    ws = w0 + tl.arange(0, tile_w)[None, :]   # [1, tile_w]
    mask = (hs < H) & (ws < W)                # [tile_h, tile_w]

    # ----- loading tile -----
    tile = tl.load(in_ptr + hs * stride + ws,
                   mask=mask,
                   other=0.0)                 # [tile_h, tile_w], float32

    # ----- pixel values -> bin index -----
    # [0,1] -> [0, n_bins-1]
    q = tl.minimum((tile * (n_bins - 1)).to(tl.int32), n_bins - 1)  # [tile_h, tile_w]

    # flattening
    q_flat = tl.reshape(q, (tile_h * tile_w,))              # [N]
    m_flat = tl.reshape(mask, (tile_h * tile_w,))           # [N]

    # ----- Atomic Update -----
    tile_hist_base = hist_ptr + tile_id * n_bins   # start pointer of int32[n_bins]

    # now each pixel points to the corresponding bins
    bin_ptrs = tile_hist_base + q_flat             # pointer[ N ]

    # add one to only valid pixel (true in mask)
    tl.atomic_add(bin_ptrs, 1, mask=m_flat)

    # ----- Read Histogram -----
    bins = tl.arange(0, n_bins)                    # [n_bins]
    hist = tl.load(tile_hist_base + bins)          # [n_bins], int32

    # ----- CLAHE Clipping -----
    over = tl.maximum(hist - clip_limit, 0)
    hist_clipped = hist - over
    extra = tl.sum(over, axis=0)                   # scalar

    # re-distribute the excess
    redist = extra // n_bins                       # scalar
    hist_redistrib = hist_clipped + redist         # [n_bins]

    # ----- CDF -----
    cdf = tl.cumsum(hist_redistrib, axis=0).to(tl.float32)   # [n_bins]

    # ----- CDF LUT  -----
    # q_flat: [N], m_flat: [N]
    # bins: [n_bins]
    bins2 = bins[:, None]                  # [n_bins, 1]
    q2 = q_flat[None, :]                   # [1, N]
    m2 = m_flat[None, :]                   # [1, N]

    # weights[b, p] = 1(q[p] == b & valid_pixel)
    weights = ((q2 == bins2) & m2).to(tl.float32)    # [n_bins, N]
    cdf2 = cdf[:, None]                              # [n_bins, 1]
    out_flat = tl.sum(weights * cdf2, axis=0)        # [N]

    # normalization
    n_valid = tl.sum(m_flat.to(tl.int32), axis=0)
    n_valid = tl.maximum(n_valid, 1)                 # 0 나누기 방지
    norm = 1.0 / n_valid
    out_flat = out_flat * norm                       # [N]

    # back to tile shape
    out_tile = tl.reshape(out_flat, (tile_h, tile_w))  # [tile_h, tile_w]

    tl.store(out_ptr + hs * stride + ws,
             out_tile,
             mask=mask)


def clahe_triton(
    x01: torch.Tensor,
    clip_limit: float = 2.0,
    tile: int = DEFAULT_TILE,
    n_bins: int = DEFAULT_BINS,
) -> torch.Tensor:
    """
    x01: [B,1,H,W] float32 in [0,1], CUDA tensor
    Returns: same shape, CLAHE applied per-image (batch loop)
    """
    assert x01.is_cuda, "x01 must be CUDA tensor"
    assert x01.dtype == torch.float32, "x01 must be float32"
    assert x01.ndim == 4 and x01.size(1) == 1, "x01 must be [B,1,H,W]"

    B, _, H, W = x01.shape
    y = torch.empty_like(x01)

    grid_h = triton.cdiv(H, tile)
    grid_w = triton.cdiv(W, tile)
    grid = (grid_h, grid_w)
    base_per_bin = (tile * tile) / n_bins
    clip_per_bin = int(clip_limit * base_per_bin)

    for b in range(B):
        inp = x01[b, 0]         # [H, W]
        out = y[b, 0]           # [H, W]
        stride = inp.stride(0)  #  W

        # histogram buffer (each row represent each tile's histogram)
        num_tiles = grid_h * grid_w
        hist_buf = torch.zeros(
            (num_tiles, n_bins),
            dtype=torch.int32,
            device=x01.device,
        )

        _clahe_kernel[grid](
            inp, out, hist_buf,
            H, W,
            stride,
            grid_w,
            tile, tile,
            n_bins,
            clip_per_bin,
        )

    return y