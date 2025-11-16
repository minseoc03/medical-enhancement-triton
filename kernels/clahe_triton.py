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

@triton.jit
def _clahe_kernel_atomic(
    in_ptr,          # float32[H, W]
    out_ptr,         # float32[H, W]
    hist_ptr,        # int32[num_tiles, n_bins] (flat)
    H, W,            # 이미지 높이, 너비
    stride,          # row stride (usually = W)
    grid_w,          # 타일 개수 (가로 방향)
    tile_h: tl.constexpr,
    tile_w: tl.constexpr,
    n_bins: tl.constexpr,
    clip_limit,      # per-tile clip limit (int)
):
    # ----- 타일 인덱스 -----
    pid_h = tl.program_id(0)   # [0, grid_h)
    pid_w = tl.program_id(1)   # [0, grid_w)

    h0 = pid_h * tile_h
    w0 = pid_w * tile_w

    # 이 타일의 1D tile index (히스토그램 버퍼에서 row 선택)
    tile_id = pid_h * grid_w + pid_w

    # ----- 타일 좌표 -----
    hs = h0 + tl.arange(0, tile_h)[:, None]   # [tile_h, 1]
    ws = w0 + tl.arange(0, tile_w)[None, :]   # [1, tile_w]
    mask = (hs < H) & (ws < W)                # [tile_h, tile_w]

    # ----- 타일 로드 -----
    tile = tl.load(in_ptr + hs * stride + ws,
                   mask=mask,
                   other=0.0)                 # [tile_h, tile_w], float32

    # ----- 픽셀값 -> bin 인덱스 -----
    # x in [0,1] 가정, 0~n_bins-1 로 quantize
    q = tl.minimum((tile * (n_bins - 1)).to(tl.int32), n_bins - 1)  # [tile_h, tile_w]

    # 평탄화
    N = tile_h * tile_w
    q_flat = tl.reshape(q, (tile_h * tile_w,))              # [N]
    m_flat = tl.reshape(mask, (tile_h * tile_w,))           # [N]

    # ----- 히스토그램 원자적 업데이트 -----
    # hist_ptr: int32[num_tiles, n_bins] 를 1D로 본 포인터
    # 이 타일의 히스토그램 row 시작 주소
    tile_hist_base = hist_ptr + tile_id * n_bins   # int32[n_bins] 의 시작

    # 각 픽셀이 쏠 bin 위치
    bin_ptrs = tile_hist_base + q_flat             # pointer[ N ]

    # 유효 픽셀만 +1
    tl.atomic_add(bin_ptrs, 1, mask=m_flat)

    # ----- 히스토그램 읽기 -----
    bins = tl.arange(0, n_bins)                    # [n_bins]
    hist = tl.load(tile_hist_base + bins)          # [n_bins], int32

    # ----- CLAHE 클리핑 -----
    # hist > clip_limit 인 부분의 초과량
    over = tl.maximum(hist - clip_limit, 0)
    hist_clipped = hist - over
    extra = tl.sum(over, axis=0)                   # scalar

    # 초과량을 모든 bin에 균등 분배 (나머지는 무시)
    redist = extra // n_bins                       # scalar
    hist_redistrib = hist_clipped + redist         # [n_bins]

    # ----- CDF -----
    cdf = tl.cumsum(hist_redistrib, axis=0).to(tl.float32)   # [n_bins]

    # ----- CDF LUT 적용 (gather 대신 브로드캐스트) -----
    # q_flat: [N], m_flat: [N]
    # bins: [n_bins]
    bins2 = bins[:, None]                  # [n_bins, 1]
    q2 = q_flat[None, :]                   # [1, N]
    m2 = m_flat[None, :]                   # [1, N]

    # weights[b, p] = 1(q[p] == b & valid_pixel)
    weights = ((q2 == bins2) & m2).to(tl.float32)    # [n_bins, N]
    cdf2 = cdf[:, None]                              # [n_bins, 1]

    # 각 픽셀 값 = sum_b weights[b, p] * cdf[b]
    out_flat = tl.sum(weights * cdf2, axis=0)        # [N]

    # 정규화: 유효 픽셀 개수로 나눔
    n_valid = tl.sum(m_flat.to(tl.int32), axis=0)
    n_valid = tl.maximum(n_valid, 1)                 # 0 나누기 방지
    norm = 1.0 / n_valid
    out_flat = out_flat * norm                       # [N]

    # 타일 형태로 되돌리기
    out_tile = tl.reshape(out_flat, (tile_h, tile_w))  # [tile_h, tile_w]

    # 결과 저장
    tl.store(out_ptr + hs * stride + ws,
             out_tile,
             mask=mask)


def clahe_triton_atomic(
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

    # OpenCV 스타일처럼 clip_limit는 평균 카운트의 배수
    # 평균 카운트 = (tile * tile) / n_bins
    base_per_bin = (tile * tile) / n_bins
    clip_per_bin = int(clip_limit * base_per_bin)

    for b in range(B):
        inp = x01[b, 0]         # [H, W]
        out = y[b, 0]           # [H, W]
        stride = inp.stride(0)  # 보통 W (row stride)

        # 히스토그램 버퍼: 각 타일마다 n_bins짜리 히스토그램
        num_tiles = grid_h * grid_w
        hist_buf = torch.zeros(
            (num_tiles, n_bins),
            dtype=torch.int32,
            device=x01.device,
        )

        _clahe_kernel_atomic[grid](
            inp, out, hist_buf,
            H, W,
            stride,
            grid_w,
            tile, tile,
            n_bins,
            clip_per_bin,
        )

    return y