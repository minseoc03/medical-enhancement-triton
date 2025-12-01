from __future__ import annotations
import torch
import triton
import triton.language as tl

# ---- Kernel Hyperparameters ----
DEFAULT_TILE = 64
DEFAULT_BINS = 256

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W', 'tile_h', 'tile_w'],
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

    # synchronization (wait until all threads calculate histogram)
    #tl.debug_barrier()

    # ----- Read Histogram -----
    bins = tl.arange(0, n_bins)                    # [n_bins]
    hist = tl.load(tile_hist_base + bins)          # [n_bins], int32

    # ----- CLAHE Clipping -----
    over = tl.maximum(hist - clip_limit, 0)
    hist_clipped = hist - over
    extra = tl.sum(over, axis=0)                   # scalar

    # re-distribute the excess
    redist = extra / n_bins                       # scalar
    hist_redistrib = hist_clipped + redist         # [n_bins]

    # ----- CDF -----
    cdf = tl.cumsum(hist_redistrib, axis=0).to(tl.float32)   # [n_bins]
    n_valid = tl.sum(m_flat.to(tl.int32), axis=0)
    n_valid = tl.maximum(n_valid, 1)
    norm_factor = 1.0 / n_valid
    cdf_norm = cdf * norm_factor

    # ----- CDF LUT  -----
    lut_ptr_int = tile_hist_base + tile_id * n_bins
    tl.store(lut_ptr_int + bins, cdf_norm.to(tl.int32))
    #tl.debug_barrier()
    
    # ----- Pixel Mapping  -----
    target_ptrs = lut_ptr_int + q_flat
    out_flat = tl.load(target_ptrs, mask=m_flat, other=0.0)
    out_tile = tl.reshape(out_flat, (tile_h, tile_w))

    tl.store(out_ptr + hs * stride + ws, out_tile, mask=mask)

# @triton.jit
# def _clahe_map_interp_kernel(
#     in_ptr,          # float32[H, W]
#     out_ptr,         # float32[H, W]
#     cdf_ptr,         # float32[num_tiles, n_bins] (flat)
#     H, W,            # image height, width
#     stride,          # row stride
#     grid_h, grid_w,  # num of tiles
#     tile_h: tl.constexpr,
#     tile_w: tl.constexpr,
#     n_bins: tl.constexpr,
# ):
#     # ### 수정됨: 이 커널은 이미지의 모든 픽셀에 대해 실행됨
#     h = tl.program_id(0)   # [0, H)
#     w = tl.program_id(1)   # [0, W)
    
#     mask = (h < H) & (w < W)
#     if not mask:
#         return
    
#     # 1. 픽셀 값 로드 및 Quantization
#     val = tl.load(in_ptr + h * stride + w)
#     q = tl.minimum((val * (n_bins - 1)).to(tl.int32), n_bins - 1)

#     # 2. 인접한 4개 타일의 인덱스 및 가중치 계산
    
#     # 픽셀이 속한 그리드 좌표 (0 ~ grid_h/w)
#     gh = (h / tile_h)
#     gw = (w / tile_w)
    
#     # TL (Top-Left) 타일 좌표 (정수)
#     pid_tl_h = tl.floor(gh - 0.5).to(tl.int32)
#     pid_tl_w = tl.floor(gw - 0.5).to(tl.int32)
    
#     # 클램핑 (경계 처리)
#     pid_tl_h = tl.maximum(0, tl.minimum(pid_tl_h, grid_h - 2))
#     pid_tl_w = tl.maximum(0, tl.minimum(pid_tl_w, grid_w - 2))

#     # 나머지 3개 타일 인덱스
#     pid_tr_w = pid_tl_w + 1
#     pid_bl_h = pid_tl_h + 1
#     pid_br_h = pid_tl_h + 1
#     pid_br_w = pid_tl_w + 1

#     # 3. 인접 4개 타일의 CDF 값 읽기
    
#     # 타일의 중심 픽셀 좌표 (보간 기준점)
#     # center_h = (pid_tl_h + 0.5) * tile_h
#     # center_w = (pid_tl_w + 0.5) * tile_w
    
#     # 픽셀이 타일 내에서 얼마나 떨어져 있는지 정규화 (0~1)
#     # w_prime (0~1) : x축 가중치
#     w_prime = (w - (pid_tl_w + 0.5) * tile_w) / tile_w
#     h_prime = (h - (pid_tl_h + 0.5) * tile_h) / tile_h
    
#     # 경계 조건 처리
#     w_prime = tl.maximum(0.0, tl.minimum(1.0, w_prime))
#     h_prime = tl.maximum(0.0, tl.minimum(1.0, h_prime))
    
#     # 4개 CDF 로드 (Quantized value q를 인덱스로 사용)
#     # CDF_i = cdf_ptr[tile_id * n_bins + q]
    
#     # ⚠️ 마찬가지로 포인터 캐스팅 오류를 피하기 위해, 로드 시 float32 타입으로 캐스팅합니다.
#     # (이 코드는 lut_ptr = tile_hist_base.to(tl.float32)가 성공했다고 가정)
    
#     # TL (Top Left)
#     tile_id_tl = pid_tl_h * grid_w + pid_tl_w
#     cdf_tl_ptr = cdf_ptr + tile_id_tl * n_bins
#     cdf_tl = tl.load(cdf_tl_ptr + q)
    
#     # TR (Top Right)
#     tile_id_tr = pid_tl_h * grid_w + pid_tr_w
#     cdf_tr_ptr = cdf_ptr + tile_id_tr * n_bins
#     cdf_tr = tl.load(cdf_tr_ptr + q)

#     # BL (Bottom Left)
#     tile_id_bl = pid_bl_h * grid_w + pid_tl_w
#     cdf_bl_ptr = cdf_ptr + tile_id_bl * n_bins
#     cdf_bl = tl.load(cdf_bl_ptr + q)

#     # BR (Bottom Right)
#     tile_id_br = pid_bl_h * grid_w + pid_br_w
#     cdf_br_ptr = cdf_ptr + tile_id_br * n_bins
#     cdf_br = tl.load(cdf_br_ptr + q)


#     # 4. Bilinear Interpolation
#     # Output = (1-w')( (1-h')CDF_TL + h'CDF_BL ) + w'( (1-h')CDF_TR + h'CDF_BR )
    
#     weight_w_l = 1.0 - w_prime
#     weight_w_r = w_prime
#     weight_h_t = 1.0 - h_prime
#     weight_h_b = h_prime

#     # Top interpolation: LERP(CDF_TL, CDF_TR)
#     cdf_top = weight_w_l * cdf_tl + weight_w_r * cdf_tr
    
#     # Bottom interpolation: LERP(CDF_BL, CDF_BR)
#     cdf_bottom = weight_w_l * cdf_bl + weight_w_r * cdf_br

#     # Final vertical interpolation: LERP(cdf_top, cdf_bottom)
#     out_val = weight_h_t * cdf_top + weight_h_b * cdf_bottom

#     # 5. Store Result
#     tl.store(out_ptr + h * stride + w, out_val)

def clahe_triton(
    x01: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
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

    # 고정된 8 픽셀이 아니라, 이미지 크기를 그리드 수로 나누어 타일 크기 결정
    # 예: 512x512 이미지, (8,8) 그리드 -> 타일 크기 64x64
    grid_rows, grid_cols = tile_grid_size
    tile_h = (H + grid_rows - 1) // grid_rows
    tile_w = (W + grid_cols - 1) // grid_cols
    
    # Triton 커널 컴파일을 위해 2의 배수 등으로 맞추는 것이 좋지만, 
    # 여기서는 간단히 처리하고 커널 내 mask로 경계 처리함.
    # 단, tile 크기가 너무 작으면 연산이 불안정하므로 최소값 보정
    tile_h = max(tile_h, 16)
    tile_w = max(tile_w, 16)

    grid_h = triton.cdiv(H, tile_h)
    grid_w = triton.cdiv(W, tile_w)
    
    # ### 수정됨: Clip Limit 계산 로직 수정
    # 타일 내 총 픽셀 수
    pixels_per_tile = tile_h * tile_w
    base_per_bin = pixels_per_tile / n_bins
    
    # clip_per_bin이 0이 되는 것을 방지하기 위해 max(..., 1) 추가
    clip_per_bin = int(clip_limit * base_per_bin)
    clip_per_bin = max(clip_per_bin, 1) 

    #print(f"Debug: Tile Size=({tile_h}x{tile_w}), Clip Limit Val={clip_per_bin}")
    num_tiles = grid_h * grid_w
    cdf_buf = torch.empty(
        (B, num_tiles, n_bins),
        dtype=torch.float32, # Pass 1에서 float CDF를 저장해야 하므로 float32로 할당
        device=x01.device,
    )

    for b in range(B):
        inp = x01[b, 0]         # [H, W]
        out = y[b, 0]           # [H, W]
        stride = inp.stride(0) 
        grid = (grid_h, grid_w)
        
        _clahe_kernel[grid](
            inp, out, cdf_buf[b],
            H, W,
            stride,
            grid_w,
            tile_h, tile_w, # Python 변수로 계산된 큰 타일 사이즈 전달
            n_bins,
            clip_limit,
        )

        # _clahe_map_interp_kernel[(H, W)](
        #     inp, out, cdf_buf[b],
        #     H, W,
        #     stride,
        #     grid_h, grid_w,
        #     tile_h, tile_w,
        #     n_bins,
        # )

    return y