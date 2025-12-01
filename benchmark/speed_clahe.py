import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from kernels.clahe_triton import clahe_triton


def cv2_clahe(x01_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8 = (np.clip(x01_np, 0, 1) * 255 + 0.5).astype(np.uint8)
    y = clahe.apply(u8).astype(np.float32) / 255.0
    return y


def bench_single(H=2048, W=2048, it=100):
    x = torch.rand(1, 1, H, W, device="cuda", dtype=torch.float32)
    # warm-up (Triton JIT + GPU Warming up)
    for _ in range(10):
        _ = clahe_triton(x)
    torch.cuda.synchronize()

    # Triton
    t0 = time.time()
    for _ in range(it):
        _ = clahe_triton(x, clip_limit=2.0, tile_grid_size=(8, 8))
    torch.cuda.synchronize()
    t1 = time.time()
    t_triton = (t1 - t0) * 1000.0 / it

    # OpenCV CPU
    x_np = x[0, 0].cpu().numpy()
    t0 = time.time()
    for _ in range(it):
        _ = cv2_clahe(x_np)
    t1 = time.time()
    t_cpu = (t1 - t0) * 1000.0 / it

    s = ssim(x_np, cv2_clahe(x_np), data_range=1.0)

    print(
        f"[{H}x{W}] Triton: {t_triton:.3f} ms   |   "
        f"OpenCV CPU: {t_cpu:.3f} ms   |   SSIM(input vs cpu): {s:.3f}"
    )

    return t_triton, t_cpu, s


def bench_multi():
    resolutions = [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (1536, 1536),
        (2048, 2048),
    ]
    iters = 50

    triton_times = []
    cpu_times = []
    ssims = []
    labels = []

    for (H, W) in resolutions:
        t_triton, t_cpu, s = bench_single(H, W, it=iters)
        triton_times.append(t_triton)
        cpu_times.append(t_cpu)
        ssims.append(s)
        labels.append(f"{H}x{W}")

    x = np.arange(len(resolutions))

    plt.figure(figsize=(8, 5))
    plt.plot(x, triton_times, marker="o", label="Triton CLAHE (GPU)")
    plt.plot(x, cpu_times, marker="o", label="OpenCV CLAHE (CPU)")
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Resolution (H x W)")
    plt.ylabel("Latency (ms)")
    plt.title("CLAHE Performance vs Resolution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("clahe_performance.png")


if __name__ == "__main__":
    bench_multi()
