import time
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from kernels.clahe_triton import clahe_triton, clahe_triton_atomic

def cv2_clahe(x01_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    u8 = (np.clip(x01_np,0,1)*255+0.5).astype(np.uint8)
    y = clahe.apply(u8).astype(np.float32)/255.0
    return y

def bench(H=2048, W=2048, it=100):
    x = torch.rand(1,1,H,W, device="cuda", dtype=torch.float32)

    # warm-up
    for _ in range(10):
        _ = clahe_triton_atomic(x)
    torch.cuda.synchronize()

    # Triton
    t0 = time.time()
    for _ in range(it):
        y = clahe_triton_atomic(x)
    torch.cuda.synchronize()
    t1 = time.time()
    t_triton = (t1 - t0) * 1000 / it

    # OpenCV CPU
    x_np = x[0,0].cpu().numpy()
    t0 = time.time()
    for _ in range(it):
        y_np = cv2_clahe(x_np)
    t1 = time.time()
    t_cpu = (t1 - t0) * 1000 / it

    s = ssim(x_np, y_np, data_range=1.0)

    print(f"Triton CLAHE: {t_triton:.3f} ms/it   |   OpenCV CPU: {t_cpu:.3f} ms/it   |   SSIM vs input: {s:.3f}")

if __name__ == "__main__":
    bench()
