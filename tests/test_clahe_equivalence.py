import numpy as np, torch, cv2
from kernels.clahe_triton import clahe_triton, clahe_triton_atomic

def cv2_clahe(x01_np):
    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    u8 = (np.clip(x01_np,0,1)*255+0.5).astype(np.uint8)
    y = op.apply(u8).astype(np.float32)/255.0
    return y

def test_basic_equivalence():
    torch.manual_seed(0)
    x = torch.rand(1,1,256,256, device="cuda", dtype=torch.float32)
    y_tr = clahe_triton(x, clip_limit=2.0, tile=16, n_bins=256)[0,0].cpu().numpy()
    y_cv = cv2_clahe(x[0,0].cpu().numpy())
    mse = np.mean((y_tr - y_cv)**2)
    assert mse < 1e-3, f"MSE too high: {mse}"

if __name__ == "__main__":
    test_basic_equivalence()