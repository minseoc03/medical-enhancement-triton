import pydicom
import numpy as np
import torch

def load_dicom(path: str):
    """Read DICOM file and return pydicom Dataset object"""
    ds = pydicom.dcmread(path)
    return ds

def get_hu_image(ds):
    """Convert CT pixel values into Houndsfield (HU) unit"""
    img = ds.pixel_array.astype(np.float32)
    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    hu_img = slope * img + intercept
    return hu_img

def window_image(img, center, width):
    """Using window center and width, adjust intensity range"""
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(img, low, high)
    return windowed

def normalize(img):
    """0-1 Normalization"""
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)

def dicom_to_tensor(path, window_center=40, window_width=400, device="cpu"):
    """
    Get DICOM file path, apply HU conversion, windowing, and normaliztion,
    return torch.Tensor
    """
    ds = load_dicom(path)
    img = get_hu_image(ds)
    img = window_image(img, window_center, window_width)
    img = normalize(img)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor.to(device), ds
