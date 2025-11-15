# Medical Image Enhancement with Triton & DICOM Pipeline

A high-performance medical image enhancement pipeline built with
**Triton GPU kernels**, **PyTorch**, and **DICOM preprocessing tools**.\
This project aims to accelerate medical image enhancement workflows
(CT/MRI) using custom GPU kernels and optimized data loading, targeting
real-time performance.

## Features

-   **DICOM I/O & Parsing**
    -   Lightweight DICOM loader using `pydicom`
    -   Windowing, normalization, resizing, and denoising utilities
    -   Batch preprocessing pipeline for CT/MRI slices
-   **GPU-Accelerated Enhancement Operators**
    -   Triton-based custom kernels (e.g., contrast stretching, CLAHE
        variants)
    -   Highly parallel operations optimized for NVIDIA GPUs (tested on
        RTX 5070 Ti)
    -   Easy-to-extend kernel interface
-   **End-to-End Enhancement Pipeline**
    -   CPU → GPU I/O handoff
    -   Preprocess → Enhancement → Postprocess
    -   Configurable via Python modules
-   **Developer-Friendly Structure**
    -   Modular code layout (`dicom_io.py`, `preprocess.py`, `kernels/`,
        etc.)
    -   Clear API for integrating additional enhancement algorithms
    -   Works in Python 3.12 (PyTorch 2.5+)

## Project Structure

    project-root/
    │
    ├── dicom_io.py        # DICOM reading utilities
    ├── preprocess.py      # Preprocessing pipeline
    ├── kernels/
    │   ├── clahe.py
    │   ├── utils.py
    │   └── ...
    ├── main.py
    ├── sample_data/
    └── README.md

## Installation

### 1) Clone

    git clone https://github.com/<your-username>/<repo>.git
    cd <repo>

### 2) Install dependencies

    pip install -r requirements.txt
    pip install triton

## Usage

### Full pipeline

    python main.py --input <dicom-folder> --output out/

### Preprocess only

    python preprocess.py --input sample.dcm

### Run kernel

    python kernels/clahe.py --image path/to/image.png

## Tech Stack

-   Python 3.12
-   PyTorch 2.5+
-   Triton
-   pydicom
-   NumPy / OpenCV

## Roadmap

-   [ ] Triton CLAHE kernel
-   [ ] Add enhancement ops
-   [ ] Metadata visualization
-   [ ] CPU vs GPU benchmarks
-   [ ] GUI preview
-   [ ] pip package

## License

MIT License.
