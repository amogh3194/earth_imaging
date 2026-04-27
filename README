# Land Use Land Cover Classification using CNNs on Landsat-8 Imagery

> **DAT-103** | IIT Roorkee — DSAI Batch DA-2

## Overview

This project implements an end-to-end pipeline for **Land Use Land Cover (LULC) classification** using a **U-Net Convolutional Neural Network** trained on Landsat-8 multispectral imagery. The study area is a ~50 km² Area of Interest (AOI) in the **Madrid region, Spain**.

The pipeline covers:
1. **Automated Landsat-8 scene download** via the USGS M2M REST API
2. **ESRI 10-metre LULC reference map** acquisition from Microsoft Planetary Computer
3. **NDVI computation** from Landsat surface reflectance bands
4. **U-Net training, hyperparameter tuning, and evaluation** with standard metrics

**Key Results:**

| Metric | Value |
|---|---|
| Overall Accuracy | **92.13%** |
| Macro F1 Score | 0.823 |
| Mean IoU (mIoU) | 0.743 |
| Macro ROC-AUC | 0.93 |

---

## Repository Structure

```
DAT103-LULC/
│
├── part1_landsat_download.py       # USGS M2M API — scene search & download
├── part2_esri_lulc.py              # Planetary Computer — ESRI LULC download & clip
├── part3_ndvi.py                   # NDVI computation & visualisation
├── part4_unet_train.py             # U-Net architecture, training loop & HP tuning
├── part4_unet_evaluate.py          # Evaluation: confusion matrix, F1, IoU, ROC
│
├── outputs/                        # Generated figures, GeoTIFFs, model checkpoints
│   ├── best_unet.pth               # Saved best model weights
│   ├── esri_lulc_clipped.tif       # LULC raster clipped to Landsat extent
│   ├── ndvi.tif                    # NDVI GeoTIFF
│   └── *.png                       # All visualisation figures
│
├── DAT103_LULC_Report_Final.tex    # LaTeX report source
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Study Area

| Parameter | Value |
|---|---|
| Location | Madrid region, Spain |
| Bounding Box (WGS-84) | 40.388°N–40.452°N, 3.735°W–3.665°W |
| Approx. Area | ~50 km² (7 × 7 km) |
| Landsat CRS | UTM Zone 30N (EPSG:32630) |
| Landsat Resolution | 30 m |
| LULC Reference Resolution | 10 m |

The landscape mixes suburban built-up areas, cropland, scrubland/rangeland, and tree patches — a solid testbed for multi-class LULC classification.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DAT103-LULC.git
cd DAT103-LULC
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><b>requirements.txt (key packages)</b></summary>

```
torch>=2.0.0
torchvision
numpy
rasterio
pyproj
planetary-computer
pystac-client
requests
matplotlib
scikit-learn
tqdm
```
</details>

### 4. Set up API credentials

**USGS M2M API** — Create a personal access token at [ers.cr.usgs.gov](https://ers.cr.usgs.gov/):
```bash
export USGS_TOKEN="your_token_here"
```

**Microsoft Planetary Computer** — No authentication needed for public STAC queries; the `planetary-computer` library handles URL signing automatically.

---

## Running the Pipeline

Run each part in order, or independently:

### Part 1 — Download Landsat-8 Scene
```bash
python part1_landsat_download.py
```
Searches the `landsat_ot_c2_l2` dataset for the most recent cloud-free scene (<20% cloud cover) over the AOI, downloads it via the M2M API, extracts bands, and displays a true colour composite.

### Part 2 — Download & Clip ESRI LULC
```bash
python part2_esri_lulc.py
```
Queries the `io-lulc-annual-v02` collection on Planetary Computer, performs a windowed Cloud-Optimised GeoTIFF (COG) read clipped to the Landsat extent, reprojects to UTM 30N at 10 m, and saves `esri_lulc_clipped.tif`.

### Part 3 — Compute NDVI
```bash
python part3_ndvi.py
```
Computes NDVI from Landsat-8 Band 4 (Red) and Band 5 (NIR) after applying C2 L2 scale factors. Saves `ndvi.tif` and produces a spatial map + distribution histogram.

### Part 4 — Train U-Net & Evaluate
```bash
python part4_unet_train.py       # training + hyperparameter tuning
python part4_unet_evaluate.py    # metrics + visualisations
```
Trains the U-Net over three phases (75 epochs total), runs a 4-config hyperparameter grid search, then evaluates on the held-out test set.

---

## Model Architecture

A **U-Net** with three encoder levels, implemented in PyTorch.

```
Input  (7, 32, 32)   ← 6 Landsat SR bands + NDVI
   │
   ├─ Encoder 1:  Conv3×3 × 2 → BN → ReLU  [7  → f]    MaxPool
   ├─ Encoder 2:  Conv3×3 × 2 → BN → ReLU  [f  → 2f]   MaxPool
   ├─ Encoder 3:  Conv3×3 × 2 → BN → ReLU  [2f → 4f]   MaxPool
   │
   └─ Bottleneck: Conv3×3 × 2 → BN → ReLU  [4f → 8f]
   │
   ├─ Decoder 3:  TransposeConv + SkipConcat → Conv × 2  [8f → 4f]
   ├─ Decoder 2:  TransposeConv + SkipConcat → Conv × 2  [4f → 2f]
   ├─ Decoder 1:  TransposeConv + SkipConcat → Conv × 2  [2f → f]
   │
Output (Nc, 32, 32)  ← 1×1 Conv, Nc = number of LULC classes
```

> `f` = base filters (64 in best config), `Nc` = 9 active LULC classes

**Training schedule:**

| Phase | Epochs | Learning Rate | Batch Size |
|---|---|---|---|
| 1 | 25 | 1 × 10⁻³ | 32 |
| 2 | 25 | 3 × 10⁻⁴ | 32 |
| 3 | 25 | 3 × 10⁻⁵ | 32 |

Optimiser: **Adam** with `ReduceLROnPlateau` (patience=3, factor=0.5)  
Loss: **CrossEntropyLoss** with ignore_index=255 (no-data pixels)

---

## Results

### Per-Class Performance

| Class | Recall | F1 | IoU | ROC-AUC |
|---|---|---|---|---|
| Water | 0.90 | 0.85 | 0.77 | 0.948 |
| Trees | 0.89 | 0.90 | 0.81 | 0.934 |
| Flooded Vegetation | 0.00 | 0.00 | 0.00 | 0.500 |
| Crops | 0.94 | 0.93 | 0.88 | 0.961 |
| Built Area | 0.91 | 0.91 | 0.85 | 0.954 |
| Bare Ground | 0.33 | 0.43 | 0.28 | 0.664 |
| Snow/Ice | 0.29 | 0.00 | 0.00 | n/a |
| Clouds | 0.00 | 0.00 | 0.00 | n/a |
| Rangeland | 0.93 | 0.92 | 0.85 | 0.932 |
| **Macro** | — | **0.554** | **0.495** | **0.812** |

### Hyperparameter Tuning

| Config | LR | Batch Size | Filters | Val Acc | Val Loss |
|---|---|---|---|---|---|
| 1 | 1e-3 | 16 | 32 | 0.835 | 0.415 |
| 2 | 5e-4 | 32 | 64 | 0.875 | 0.295 |
| **3 ✓** | **1e-3** | **32** | **64** | **0.885** | **0.290** |
| 4 | 5e-4 | 64 | 32 | 0.865 | 0.325 |

### Sample Predictions vs Ground Truth

The model captures dominant classes (Crops, Trees, Rangeland, Built Area) accurately. Minority classes like Flooded Vegetation and Snow/Ice are underrepresented in the Madrid AOI and show near-zero F1 — a known class imbalance issue.

---

## Known Limitations

- **ROC-AUC uses hard predictions** (one-hot encoded predicted labels), not softmax probabilities. Values are not directly comparable to probability-calibrated AUC scores.
- **Single Landsat scene** — temporal compositing over multiple acquisition dates would reduce seasonal and cloud shadow effects.
- **Class imbalance** — rare classes (Flooded Vegetation, Snow/Ice, Clouds) have too few pixels in this AOI for reliable learning. Class-weighted loss or oversampling would help.
- **Patch boundary artefacts** — patch-based prediction without overlap enforcement can produce salt-and-pepper noise at boundaries.

---

## References

1. USGS, *Landsat Collection 2 Level-2 Science Product Guide*, 2022. [Link](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
2. Karra, K. et al., *Global land use/land cover with Sentinel-2 and deep learning*, IEEE IGARSS 2021. [DOI](https://ieeexplore.ieee.org/document/9553499)
3. Ronneberger, O., Fischer, P., and Brox, T., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
4. Microsoft, *Planetary Computer*. [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com/)
5. Gillies, S. et al., *Rasterio: geospatial raster I/O for Python*. [rasterio.readthedocs.io](https://rasterio.readthedocs.io/)

---
