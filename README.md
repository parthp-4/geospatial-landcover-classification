# 🛰️ Deep Learning Workflow for Geo-Spatial Data Classification

> Pixel-level land cover classification of Sentinel-2 L2A satellite imagery using Random Forest, CNN, and U-Net deep learning models — trained on Rajasthan, validated on Maharashtra.

**Authors:** Parth Porwal · Yashvi Solanki  
**Research Internship:** BIT Mesra — Geospatial Deep Learning (2025–2026)

---

## 📌 Overview

This project implements an end-to-end unsupervised land cover classification pipeline for Sentinel-2 L2A satellite imagery. The pipeline requires **no manual pixel labelling** — pseudo-labels are generated automatically using SLIC superpixel segmentation and KMeans clustering, which then supervise four progressively powerful classifiers.

The model is trained on 20 tiles from **Jaipur-Ajmer & Bikaner (Rajasthan)** and validated on 10 geographically separated tiles from **Chandrapur (Maharashtra)** — testing cross-region generalisation under real domain shift.

---

## 🗺️ Land Cover Classes

| ID | Class | Colour |
|----|-------|--------|
| 0 | Hills / Rocky | 🟫 `(139, 69, 19)` |
| 1 | Crop Fields | 🟩 `(34, 139, 34)` |
| 2 | Fallow Land | 🟨 `(255, 215, 0)` |
| 3 | Water Body | 🟦 `(0, 0, 255)` |
| 4 | Sandy River | 🟧 `(255, 165, 0)` |
| 5 | Plantation | 🟢 `(0, 100, 0)` |
| 6 | Built-up | 🔴 `(255, 0, 0)` |

---

## 📁 Repository Structure

```
satellite_classification/
├── notebooks/
│   └── Implementation_6.ipynb      ← Main notebook (all 12 cells)
├── data/
│   ├── training_grids/             ← 20 × 512×512 GeoTIFF tiles (Rajasthan)
│   └── validation_grids/           ← 10 × 512×512 GeoTIFF tiles (Maharashtra)
├── models/
│   └── saved_models/               ← Trained model checkpoints (.h5)
├── outputs/
│   ├── visualizations/             ← All output figures (PNG)
│   ├── results/classified_tifs/    ← Classified GeoTIFFs per model
│   └── reports/                    ← CSV summary + Markdown report
├── .gitignore
└── README.md
```

> **Note:** `data/`, `models/saved_models/`, and `outputs/visualizations/` are excluded from Git (see `.gitignore`) due to file size. All outputs are stored in Google Drive.

---

## 🔬 Methodology

### Step 1 — Data Acquisition
Sentinel-2 L2A imagery downloaded from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) with cloud cover < 1%. Three scenes were acquired:

| Region | Tile ID | Date | Platform |
|--------|---------|------|----------|
| Jaipur-Ajmer, Rajasthan | T43RDK | 2025-05-01 | Sentinel-2C |
| Bikaner, Rajasthan | T43RCM | 2023-01-02 | Sentinel-2A |
| Chandrapur, Maharashtra | T44QLH | 2024-04-02 | Sentinel-2B |

### Step 2 — Tiling (SAGA GIS)
Large scenes tiled to **512×512 pixels** with 5-pixel overlap using SAGA GIS, yielding 20 training tiles and 10 validation tiles.

### Step 3 — Pseudo-Label Generation
Since no manual annotations exist, labels are generated automatically:
1. **SLIC** superpixel segmentation (400 segments, compactness 8)
2. Per-segment statistics computed (mean, std, median, percentiles, GLCM)
3. **KMeans** clustering (k=7) assigns each segment to a land cover class

### Step 4 — Feature Engineering (for RF & CNN)
A 13-dimensional feature vector is extracted per pixel:
- Raw NIR intensity
- Gradient magnitude (Sobel)
- Local Binary Pattern (LBP)
- Multi-scale local mean & std (5×5, 9×9, 15×15 kernels)
- GLCM texture (contrast, homogeneity, energy)

### Step 5 — Model Training & Evaluation

| Model | Input | Approach | Key Config |
|-------|-------|----------|-----------|
| Random Forest | Per-pixel (13D) | Classical ML | 120 trees, depth 18, balanced weights |
| CNN | 32×32 patch | Deep learning | 3 conv blocks, 12 epochs, batch 64 |
| U-Net (Scratch) | 512×512 image | Segmentation | 4-level encoder-decoder, 15 epochs |
| U-Net (Pretrained) | 512×512 image | Transfer learning | ResNet34 encoder, ImageNet weights |
| Ensemble | All predictions | Majority vote | Combines all 4 models |

---

## 📊 Results

### Class Distribution on Validation Set (% pixels)

| Class | RF | CNN | U-Net Scratch | U-Net Pretrained | Ensemble |
|-------|----|-----|--------------|-----------------|----------|
| Hills/Rocky | 20.91 | 26.69 | **98.68** | 8.23 | 68.75 |
| Crop Fields | 10.86 | 25.99 | 0.00 | **80.59** | 22.17 |
| Fallow Land | 13.09 | 18.19 | 1.32 | 4.30 | 4.62 |
| Water Body | 21.18 | 4.77 | 0.00 | 0.00 | 1.37 |
| Sandy River | 4.90 | 8.43 | 0.00 | 0.16 | 1.28 |
| Plantation | 23.06 | 6.44 | 0.00 | 0.18 | 1.06 |
| Built-up | 5.99 | 9.50 | 0.00 | 6.55 | 0.75 |

### Cross-Model Pixel Agreement

| Pair | Agreement |
|------|-----------|
| CNN vs Ensemble | 54.2% |
| U-Net Scratch vs Ensemble | 68.5% |
| RF vs CNN | 22.0% |
| U-Net Scratch vs U-Net Pretrained | 8.1% |

### Key Findings
- **Random Forest** identified all 7 classes in the geographically unseen validation region — most robust under data-scarce pseudo-label conditions
- **U-Net Scratch** collapsed to predicting Hills/Rocky for ~99% of pixels due to label imbalance and insufficient training data (17 tiles for a 5.4M parameter model)
- **U-Net Pretrained** (ResNet34/ImageNet) showed faster convergence (~55% training accuracy vs ~17% for scratch) but produced border artefacts due to domain mismatch between ImageNet natural images and single-band NIR satellite data
- Low cross-model agreement (8–68%) reflects fundamentally different inductive biases across the four approaches

---

## 🚀 Running the Notebook

### On Google Colab (recommended)

1. Open [`notebooks/Implementation_6.ipynb`](notebooks/Implementation_6.ipynb) in Colab
2. Run the **Colab Setup Cell** first — it mounts Drive, installs packages, and verifies data
3. Select **Runtime → Change runtime type → T4 GPU**
4. Click **Runtime → Run all**

### On Local Machine

```bash
# Clone the repo
git clone https://github.com/parthp-4/satellite_classification.git
cd satellite_classification

# Install dependencies
pip install tensorflow rasterio scikit-image scikit-learn opencv-python \
            segmentation-models tabulate seaborn tqdm

# Open notebook
jupyter notebook notebooks/Implementation_6.ipynb
```

> The notebook auto-detects whether it is running on Colab or locally and sets paths accordingly.

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow >= 2.16` | CNN and U-Net training |
| `segmentation-models` | Pretrained ResNet34 U-Net backbone |
| `rasterio` | GeoTIFF reading and georeferenced output |
| `scikit-learn` | Random Forest, KMeans, preprocessing |
| `scikit-image` | SLIC superpixels, LBP, GLCM |
| `opencv-python` | Image loading, resizing, Sobel features |
| `numpy / pandas` | Array and data operations |
| `matplotlib / seaborn` | Visualisation |
| `tqdm` | Progress bars |
| `tabulate` | Report formatting |

---

## 📋 Notebook Structure

| Cell | Description |
|------|-------------|
| Setup | Colab drive mount, package install, directory creation |
| Cell 1 | Environment setup, path detection, land cover scheme |
| Cell 2 | Image loading and normalisation |
| Cell 3 | SLIC superpixels + KMeans pseudo-label generation |
| Cell 4 | Random Forest training and validation predictions |
| Cell 5 | CNN patch classifier training and predictions |
| Cell 6 | U-Net scratch + pretrained ResNet34 training |
| Cell 7 | Ensemble voting + all comparison figures |
| Cell 8 | Export classified GeoTIFFs |
| Cell 9 | Quantitative analysis and CSV export |
| Cell 10 | Hero figure (report-ready, 200 dpi) |
| Cell 11 | Auto-generate Markdown report |
| Cell 12 | Final execution summary |

---

## 🗂️ Data Access

The training and validation GeoTIFF tiles are stored in Google Drive (not included in this repo due to file size):

- **Training data:** `data/training_grids/` — 20 tiles, Jaipur-Ajmer & Bikaner
- **Validation data:** `data/validation_grids/` — 10 tiles, Chandrapur
- **Raw Sentinel-2 scenes:** Downloaded from [Copernicus Data Space](https://dataspace.copernicus.eu) (Tile IDs: T43RDK, T43RCM, T44QLH)

Tiling was performed in **SAGA GIS** with settings: Tile Size 512×512 px, Overlap 5 px, Output format GeoTIFF.

---

## 📄 Citation / Reference

If referencing this work:

```
Porwal, P. & Solanki, Y. (2026). Deep Learning Workflow for Geo-Spatial Data Classification.
BIT Mesra Geospatial Deep Learning Research Internship.
GitHub: https://github.com/parthp-4/satellite_classification
```

---

## 📜 License

This project is for academic and research purposes. Sentinel-2 data is provided by ESA under the [Copernicus Open Access Hub Terms](https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf).
