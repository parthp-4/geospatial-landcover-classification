# Pixel-Level Land Cover Classification — Report Summary

## 1. Study Overview

| Parameter | Value |
|-----------|-------|
| Satellite | Sentinel-2 L2A |
| Training region | Jaipur-Ajmer & Bikaner, Rajasthan |
| Validation region | Chandrapur, Maharashtra |
| Training tiles | 20 |
| Validation tiles | 10 |
| Image size | 512 × 512 pixels |
| Land cover classes | 7 |

## 2. Land Cover Classes

| ID | Class | RGB Colour |
|----|-------|------------|
| 0 | Hills/Rocky | (139, 69, 19) |
| 1 | Crop Fields | (34, 139, 34) |
| 2 | Fallow Land | (255, 215, 0) |
| 3 | Water Body | (0, 0, 255) |
| 4 | Sandy River | (255, 165, 0) |
| 5 | Plantation | (0, 100, 0) |
| 6 | Built-up | (255, 0, 0) |

## 3. Methodology

### Pseudo-label Generation
- SLIC superpixel segmentation (n_segments=400, compactness=8)
- KMeans clustering (k=7) on per-segment statistics (mean, std, median, percentiles, GLCM contrast/homogeneity/energy)

### Models

**Random Forest** — pixel-level classifier trained on hand-crafted features (intensity, gradient magnitude, LBP, multi-scale local statistics, GLCM summary). 120 trees, max depth 18, balanced class weights.

**CNN Patch Classifier** — 32×32 patch classification with 3-block encoder (32→64→128 filters) + GlobalAveragePooling + Dense head. Trained for 12 epochs, batch 64.

**U-Net Segmentation** — 4-level encoder-decoder with skip connections. Combined Dice + Categorical-CE loss. Trained for 15 epochs, batch 4.

**Ensemble** — majority-vote across all trained models.

## 4. Cross-Model Pixel Agreement

| Comparison | Agreement |
|------------|----------|
| RF vs CNN | 20.63% |
| RF vs U-Net Scratch | 20.83% |
| RF vs U-Net Pretrained | 11.22% |
| RF vs Ensemble | 36.11% |
| CNN vs U-Net Scratch | 25.03% |
| CNN vs U-Net Pretrained | 24.34% |
| CNN vs Ensemble | 50.86% |
| U-Net Scratch vs U-Net Pretrained | 9.14% |
| U-Net Scratch vs Ensemble | 70.44% |
| U-Net Pretrained vs Ensemble | 31.14% |

## 5. Class Distribution Summary (%)

| Class       |   CNN |   Ensemble |    RF |   U-Net Pretrained |   U-Net Scratch |
|:------------|------:|-----------:|------:|-------------------:|----------------:|
| Built-up    | 15.9  |       1.08 |  5.99 |               5.78 |            0    |
| Crop Fields | 25.03 |      20.61 | 10.86 |              77.94 |            0    |
| Fallow Land | 12.12 |       3.53 | 13.09 |               6.68 |            1.34 |
| Hills/Rocky | 25.5  |      70.66 | 20.91 |               9.22 |           98.66 |
| Plantation  |  7.29 |       1.08 | 23.06 |               0.22 |            0    |
| Sandy River |  7.73 |       1.31 |  4.9  |               0.16 |            0    |
| Water Body  |  6.43 |       1.72 | 21.18 |               0    |            0    |

## 6. Output Artefacts

- Classified GeoTIFFs: `/content/drive/MyDrive/satellite_classification/outputs/results/classified_tifs`
- Figures: `/content/drive/MyDrive/satellite_classification/outputs/visualizations`
- CSV summary: `/content/drive/MyDrive/satellite_classification/outputs/reports/class_distribution_summary.csv`