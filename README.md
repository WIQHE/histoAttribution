# HistoAttribution Pipeline

This module provides a pipeline for histopathology image analysis using Vision Transformers (ViT) and graph-based methods. The workflow is organized into several stages, each with dedicated scripts, notebooks, and data structures.

## 1. Data Organization
- **`bach_labels.csv`**: Contains mapping of case IDs, slide IDs, and class labels (0: Benign, 1: InSitu, 2: Invasive, 3: Normal).
- **`bach_patches/`**: Intended for storing extracted image patches, masks, and stitched images (currently empty in this snapshot).
- **`data/Photos/`**: Source images organized by subtype.

## 2. Patch Extraction
- Patches are extracted from whole-slide images using a sliding window approach (see code in `graphGeneration.ipynb`).
- Each patch is further subdivided into 16 subpatches (4x4 grid), enabling localized analysis.

## 3. Feature Extraction (ViT)
- The pipeline uses a fine-tuned Vision Transformer (`finetuned_vit_breastcancer.pth`) to extract features from each subpatch.
- For each subpatch, all other regions are blurred to focus the model's attention.
- Features are collected via a forward hook on the ViT model's head.

## 4. Graph Construction
- **Delaunay Graphs**: Nodes represent subpatches; edges are formed using Delaunay triangulation based on subpatch centroids. Edge attributes combine spatial distance and feature similarity.
- **k-NN Graphs**: Each node connects to its k nearest neighbors (k=4 by default), with similar edge attributes.
- Graphs are saved as `.pt` files in `graphs_delaunay/` and `graphs_knn/`.
- Metadata for each graph (paths, labels, accuracy) is intended to be saved in `d_meta.csv` and `k_meta.csv` (currently empty).

## 5. Notebooks
- **`graphGeneration.ipynb`**: Main pipeline for patch extraction, feature extraction, and graph construction.
- **`GAT.ipynb`**: Implements and evaluates Graph Attention Networks (GAT) and other GNNs on the generated graphs. Includes training, evaluation, and visualization.
- **`HnEAttribution.ipynb`**: Handles data preparation and attribution analysis for H&E-stained images.

## 6. Results and Visualization
- **Heatmaps**: Visual overlays (e.g., `green_heatmap_overlay.png`, `red_misleading_heatmap.png`) show model attributions and predictions.
- **Graph Files**: `.pt` files in `graphs_delaunay/` and `graphs_knn/` are ready for downstream GNN analysis.

## 7. Usage
1. Prepare your data in `data/Photos/` and update `bach_labels.csv` as needed.
2. Run `graphGeneration.ipynb` to extract patches, compute features, and build graphs.
3. Use `GAT.ipynb` for graph-based classification and attribution.
4. Visualize results using the provided heatmaps and graph files.

## 8. Requirements
- Python, PyTorch, Torch Geometric, OpenCV, scikit-learn, torchvision, matplotlib, seaborn, tqdm, PIL
- See code cells for specific import statements and package usage.

---

For questions or contributions, please refer to the code comments and notebook documentation.

## 9. GDC Diffusion Variants
- Utilities in `util/` support topology diffusion (GDC) and the new feature diffusion (GDC-Feat: X_s = S X).
- Data preparation exposes `raw`, `gdc`, and `gdc_feat` loader splits.
- Runners (`util/runner.py`) compare baseline vs `+GDC` vs `+GDC-Feat` for GAT/GCN/SAGE/PNA and save artifacts under `*_gdc_ckpts/`.
- Use `aggregate_gnn_gdc_report.py` to consolidate metrics and plots into an HTML report.
