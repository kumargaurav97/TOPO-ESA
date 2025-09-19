# Topological Data Analysis (TDA) on EEG Signals

This repository provides code for applying Topological Data Analysis (TDA) to EEG (and other time series) signals using Takens Embedding and Vietoris–Rips Persistence. It extracts persistence diagrams and barcodes, saving them for visualization and further analysis.

## 📂 Repository Structure

```
.
├── main_wauc.py        # Main Python script for processing signals and generating plots
├── TDA_Plots/          # Folder containing generated persistence diagrams & barcodes
└── README.md           # Documentation (this file)
```

## 🚀 Features

- **Takens Embedding:** Converts 1D time-series windows into higher-dimensional trajectories.
- **Vietoris–Rips Persistence:** Computes homology in dimensions H0 (connected components) and H1 (loops).
- **Persistence Diagrams:** Plotted using `giotto-tda` + `plotly`.
- **Persistence Barcodes:** Plotted using a custom `matplotlib` function.
- **Automatic Plot Saving:** All plots are saved into the `tda_plots/` directory with subject/session labels.
- **Configurable Parameters:** Easily adjust embedding dimension, time delay, and stride.

## 📊 Example Outputs

**Persistence Diagram**  
Shows birth–death times of topological features.

**Persistence Barcode**  
Shows feature lifetimes across filtration values.

_All plots are saved in `TDA_Plots/` as `.png` files with proper subject/session labeling._


**Minimal requirements:**

- numpy
- matplotlib
- plotly>=6.1.1
- kaleido (for saving Plotly figures as PNGs)
- giotto-tda
- scikit-learn
- tqdm

## ▶️ Usage

1. **Prepare your data:**  
   Ensure signals are available in the `labeled_data` format (subject → sessions → signals, labels).

2. **Run the script:**

   ```bash
   python main_wauc.py
   ```

3. **View results:**  
   Generated plots will appear in the `TDA_Plots/` folder.

## 📌 Notes

- Short sessions are either padded or dropped depending on configuration.
- Embedding parameters can be tuned inside the script.
- If you see checkered backgrounds in saved figures, ensure `facecolor="white"` is set in matplotlib and that Plotly + Kaleido versions are compatible.

---
