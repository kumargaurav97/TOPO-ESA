# TOPO-ESA: TOPOLOGY-DRIVEN EEG SIGNAL ANALYSIS FOR COGNITIVE WORKLOAD RECOGNITION

This repository provides code for applying topology-driven framework to EEG signals (for cognitive worload recognition under various physical activity levels) signals using Takens Embedding and Vietoris–Rips Persistence. It extracts topology-driven interpretable features(Total Peristence and Persistent Entopy) from topological summaries (persistence diagrams and barcodes).

# For validation WAUC dataset [1] is utilized.

[1] Isabela Albuquerque, Abhishek Tiwari, Mark Parent, Raymundo Cassani, Jean-Franc¸ois Gagnon, Daniel Lafond, S´ebastien Tremblay, and Tiago H Falk, “Wauc: a multi-modal database for mental workload assessment under physical activity,” Frontiers in Neuroscience, vol.
 14, pp. 549524, 2020.

EEG data were collected from 48 participants(25 male, 13 female; meanage 27.4±6.6years) using an eight-channel Enobiowireless headset (Neuroelectrics, Spain) during cognitive tasks performed at three physical activity levels: no movement, medium (treadmill 3km/h or cycling 50rpm), and high(treadmil l5km/h or cycling 70rpm). CWL was induced with the MATB-II task at low and high difficulty. Each 18-minute session included baseline, warm up, task execution, and subjective evaluation, with workload ratings from NASA-TLX. In this study, only activity intensity is considered (treadmill and cycling combined), yielding six experimental conditions across two workload and three physical activity(PW) levels for binary CWL classification. EEG signals were recorded at 500Hz from eight electrodes(P3,T9,AF7,FP1,FP2,AF8,T10,P4) with Fpz and Nz as references. A fourth-order zero-phase Butterworthfilter (1–35Hz) and additional preprocessing were applied to remove noise and motion artifacts.The cleaned signals were then segmented into 4.5-second non-overlapping epochs for topology-drivenfeatureextraction.

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
