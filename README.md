# 24S1_wine_quality_classification

This repository contains a compact, reproducible project for wine quality classification using physicochemical features.  
It includes a Jupyter notebook with the full workflow, a concise PDF report, and provided train/test CSV files.

## Repository Structure
```text
.
├── wine_quality_analysis.ipynb         # Main analysis and custom KNN implementation
├── wine_quality_analysis_report.pdf    # Report with methods, results, and figures
├── winequality-train.csv               # Training split
└── winequality-test.csv                # Test split
```

## Objective
Predict wine quality as a binary label (low vs. high). The project:
- Implements a from-scratch K-Nearest Neighbors (KNN) classifier (Euclidean distance, majority vote).
- Evaluates normalization strategies: Min–Max scaling and Standardization.
- Compares against a Gaussian Naive Bayes (GNB) baseline.

## Data
- Input: Provided `winequality-train.csv` and `winequality-test.csv`.
- Features: Standard physicochemical measurements (e.g., acidity, sugar, sulfur dioxide, pH, alcohol).
- Target: `quality` (0 = low, 1 = high).
- Note: Fit normalization parameters on train only; apply to both train and test.

## Environment
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Packages: numpy, pandas, matplotlib, scikit-learn

### Quick install
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run
1. Start Jupyter
2. Open `wine_quality_analysis.ipynb`.
3. Run all cells in order. The notebook will:
   - Load and validate the provided train/test data.
   - Apply normalization (Min–Max and Standardization).
   - Train/evaluate the custom KNN across settings and optionally run GNB.
   - Report accuracy and show key plots.

## Methods (Summary)
- KNN (k from 1 to 20 explored in the notebook)
- Distance: Euclidean
- Ties: broken by smallest average distance among tied classes
- Scaling: Min–Max to [0,1] and Z-score Standardization
- Baseline: Gaussian Naive Bayes using scikit-learn

## Key Findings (Brief)
- Normalization materially improves KNN performance compared with raw features.
- The best KNN setting outperforms GNB on this split.
- Scaling is important due to heterogeneous feature ranges.

For exact metrics and figures, see `wine_quality_analysis_report.pdf`.

## Reproducibility Notes
- Use the provided split without reshuffling.
- Randomness in KNN is minimal; any randomness (e.g., during tie handling) is controlled in the notebook.
- All steps are contained in a single notebook for clarity.

## Project Notes
- The notebook favors vectorized NumPy operations for speed.
- Evaluation uses accuracy on the fixed test split.
- Code is intentionally kept simple for instructional clarity.

