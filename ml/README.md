# EFL Fantasy ML Setup

## Quick Setup

### Windows

```bash
# 1. Create and activate venv
python -m venv ../venv
..\venv\Scripts\activate

# 2. Install packages
pip install -r requirements.txt

# 3. Register venv as Jupyter kernel
setup_kernel.bat

# 4. Launch Jupyter
jupyter lab
```

### macOS / Linux

```bash
# 1. Create and activate venv
python -m venv ../venv
source ../venv/bin/activate

# 2. Install packages
pip install -r requirements.txt

# 3. Register venv as Jupyter kernel
chmod +x setup_kernel.sh
./setup_kernel.sh

# 4. Launch Jupyter
jupyter lab
```

## Kernel Selection

After step 3, when you open `predict_points.ipynb` in Jupyter Lab:

1. Look for the **kernel selector** in the top-right corner (shows "Select Kernel")
2. Click it and choose **"EFL Fantasy (venv)"**
3. All cells will now run using your venv's Python environment

## What's in the Notebook

- **predict_points.ipynb** — Complete regression pipeline
  - Load & explore 34 gameweeks of CSV data
  - Feature engineering (encode categorical vars)
  - Build training data (shift points forward for next-GW prediction)
  - Train 3 models: Linear Regression, Random Forest, XGBoost
  - Compare metrics & visualize results
  - Feature importance analysis

## Requirements

See `requirements.txt` for all packages. Key ones:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — ML models & metrics
- `xgboost` — gradient boosting
- `matplotlib`, `seaborn` — visualization
- `jupyterlab`, `ipykernel` — notebook environment
