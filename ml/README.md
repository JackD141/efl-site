# EFL Fantasy ML Setup

## Quick Setup

### Windows

```bash
# 1. Create venv
python -m venv ../venv

# 2. Activate venv
..\venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# Done! Open ml/predict_points.ipynb in VSCode
```

### macOS / Linux

```bash
# 1. Create venv
python -m venv ../venv

# 2. Activate venv
source ../venv/bin/activate

# 3. Install packages
pip install -r requirements.txt

# Done! Open ml/predict_points.ipynb in VSCode
```

## Auto-Kernel Detection (VSCode)

VSCode is configured (`.vscode/settings.json`) to automatically detect your venv and use it as the Jupyter kernel when you open `predict_points.ipynb`.

Just open the notebook in VSCode — no kernel selection needed!

## What's in the Notebook

- **predict_points.ipynb** — Complete regression pipeline
  - Load & explore 34 gameweeks of CSV data
  - Feature engineering (encode categorical vars)
  - Build training data (shift points forward for next-GW prediction)
  - Train 3 models: Linear Regression, Random Forest, XGBoost
  - Compare metrics & visualize results
  - Feature importance analysis

## Optional: Using Jupyter Lab Instead

If you prefer Jupyter Lab over VSCode:

```bash
# After pip install -r requirements.txt:
setup_kernel.bat    # Windows
./setup_kernel.sh   # macOS/Linux

# Then launch Jupyter Lab and select "EFL Fantasy (venv)" kernel
jupyter lab
```

## Requirements

See `requirements.txt` for all packages. Key ones:
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — ML models & metrics
- `xgboost` — gradient boosting
- `matplotlib`, `seaborn` — visualization
- `jupyterlab`, `ipykernel` — notebook environment
