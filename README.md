# 🔬 Applications of Machine Learning in Surfaces and Interfaces

> Predicting adsorption energies on metal surfaces using classical and advanced machine learning models — bridging computational chemistry and AI.

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Repository Structure](#-repository-structure)
- [ML Pipeline](#-ml-pipeline)
- [Baseline Models](#-phase-1-baseline-models)
- [Advanced Models](#-phase-2-advanced-models)
- [Model Comparison](#-model-comparison)
- [Visualizations](#-visualizations)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🧪 Project Overview

This project applies **Machine Learning (ML)** techniques to predict the **adsorption energy** of different adsorbates on metal surfaces. Adsorption energy is a fundamental quantity in surface science — it governs how strongly a molecule binds to a surface, directly impacting catalytic efficiency, corrosion behavior, and material design.

By training ML models on structural and chemical features, this project demonstrates that **data-driven methods can accurately estimate adsorption energies** — significantly reducing the need for expensive DFT (Density Functional Theory) calculations.

---

## 💡 Motivation

Understanding surface–adsorbate interactions is critical for:

- **Heterogeneous Catalysis** — designing better catalysts for industrial reactions
- **Electrochemistry** — modeling electrode–electrolyte interfaces
- **Corrosion Science** — predicting degradation mechanisms
- **Material Discovery** — screening new materials computationally

Traditional quantum chemistry methods (DFT) are computationally expensive and time-consuming. Machine learning models trained on existing data can **predict adsorption energies orders of magnitude faster**, enabling high-throughput screening.

---

## 📁 Repository Structure

```
Applications-of-ML-in-Surfaces-and-Interfaces/
│
├── data/                        # Dataset files (raw & processed)
├── models/                      # Saved trained models (.pkl)
│   ├── best_model.pkl           # Best baseline model
│   ├── best_xgb.pkl             # Best XGBoost model
│   └── best_cat.pkl             # CatBoost model
├── src/                         # Source code & utilities
├── training.ipynb               # Baseline model training notebook
├── advanced_models.ipynb        # Advanced model training notebook
├── Comparison_chart.jpg         # Model comparison bar chart
├── radar_chart.jpg              # Radar chart of model metrics
├── Requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ ML Pipeline

The full end-to-end pipeline is structured as follows:

```
Dataset Loading & Cleaning
        ↓
Feature Engineering
        ↓
Encoding (Label/OHE) & Scaling
        ↓
Baseline Model Training
(Linear Regression, Random Forest, Gradient Boosting)
        ↓
Advanced Model Training
(XGBoost, LightGBM, CatBoost, Neural Network)
        ↓
Evaluation (MAE, RMSE, R²)
        ↓
Model Saving (.pkl)
        ↓
Visualization & Interpretation
```

---

## 📊 Phase 1: Baseline Models

Four classical machine learning models were trained and evaluated to establish a performance baseline.

**Models Trained:**
- Linear Regression
- Random Forest Regressor
- Tuned Random Forest (via GridSearchCV)
- Gradient Boosting Regressor

### Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| **Linear Regression** | ~0.48 | ~0.71 | ~0.88 |
| **Random Forest** | ~0.44 | ~0.77 | ~0.86 |
| Tuned Random Forest | ~0.45 | ~0.78 | ~0.86 |
| Gradient Boosting | ~0.68 | ~0.91 | ~0.81 |

> ✅ **Best Baseline:** Linear Regression and Random Forest offered the best balance of accuracy and simplicity.

**Notebook:** [`training.ipynb`](training.ipynb)

---

## 🚀 Phase 2: Advanced Models

Building on the baseline, four advanced models were implemented to capture complex nonlinear relationships in the dataset.

**Models Implemented:**
- **XGBoost Regressor** — gradient boosted trees with regularization
- **LightGBM Regressor** — fast, leaf-wise boosting
- **CatBoost Regressor** — handles categorical features natively
- **Neural Network (MLPRegressor)** — multi-layer perceptron

### Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Notes |
|---|---|---|---|---|
| **XGBoost** | ~0.554 | ~0.930 | ~0.809 | 🏆 Best overall — excellent fit, high R² |
| **CatBoost** | ~0.618 | ~0.953 | ~0.800 | Very close second; strong generalization |
| **Neural Network (MLP)** | ~0.611 | ~0.974 | ~0.791 | Decent but may benefit from deeper tuning |
| LightGBM | ~1.054 | ~1.560 | ~0.463 | Underperformed — sensitive to dataset size |

> 🏆 **Best Advanced Model: XGBoost**
>
> XGBoost outperformed all others due to:
> - Superior handling of nonlinear relationships
> - Built-in L1/L2 regularization preventing overfitting
> - Optimized gradient boosting for high accuracy
> - Strong robustness on smaller datasets

**Notebook:** [`advanced_models.ipynb`](advanced_models.ipynb)

---

## 📈 Model Comparison

### Baseline vs Advanced — At a Glance

| Phase | Best Model | R² Score |
|---|---|---|
| Baseline | Linear Regression | ~0.88 |
| Advanced | XGBoost | ~0.81 |

> Note: While the baseline Linear Regression achieved a higher R² on this dataset, the XGBoost model provides far stronger generalization and scalability for larger, more complex datasets.

Visual comparisons are available in the repository:

- **`Comparison_chart.jpg`** — Bar chart comparing MAE/RMSE across all models
- **`radar_chart.jpg`** — Radar chart visualizing multi-metric model performance

---

## 📉 Visualizations

| Chart | Description |
|---|---|
| `radar_chart.jpg` | Multi-axis radar chart comparing MAE, RMSE, and R² for all models |
| `Comparison_chart.jpg` | Bar chart comparing model performance across evaluation metrics |

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ishan28-tech/Applications-of-Machine-Learning-in-Surfaces-and-Interfaces.git
cd Applications-of-Machine-Learning-in-Surfaces-and-Interfaces
```

### 2. Install Dependencies

```bash
pip install -r Requirements.txt
```

**Dependencies include:**

```
pandas, numpy, scikit-learn, seaborn, matplotlib, joblib
```

> For advanced models, also install:
> ```bash
> pip install xgboost lightgbm catboost
> ```

### 3. Run the Notebooks

```bash
jupyter notebook training.ipynb          # Baseline models
jupyter notebook advanced_models.ipynb   # Advanced models
```

---

## 🔍 Usage

### Test the Best Baseline Model

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load("models/best_model.pkl")

# Create a sample input
demo = pd.DataFrame([{
    "Element": "Cu",
    "Adsorbate Smiles": "O=O",
    "h": 1, "k": 1, "l": 1,
    "Surface Shift": 0
}])

print("Predicted Adsorption Energy:", model.predict(demo)[0])
```

### Test the Best Advanced Model (XGBoost)

```python
import joblib
import pandas as pd

# Load XGBoost model
model = joblib.load("models/best_xgb.pkl")

# Create a sample input
demo = pd.DataFrame([{
    "Element": "Cu",
    "Adsorbate Smiles": "O=O",
    "h": 1, "k": 1, "l": 1,
    "Surface Shift": 0
}])

print("Predicted Adsorption Energy (XGBoost):", model.predict(demo)[0])
```

---

## 🔮 Future Scope

This project lays a strong foundation for future research directions:

- **Expand the dataset** to multi-element alloy surfaces for broader coverage
- **Add richer descriptors** — DFT-derived features, atomic fingerprints, d-band center
- **Graph Neural Networks (GNNs)** for atomic-level structural learning
- **Hybrid DFT + ML pipelines** for high-accuracy, low-cost simulations
- **Web-based prediction tool** — a deployable app for adsorption energy estimation
- **Active learning** to iteratively improve model performance with fewer labels

---

## 🏁 Conclusion

This project demonstrates that advanced ML models — especially **XGBoost** — can reliably predict adsorption energies with high accuracy and at a fraction of the computational cost of DFT. Such models have the potential to **accelerate research** in:

- Surface science
- Heterogeneous catalysis
- Interface engineering
- High-throughput material discovery

The repository is designed to be **modular, extensible, and research-grade** — ready for further experimentation and deployment.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/Ishan28-tech">Ishan</a> · Materials Science × Machine Learning</sub>
</div>
