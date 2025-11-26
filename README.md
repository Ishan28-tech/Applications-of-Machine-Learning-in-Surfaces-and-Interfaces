# Applications-of-Machine-Learning-in-Surfaces-and-Interfaces
PROJECT OVERVIEW

This project applies Machine Learning (ML) techniques to predict adsorption energy of different adsorbates on metal surfaces. Understanding adsorption behavior is critical for catalysis, surface chemistry, and material design.

🔥BASELINE MODELS

We trained baseline ML models using structural and chemical features to estimate adsorption energy, and compared their performance.


MODELS TRAINED

We trained and evaluated four baseline models:

Linear Regression
Random Forest Regressor
Tuned Random Forest (GridSearchCV)
Gradient Boosting Regressor


EVALUATION METRICS
| Model               | MAE ↓  | RMSE ↓ | R² ↑   |
| ------------------- | ------ | ------ | ------ |
| Linear Regression   | ~0.48  | ~0.71   | ~0.88 |
| Random Forest       | ~0.44  | ~0.77   | ~0.86 |
| Tuned Random Forest | ~0.45  | ~0.78   | ~0.86 |
| Gradient Boosting   | ~0.68  | ~0.91   | ~0.81 |

Linear Regression and Random Forest gave the best balance of accuracy and simplicity


TEST THE SAVED MODEL:
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

demo = pd.DataFrame([{
    "Element": "Cu",
    "Adsorbate Smiles": "O=O",
    "h": 1, "k": 1, "l": 1,
    "Surface Shift": 0
}])

print("Predicted Energy:", model.predict(demo)[0])

🔥ADVANCED MODELS 

After completing the baseline model, we further improved the project by training a set of advanced machine learning models to achieve higher accuracy and stronger generalization capabilities.

The following models were implemented:

XGBoost Regressor

LightGBM Regressor

CatBoost Regressor

Neural Network (MLPRegressor)

These models are more powerful than classical ML techniques and can capture complex nonlinear relationships in the adsorption dataset.

Model	MAE ↓	RMSE ↓	R² ↑	Performance Summary
XGBoost	0.554	0.930	0.809	Best overall model — excellent fit, low error, high R² (≈81% variance explained).
CatBoost	0.618	0.953	0.800	Very close second; strong predictive power, slightly higher error than XGBoost.
Neural Network (MLP)	0.611	0.974	0.791	Performs decently but slightly less stable; might improve with tuning or deeper layers.
LightGBM	1.054	1.560	0.463	Underperformed — possibly due to insufficient parameter tuning or sensitivity to small dataset size.

✔ Best Advanced Model: XGBoost

XGBoost outperformed all other models because:

It handles nonlinearity extremely well

Built-in regularization prevents overfitting

Optimized tree-boosting improves accuracy

Robust with smaller datasets

📉 VISUALIZATIONS
1. Predicted vs Actual (XGBoost)

Add your saved plot here:

![Predicted vs Actual - XGBoost](assets/xgb_scatter.png)

2. Advanced Model Comparison (Bar Chart)
![Advanced Model Comparison](assets/adv_model_comparison.png)

📁 NEW FILES ADDED (Advanced Model Section)

Inside your repository, the advanced model implementation includes:

src/advanced_models.ipynb → Jupyter Notebook for full advanced model training

models/best_xgb.pkl → Best advanced model saved

models/best_cat.pkl → CatBoost saved model

(Optional) visualizations inside assets/ folder

This keeps your project structured and scalable.

🏗 WORKFLOW SUMMARY

The full ML pipeline now includes:

Dataset Loading & Cleaning

Feature Engineering

Encoding & Scaling

Baseline Model Training

Advanced Model Training (XGBoost, CatBoost, NN)

Model Evaluation (MAE, RMSE, R²)

Model Saving

Visualization & Interpretation

⚙️ TESTING THE ADVANCED MODEL

You can test the XGBoost model the same way as baseline:

import joblib
import pandas as pd

model = joblib.load("models/best_xgb.pkl")

demo = pd.DataFrame([{
    "Element": "Cu",
    "Adsorbate Smiles": "O=O",
    "h": 1, "k": 1, "l": 1,
    "Surface Shift": 0
}])

print("Predicted Energy:", model.predict(demo)[0])

🔮 FUTURE SCOPE OF THE PROJECT

Expand dataset to multi-element alloy surfaces

Add more physical descriptors (DFT-derived, atomic fingerprints)

Apply Graph Neural Networks (GNNs) for atomic-level learning

Develop a web-based app for adsorption energy prediction

Integrate ML with DFT pipelines for hybrid, high-accuracy workflows

🏁 Final Note

The project now has:

✔ Baseline Models
✔ Advanced Models
✔ Evaluation Metrics
✔ Saved Models
✔ Visualizations
✔ Clean Repository Structure
✔ Ready-to-Extend Framework for Research

You can now confidently present this as a research-grade ML project in materials science.


