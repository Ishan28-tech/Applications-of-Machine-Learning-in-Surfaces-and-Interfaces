📘 Applications of Machine Learning in Surfaces and Interfaces
🚀 PROJECT OVERVIEW

This project applies Machine Learning (ML) techniques to predict adsorption energy of different adsorbates on metal surfaces.
Understanding adsorption behavior is crucial for:

Catalysis

Surface chemistry

Material design

🔥 BASELINE MODELS

Baseline ML models were trained using structural and chemical features to estimate adsorption energy and compare performance.

Models Trained

Linear Regression

Random Forest Regressor

Tuned Random Forest (GridSearchCV)

Gradient Boosting Regressor

📊 BASELINE MODEL EVALUATION METRICS
Model	MAE ↓	RMSE ↓	R² ↑
Linear Regression	~0.48	~0.71	~0.88
Random Forest	~0.44	~0.77	~0.86
Tuned Random Forest	~0.45	~0.78	~0.86
Gradient Boosting	~0.68	~0.91	~0.81

✔ Linear Regression and Random Forest gave the best balance of accuracy and simplicity.

🧪 TEST THE SAVED BASELINE MODEL
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

🔥 ADVANCED MODELS

After completing the baseline model, a set of advanced ML models was trained for improved accuracy and generalization.

Models Implemented

XGBoost Regressor

LightGBM Regressor

CatBoost Regressor

Neural Network (MLPRegressor)

These models capture more complex nonlinear patterns in the adsorption dataset.

📊 ADVANCED MODEL EVALUATION METRICS
Model	MAE ↓	RMSE ↓	R² ↑	Performance Summary
XGBoost	~0.554	~0.930	~0.809	⭐ Best model — excellent fit, low error, high R²
CatBoost	~0.618	~0.953	~0.800	Strong performance, slightly higher error than XGBoost
Neural Network (MLP)	~0.611	~0.974	~0.791	Good performance, may improve with tuning
LightGBM	~1.054	~1.560	~0.463	Underperformed — sensitive to dataset size
✔ Best Advanced Model: XGBoost

Why XGBoost wins:

Handles nonlinear interactions extremely well

Built-in regularization prevents overfitting

Optimized tree boosting = better accuracy

Robust even with small datasets

📉 VISUALIZATIONS
Predicted vs Actual (XGBoost)

(Insert your image)

Predicted vs Actual – XGBoost

Advanced Model Comparison (Bar Chart)

(Insert your image)

Advanced Model Comparison

📁 NEW FILES ADDED (Advanced Model Section)

Your repository now includes:

src/advanced_models.ipynb → Full advanced model pipeline

models/best_xgb.pkl → Best model saved

models/best_cat.pkl → CatBoost model

(Optional) /assets → Plots and visualizations

This keeps the project organized and scalable.

🏗 WORKFLOW SUMMARY

The complete ML pipeline includes:

Dataset Loading & Cleaning

Feature Engineering

Encoding & Scaling

Baseline Model Training

Advanced Model Training (XGBoost, CatBoost, NN)

Model Evaluation (MAE, RMSE, R²)

Model Saving

Visualization & Interpretation

⚙️ TESTING THE ADVANCED MODEL
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

Include DFT-derived physical descriptors

Apply Graph Neural Networks (GNNs)

Build a web-based prediction app

Integrate DFT + ML hybrid pipelines

🧾 CONCLUSION

This project demonstrates that advanced ML models—especially XGBoost—can reliably predict adsorption energies with high accuracy.
Such models significantly reduce computational cost and accelerate research in:

Surface science

Catalysis

Interface engineering

Materials discovery

🔧 FUTURE WORK

Extend dataset to multi-element alloy surfaces

Integrate GNNs for atomic-level understanding

Develop hybrid DFT + ML methodologies

Deploy a web-based prediction tool
