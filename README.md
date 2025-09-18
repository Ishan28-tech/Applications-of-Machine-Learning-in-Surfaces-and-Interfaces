# Applications-of-Machine-Learning-in-Surfaces-and-Interfaces
PROJECT OVERVIEW

This project applies Machine Learning (ML) techniques to predict adsorption energy of different adsorbates on metal surfaces. Understanding adsorption behavior is critical for catalysis, surface chemistry, and material design.

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


