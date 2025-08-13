# Indian Airlines Ticket Price Prediction

Predict airline ticket prices using machine learning and data analytics.

---

## Overview

This repository contains a complete workflow for predicting airline ticket prices using Python and machine learning. The project covers data exploration, visualization, preprocessing, feature engineering, model training, evaluation, and deployment.

---

## Features

- **Exploratory Data Analysis (EDA):**
  - Summary statistics, null value checks, duplicate detection.
  - Visualizations: histograms, bar charts, scatter plots, box plots, categorical value counts.
- **Data Preprocessing:**
  - Dropping unnecessary columns.
  - Label encoding for categorical variables.
  - Exporting cleaned dataset for reproducibility.
- **Feature Engineering:**
  - Selection of relevant features for modeling.
- **Model Development:**
  - Train/test split using scikit-learn.
  - Random Forest Regressor with hyperparameter tuning (`n_estimators=150`, `max_depth=30`, `min_samples_leaf=5`, `min_samples_split=30`).
- **Model Evaluation:**
  - Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score.
- **Model Deployment:**
  - Model serialization with pickle for production use.
  - Example of loading and making predictions with the saved model.

---

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

## Workflow

1. **Import Libraries**
2. **Load Dataset**
3. **EDA & Visualization**
4. **Preprocessing**
5. **Feature Selection**
6. **Model Training**
7. **Model Saving & Loading**

---

## Usage

1. **Clone the repository.**
2. **Place your dataset (`Clean_Dataset.csv`) in the root directory.**
3. **Run `Flyhigh.ipynb` in Jupyter Notebook.**
4. **Follow the notebook cells for EDA, preprocessing, modeling, and prediction.**

---

## Saving and Loading the Model

> **Special Note:**  
> The trained Random Forest model (`rf-cyber.pkl`) can be very large and may exceed GitHub's upload limits.  
> **The `.pkl` file is NOT included in this repository.**  
> To use the model, you must generate and save it yourself by running the notebook.

**To save the model from your notebook:**
```python
import pickle
model = rf  # Your trained RandomForestRegressor object
file_path = 'rf-cyber.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(model, file)
```

**To load the model for prediction:**
```python
import pickle
file_path = 'rf-cyber.pkl'
with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)
```

**Example prediction:**
```python
# Replace the values below with your feature set
loaded_model.predict([[3,2,4,2,0,5,1,2.17,1]])
```

---

## Files

- `Flight.ipynb`: notebook with full workflow.
- `Airlines-ticket-price-prediction.ipynb`: notebook with full workflow.
- `Cleaned_Dataset.csv`: Raw dataset.
- `Cleaned_Dataset.csv`: Cleaned and encoded dataset.
- `rf-cyber.pkl`: **Not included** (see note above).

---

## Contact

For questions or collaboration, feel free to reach out via personal website: https://datatrendx.com/ or Gmail: praveen11x@gmail.com.

---

**Disclaimer:**  
This project is for educational and demonstration purposes. For production use, further validation and optimization
