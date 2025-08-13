# Flycheap Airlines Ticket Price Prediction

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
   - All necessary libraries for data manipulation, visualization, and machine learning.
2. **Load Dataset**
   - Reads `Clean_Dataset.csv` for analysis.
3. **EDA & Visualization**
   - Explore data distributions and relationships.
4. **Preprocessing**
   - Handle missing values, duplicates, and encode categorical features.
5. **Feature Selection**
   - Drop irrelevant columns and prepare data for modeling.
6. **Model Training**
   - Split data, train Random Forest model, and evaluate.
7. **Model Saving & Loading**
   - Save trained model as `rf-cyber.pkl` and demonstrate loading for predictions.

---

## Usage

1. **Clone the repository.**
2. **Place your dataset (`Clean_Dataset.csv`) in the root directory.**
3. **Run `Flyhigh.ipynb` in Jupyter Notebook.**
4. **Follow the notebook cells for EDA, preprocessing, modeling, and prediction.**

---

## Saving and Loading the Model

> **Note:**  
> The `.pkl` file (`rf-cyber.pkl`) generated for the trained Random Forest model can be quite large due to the number of trees and depth.  
> If you encounter issues uploading or downloading the file from GitHub, consider compressing it or using cloud storage.

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

- `Flyhigh.ipynb`: Main notebook with full workflow.
- `Clean_Dataset.csv`: Raw dataset.
- `Indian_Airlines_cleaned_data.csv`: Cleaned and encoded dataset.
- `rf-cyber.pkl`: Saved Random Forest model (large file, see note above).

---

## Contact

For questions or collaboration, feel free to reach out on my personal website: https://datatrendx.com/ and GMail: praveen11x@gmail.com

---

**Disclaimer:**  
This project is for educational and demonstration purposes. For production use, further validation and optimization are
