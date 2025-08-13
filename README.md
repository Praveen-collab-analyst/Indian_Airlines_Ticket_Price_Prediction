# Flycheap Airlines Ticket Price Prediction

Predict airline ticket prices using machine learning and data analytics.

## Overview

This project leverages Python and machine learning to analyze and predict airline ticket prices based on various flight features. The workflow includes data exploration, visualization, preprocessing, feature engineering, model training, evaluation, and deployment.

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

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

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

## Usage

1. Clone the repository.
2. Place your dataset (`Clean_Dataset.csv`) in the root directory.
3. Run `Flyhigh.ipynb` in Jupyter Notebook.
4. Follow the notebook cells for EDA, preprocessing, modeling, and prediction.

## Example Prediction

After training and saving the model, you can load it and predict ticket prices for new data:

```python
import pickle
loaded_model = pickle.load(open('rf-cyber.pkl', 'rb'))
# Example input: [airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left]
prediction = loaded_model.predict([[3,2,4,2,0,5,1,2.17,1]])
print("Predicted Price:", prediction)
```

## Files

- `Flyhigh.ipynb`: Main notebook with full workflow.
- `Clean_Dataset.csv`: Raw dataset.
- `Indian_Airlines_cleaned_data.csv`: Cleaned and encoded dataset.
- `rf-cyber.pkl`: Saved Random Forest model.

## Contact

For questions or collaboration, feel free to reach out.
Personal Website: https://datatrendx.com/
Email: praveen11x@gmail.com

---

**Note:** This project is for educational and demonstration purposes. For production use, further validation and optimization are recommended.
