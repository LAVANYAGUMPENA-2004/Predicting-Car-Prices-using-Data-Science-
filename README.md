PREDICTING CAR PRICES USING DATA SCIENCE


Data Acquisition:
The dataset, containing car-related features (like brand, model, mileage, etc.), is loaded from a CSV file.

Data Exploration:
The data is explored using basic statistics, checks for missing values, and visualizations to understand its structure.

Data Preprocessing:
Categorical features are encoded using Label Encoding and One-Hot Encoding to convert text into numerical format. Feature scaling is applied to standardize the data, making it suitable for machine learning algorithms.

Model Training:
Several models, including Random Forest, Linear Regression, Extra Trees, and CatBoost, are trained to predict car prices. Each model is evaluated using performance metrics such as R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).

Hyperparameter Tuning:
The Random Forest model is fine-tuned using RandomizedSearchCV to optimize hyperparameters.

Model Deployment:
The best-performing model (CatBoost) is saved using pickle, allowing it to be reloaded later for making predictions without retraining.
