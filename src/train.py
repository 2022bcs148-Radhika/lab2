import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso  # Changed from LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Dataset (Wine Quality)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# 2. Pre-processing & Feature Selection [cite: 38]
X = df.drop('quality', axis=1)
y = df['quality']
# You can also experiment with test_size (e.g., 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model - Experiment 2 [cite: 39, 79]
# We are using Lasso with a specific alpha hyperparameter
model = Lasso(alpha=0.1) 
model.fit(X_train, y_train)

# 4. Evaluate [cite: 40, 41, 42]
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 5. Save Outputs to the 'outputs' directory [cite: 30, 45, 47, 48]
os.makedirs('outputs', exist_ok=True)
metrics = {
    "name": "Radhika",
    "roll_no": "2022bcs148",
    "mse": float(mse),
    "r2_score": float(r2)
}

with open('outputs/results.json', 'w') as f:
    json.dump(metrics, f)

joblib.dump(model, 'outputs/model.pkl')

# 6. Print metrics to standard output [cite: 49]
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
