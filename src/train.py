import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # Change this for Task 5
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load Dataset (Wine Quality)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# 2. Pre-processing
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model (Task 5: Modify this part for each experiment)
model = LinearRegression() 
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 5. Save Outputs
os.makedirs('outputs', exist_ok=True)
with open('outputs/results.json', 'w') as f:
    json.dump({"mse": mse, "r2_score": r2}, f)
joblib.dump(model, 'outputs/model.pkl')

# 6. Print for Logs
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
