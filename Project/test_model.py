# Example of loading and using a saved RandomForestRegressor model to predict with a single value of x

# 1. Import necessary libraries
import joblib
import numpy as np  # Assuming you're using NumPy for data manipulation

# 2. Load the trained model
model_path = 'power_prediction.sav'  # Replace with your actual file path
loaded_model = joblib.load(model_path)

# 3. Prepare the single input value for prediction
x_value = 4.5  # Replace with your actual single value of x

# Convert x_value into a NumPy array with the correct shape
X_single = np.array([[x_value]])  # Double brackets for a single feature input, shape (1, 1)

# 4. Make prediction
prediction = loaded_model.predict(X_single)

# 5. Print or use the prediction
print("Prediction for x =", x_value, ":", prediction)
