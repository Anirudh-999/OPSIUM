import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("demand_data_with_random_holidays_and_weather.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Create DayOfYear feature
data["DayOfYear"] = data["Date"].dt.dayofyear

X = data[["DayOfYear"]] 
y = data["Demand"]

# Train-test split (using 3 years for training and 1 year for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)  

# Train the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost Mean Squared Error: {mse:.2f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")

# Ask for user input for a specific date
input_date = input("Enter a date (YYYY-MM-DD) to predict demand: ")

# Convert the input date to datetime and extract the DayOfYear
input_date = pd.to_datetime(input_date)
input_day_of_year = input_date.dayofyear

# Predict the demand for the entered date using the trained model
input_pred = model.predict(np.array([[input_day_of_year]]))

# Find the actual demand for that date (if it exists in the test set)
actual_demand = data[data["Date"] == input_date]["Demand"].values

# Display the result
if len(actual_demand) > 0:
    print(f"Actual Demand for {input_date.strftime('%Y-%m-%d')}: {actual_demand[0]:.2f}")
else:
    print(f"No actual data available for {input_date.strftime('%Y-%m-%d')}.")

print(f"Predicted Demand for {input_date.strftime('%Y-%m-%d')}: {input_pred[0]:.2f}")

# Optionally, plot the actual vs predicted demand for the whole test set
plt.figure(figsize=(10, 6))
plt.plot(data["Date"].iloc[len(X_train):], y_test, label="Actual Demand", color="green")
plt.plot(data["Date"].iloc[len(X_train):], y_pred, label="Predicted Demand", color="red")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("Actual vs Predicted Demand (XGBoost)")
plt.legend()
plt.show()
