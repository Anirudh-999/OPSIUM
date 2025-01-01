import pandas as pd
import numpy as np
from datetime import timedelta, datetime

days = np.arange(1, 1461) 
base_demand = 300  # Base demand level
seasonality_amplitude = 100  # Amplitude of seasonal variation

# Add seasonality (sinusoidal pattern to simulate yearly demand variation)
demand = base_demand + seasonality_amplitude * np.sin(2 * np.pi * (days % 365) / 365) + np.random.normal(0, 10, size=1460)

# Create a DataFrame
data = pd.DataFrame({
    "Date": pd.to_datetime("2020-01-01") + pd.to_timedelta(days - 1, unit="D"),
    "Demand": demand
})

data["DayOfYear"] = data["Date"].dt.dayofyear

# Simulate weather conditions (Sunny, Cloudy, Rainy)
weather_condition = np.random.choice(["Sunny", "Cloudy", "Rainy"], size=len(days), p=[0.7, 0.2, 0.1])
data["WeatherCondition"] = weather_condition

# Simulate product launch (0 = no launch, 1 = launch)
product_launch = np.random.choice([0, 1], size=len(days), p=[0.95, 0.05])
data["ProductLaunch"] = product_launch

np.random.seed(42)  
holidays = []

for year in range(2020, 2024):
    holiday_dates = []
    for _ in range(10):  
        random_day = np.random.randint(1, 366)  # Random day of the year (1 to 365)
        holiday_date = datetime(year, 1, 1) + timedelta(days=random_day - 1)
        holiday_dates.append(holiday_date)
    holidays.extend(holiday_dates)

data["Holiday"] = data["Date"].isin(pd.to_datetime(holidays)).astype(int)

data["Demand"] += data["Holiday"] * 50  
data["Demand"] -= (data["WeatherCondition"] == "Rainy") * 30  
data["Demand"] += data["ProductLaunch"] * 100 

data.to_csv("demand_data.csv", index=False)

print("Dataset with random holidays, weather, and product launches created and saved as 'demand_data.csv'")
