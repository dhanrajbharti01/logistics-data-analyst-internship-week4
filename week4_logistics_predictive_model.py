import pandas as pd

# Load dataset
df = pd.read_csv("logistics_orders.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Prepare data for predictive modeling

# Convert traffic level into numerical values
df["traffic_level_encoded"] = df["traffic_level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

print("\nEncoded Traffic Level:")
print(df[["traffic_level", "traffic_level_encoded"]].head())

# Select features and target

features = [
    "distance_km",
    "parcel_count",
    "traffic_level_encoded",
    "hour",
    "vehicle_capacity",
    "planned_time_min"
]

X = df[features]
y = df["actual_time_min"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
 
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

from sklearn.linear_model import LinearRegression
# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
linear_predictions = linear_model.predict(X_test)

print("\nLinear Regression Predictions:")
print(linear_predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Evaluate Linear Regression model
mae = mean_absolute_error(y_test, linear_predictions)
rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R-squared:", r2)

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest model
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

# Make predictions
rf_predictions = random_forest_model.predict(X_test)

print("\nRandom Forest Predictions:")
print(rf_predictions)

# Evaluate Random Forest model
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R-squared:", rf_r2)