import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset from a local path
url = "/Users/sairishi/Desktop/Data-Science-Project-Series/Project_1/infolimpioavanzadoTarget.csv"
df = pd.read_csv(url)

# Print column names to verify the correct name for the date column
print(df.columns)

# Update these variable names based on the actual column names in your dataset
date_column = 'date'
close_column = 'close'
volume_column = 'volume'
open_column = 'open'
high_column = 'high'
low_column = 'low'

# Exploratory Data Analysis (EDA)
# Convert 'date' column to datetime
df[date_column] = pd.to_datetime(df[date_column])

# Display basic information about the dataset
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# Visualize stock price trends over time
plt.figure(figsize=(14, 7))
plt.plot(df[date_column], df[close_column], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Close Price Over Time')
plt.legend()
plt.show()

# Visualize volume trends over time
plt.figure(figsize=(14, 7))
plt.plot(df[date_column], df[volume_column], label='Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Volume Over Time')
plt.legend()
plt.show()

# Distribution of stock prices
plt.figure(figsize=(10, 6))
sns.histplot(df[close_column], kde=True)
plt.title('Distribution of Stock Prices')
plt.show()

# Box plot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(df[close_column])
plt.title('Box Plot of Stock Prices')
plt.show()

# Inspect the DataFrame to find non-numeric columns
print(df.dtypes)
print(df.head())

# Drop non-numeric columns or convert them to numeric if appropriate
# Example of converting all columns that can be converted to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df = df.dropna()

# Correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
# Drop rows with missing values
df.dropna(inplace=True)

# Convert date column to datetime and set as index
df[date_column] = pd.to_datetime(df[date_column])
df.set_index(date_column, inplace=True)

# Feature selection
features = df[[open_column, high_column, low_column, volume_column]]
target = df[close_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predictive Modeling
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)

# Model Evaluation
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print evaluation metrics
print("Linear Regression - MAE:", mae_lr, "MSE:", mse_lr, "R2:", r2_lr)
print("Random Forest - MAE:", mae_rf, "MSE:", mse_rf, "R2:", r2_rf)
print("Decision Tree - MAE:", mae_dt, "MSE:", mse_dt, "R2:", r2_dt)

# Forecast future stock values
future_dates = pd.date_range(start=df.index[-1], periods=30, freq='B')
future_features = pd.DataFrame(index=future_dates, columns=[open_column, high_column, low_column, volume_column])

# Fill future_features with average values from the training set (or another strategy)
future_features[open_column] = X_train[open_column].mean()
future_features[high_column] = X_train[high_column].mean()
future_features[low_column] = X_train[low_column].mean()
future_features[volume_column] = X_train[volume_column].mean()

# Scale future features
future_features_scaled = scaler.transform(future_features)

future_predictions_lr = lr_model.predict(future_features_scaled)
future_predictions_rf = rf_model.predict(future_features_scaled)
future_predictions_dt = dt_model.predict(future_features_scaled)

# Save results and figures
df.to_csv('processed_stock_data.csv')

with open('model_evaluation.txt', 'w') as f:
    f.write(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, R2: {r2_lr}\n")
    f.write(f"Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}\n")
    f.write(f"Decision Tree - MAE: {mae_dt}, MSE: {mse_dt}, R2: {r2_dt}\n")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df[close_column], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Close Price Over Time')
plt.legend()
plt.savefig('stock_close_price_over_time.png')
plt.close()

plt.figure(figsize=(14, 7))
plt.plot(df.index, df[volume_column], label='Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Volume Over Time')
plt.legend()
plt.savefig('stock_volume_over_time.png')
plt.close()

correlation_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()
