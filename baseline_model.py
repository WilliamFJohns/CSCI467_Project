import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reads data from csv file
label_encoder = LabelEncoder()
data = pd.read_csv('./data/market_analysis_2019.csv', delimiter=';')

data['month'] = pd.to_datetime(data['month'], format='%Y-%m')
data['revenue'] = data['revenue'].str.replace(',', '.').astype(float)

# Assuming 'guests' column might also contain '15+' or similar strings
data['guests'] = data['guests'].replace('15+', 15).astype(float)

# Encode 'city' column
data['city_encoded'] = label_encoder.fit_transform(data['city'])

# Calculate MonthSin and MonthCos
data['MonthSin'] = np.sin(2 * np.pi * data['month'].dt.month / 12)
data['MonthCos'] = np.cos(2 * np.pi * data['month'].dt.month / 12)

# Select features and target before removing NaN values
cleaned_data = data[['bedrooms', 'bathrooms', 'guests', 'city_encoded', 'MonthSin', 'MonthCos']]
actual_rev = data['revenue'].copy()

# Remove rows with NaN values from both features and target
combined = cleaned_data.copy()
combined['revenue'] = actual_rev
combined_clean = combined.dropna()

# Separate features and target again after cleaning
cleaned_data = combined_clean[['bedrooms', 'bathrooms', 'guests', 'city_encoded', 'MonthSin', 'MonthCos']]
actual_rev = combined_clean['revenue']

# Splitting the dataset where 20% is test set
X_train_val, X_test, y_train_val, y_test = train_test_split(cleaned_data, actual_rev, test_size=0.2, random_state=42)

# Split rest where approximately 10% of dataset is dev set
X_train, X_dev, y_train, y_dev = train_test_split(X_train_val, y_train_val, test_size=0.13, random_state=42)  # 0.25 x 0.8 = 0.2

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)



# Use dev set to find best hyperparameter n
best_n = None
best_y_dev = None
best_val_mae = float('inf')
for n in range(1,20):
    model = KNeighborsRegressor(n_neighbors=n)
    model.fit(X_train_scaled, y_train)
    
    y_dev_pred = model.predict(X_dev_scaled)
    
    val_mae = mean_absolute_error(y_dev, y_dev_pred)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_n = n
        best_y_dev = y_dev_pred

print(f"Best n_neighbors: {best_n} with Mean Absolute Error: {best_val_mae}")

# Test best model on test set
model = KNeighborsRegressor(n_neighbors=n)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {best_val_mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#For discussion
errors = np.abs(y_dev - best_y_dev)

# Determine threshold for data to be analyzed
# We grab data with the top 90th percentile
threshold = np.percentile(errors, 95)

# Extract samples where the error exceeds the threshold
significant_errors_indices = np.where(errors > threshold)[0]
significant_errors_samples = X_dev.iloc[significant_errors_indices]
smaller_errors_samples_indices = np.where(errors < np.percentile(errors, 10))[0]
smaller_errors_samples = X_dev.iloc[smaller_errors_samples_indices]
print(f"Bigger Error In 95th Percentile: \n {significant_errors_samples.head(5)}")
print(f"Smaller Error Less Than 5th Percentile: \n {smaller_errors_samples.head(5)}")