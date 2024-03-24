import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reads data from csv file
label_encoder = LabelEncoder()
data = pd.read_csv('data/market_analysis_2019.csv', delimiter=';')

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

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cleaned_data, actual_rev, test_size=0.2, random_state=42)

# Now, fit the model
model = KNeighborsRegressor(n_neighbors=15)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
