import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load processed data
train = pd.read_csv('../data/processed/train_processed.csv')

# Feature selection
features = ['Store', 'Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek', 'Month', 'CompetitionDistance', 'StoreType', 'Assortment']
X = train[features]
y = train['Sales']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
joblib.dump(rf_model, '../models/rf_model_2023.pkl')
