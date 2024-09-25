from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib
import datetime

def build_model(X, y):
    pipeline = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model MSE: {mse}')
    
    # Serialize the model with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_path = f"model/random_forest_{timestamp}.pkl"
    joblib.dump(pipeline, model_path)
    print(f'Model saved at {model_path}')

if __name__ == "__main__":
    df = pd.read_csv('processed_train.csv')
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    build_model(X, y)
