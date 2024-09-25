import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(X, y):
    model = create_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    return model

def preprocess_for_lstm(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    
    # Preprocess for LSTM (Scaling and preparing sliding windows)
    sales_data = df[['Sales']].values
    scaled_data, scaler = preprocess_for_lstm(sales_data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    lstm_model = train_lstm_model(X, y)

    # Serialize the model
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_path = f"model/lstm_{timestamp}.h5"
    lstm_model.save(model_path)
    joblib.dump(scaler, f"model/lstm_scaler_{timestamp}.pkl")
    print(f'Model saved at {model_path}')
