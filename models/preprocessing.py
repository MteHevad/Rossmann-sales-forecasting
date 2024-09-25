import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df):
    # Handle missing values
    df.fillna(0, inplace=True)

    # Feature extraction from datetime columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    # Feature scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    processed_df = preprocess_data(df)
    processed_df.to_csv('processed_train.csv', index=False)
