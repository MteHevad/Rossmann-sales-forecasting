from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, path='models/model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
