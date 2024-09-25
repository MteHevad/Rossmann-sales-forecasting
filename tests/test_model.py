import unittest
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

class TestModel(unittest.TestCase):
    
    def setUp(self):
        # Load the model and test data
        self.model = joblib.load('../models/rf_model_2023.pkl')
        self.X_val = pd.read_csv('../data/processed/validation_data.csv')
        self.y_val = pd.read_csv('../data/processed/validation_target.csv')
    
    def test_model_prediction(self):
        # Ensure the model can make predictions without errors
        predictions = self.model.predict(self.X_val)
        self.assertEqual(len(predictions), len(self.X_val), "Prediction length does not match input length.")
    
    def test_model_accuracy(self):
        # Check if the model produces predictions within an acceptable MSE range
        predictions = self.model.predict(self.X_val)
        mse = mean_squared_error(self.y_val, predictions)
        self.assertLess(mse, 1.5e6, "MSE exceeds acceptable range.")
    
if __name__ == '__main__':
    unittest.main()
