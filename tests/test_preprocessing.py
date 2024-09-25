import unittest
import pandas as pd
import numpy as np
from scripts.preprocess_data import preprocess_data

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Load raw data (sample)
        self.train = pd.read_csv('../data/raw/train.csv', dtype={'StateHoliday': str})
        self.store = pd.read_csv('../data/raw/store.csv')
    
    def test_missing_values(self):
        # Test if missing values are filled correctly
        processed_train = preprocess_data(self.train, self.store)
        self.assertFalse(processed_train['CompetitionDistance'].isnull().values.any(), "Missing CompetitionDistance values were not handled correctly.")
    
    def test_feature_engineering(self):
        # Test if date features are engineered correctly
        processed_train = preprocess_data(self.train, self.store)
        self.assertIn('Month', processed_train.columns, "Feature 'Month' was not created.")
        self.assertIn('DayOfWeek', processed_train.columns, "Feature 'DayOfWeek' was not created.")
    
if __name__ == '__main__':
    unittest.main()
