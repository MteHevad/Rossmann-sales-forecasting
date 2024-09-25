import unittest
import json
from app import app

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        # Set up the testing client for Flask
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Test the /predict endpoint with a sample payload
        payload = {
            "Store": 1,
            "Promo": 1,
            "StateHoliday": 0,
            "SchoolHoliday": 0,
            "DayOfWeek": 4,
            "Month": 7,
            "CompetitionDistance": 300,
            "StoreType": 1,
            "Assortment": 1
        }
        
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        data = json.loads(response.get_data(as_text=True))
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_sales', data)
        self.assertGreater(data['predicted_sales'], 0, "Predicted sales should be greater than 0.")
    
if __name__ == '__main__':
    unittest.main()
