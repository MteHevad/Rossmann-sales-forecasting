from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/rf_model_2023.pkl')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['Store'], data['Promo'], data['StateHoliday'], data['SchoolHoliday'], 
                         data['DayOfWeek'], data['Month'], data['CompetitionDistance'], 
                         data['StoreType'], data['Assortment']]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({'predicted_sales': prediction})

if __name__ == '__main__':
    app.run(debug=True)

