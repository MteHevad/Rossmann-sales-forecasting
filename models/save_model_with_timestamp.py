import os
import joblib
from datetime import datetime

# Define the directory where you want to save the model
model_dir = r'C:\Users\hp\Desktop\KAIM\Week 4 Deliverable\model\\'

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create a timestamp for the model filename
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_filename = model_dir + f"random_forest_model_{timestamp}.pkl"

# Save the trained pipeline model
joblib.dump(pipeline, model_filename)

print(f"Model saved to {model_filename}")
