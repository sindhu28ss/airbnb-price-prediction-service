from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
with open('price_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the predictors (input features)
predictors = ['availability_365', 'calculated_host_listings_count', 'minimum_nights', 'number_of_reviews', 'room_type_Private room']

@app.route('/')
def home():
    return "Welcome to the Airbnb Price Prediction API!"

@app.route('/predict_price', methods=['GET'])
def predict_price():
    try:
        # Extract query parameters from the GET request
        query_params = request.args

        # Convert query parameters to a DataFrame
        input_data = pd.DataFrame([{key: float(query_params[key]) for key in predictors}])

        # Predict using the loaded model
        predictions = model.predict(input_data)

        # Return the predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
