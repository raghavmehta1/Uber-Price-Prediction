from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Uber Ride Price Prediction Model"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [data['distance'], data['time'], data['traffic_conditions']]
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
