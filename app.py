import urllib.request
from flask import Flask, jsonify, request
from flask_cors import CORS
from deploy.single_prediction import SinglePrediction

app = Flask(__name__)
CORS(app)

url_model = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/8q8aq2gu/model_scripted.pt"
model_path = "model_scripted.pt"
X_url_scaler = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/8q8aq2gu/X_scaler.pkl"
X_scaler_path = "X_scaler.pkl"
y_url_scaler = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/8q8aq2gu/y_scaler.pkl"
y_scaler_path = "y_scaler.pkl"

urllib.request.urlretrieve(url_model, model_path)
urllib.request.urlretrieve(X_url_scaler, X_scaler_path)
urllib.request.urlretrieve(y_url_scaler, y_scaler_path)

single_prediction = SinglePrediction(model_path, X_scaler_path, y_scaler_path,
                                     "data/preprocessing/raw_features_2024.csv",
                                     "data/metadata/metadata.json")


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = request.json['date']
        print(date)

        output_dicts = single_prediction.predict_for_date(date)
        return jsonify(output_dicts)


if __name__ == '__main__':
    app.run()
