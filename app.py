import json
import urllib.request
from flask import Flask, jsonify, request
from flask_cors import CORS
from deploy.single_prediction import SinglePrediction

app = Flask(__name__)
CORS(app)

url_model = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/2cg5mebb/model_scripted.pt"
model_path = "model_scripted.pt"
url_scaler = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/2cg5mebb/scaler.pkl"
scaler_path = "scaler.pkl"

urllib.request.urlretrieve(url_model, model_path)
urllib.request.urlretrieve(url_scaler, scaler_path)

single_prediction = SinglePrediction(model_path, scaler_path, "data/preprocessing/raw_features_2024.csv")


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/metadata')
def metadata():
    metadata_json = json.load(open("data/metadata/metadata.json"))
    return metadata_json


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = request.json['date']
        print(date)

        output_dicts = single_prediction.predict_for_date(date)
        return jsonify(output_dicts)


if __name__ == '__main__':
    app.run()
