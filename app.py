import json
from flask import Flask, jsonify, request
from deploy.single_prediction import SinglePrediction

app = Flask(__name__)
single_prediction = SinglePrediction("deploy/model_scripted.pt", "data/preprocessing/raw_features_2024.csv")


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
