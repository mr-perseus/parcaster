import json
from flask import Flask, jsonify, request
from deploy.predict import Predict


app = Flask(__name__)
predict = Predict("deploy/model_scripted.pt")


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

        output_dicts = predict.predict_for_date(date)
        return jsonify(output_dicts)


if __name__ == '__main__':
    app.run()
