import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, jsonify, request

app = Flask(__name__)

model_path = "model_scripted.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

batch_size = 1  # Required for model input
parking_data_labels = ["P24", "P44", "P42", "P33", "P23", "P25", "P21", "P31", "P53", "P32", "P22", "P52", "P51",
                       "P43"]  # TODO get these from metadata file
ignored_columns = ["datetime", "date", "year", "month", "day", "weekdayname", "weekday", "time", "hour", "minute"]


def get_features_df(date):
    # TODO run this method from preprocessing, and remove this mock from here
    feature_columns = ['ferien', 'feiertag', 'covid_19', 'olma_offa', 'temperature_2m_max',
                       'temperature_2m_min', 'rain_sum', 'snowfall_sum', 'sin_minute',
                       'cos_minute', 'sin_hour', 'cos_hour', 'sin_weekday', 'cos_weekday',
                       'sin_day', 'cos_day', 'sin_month', 'cos_month']

    features_length = len(feature_columns)

    df = pd.DataFrame([0] * len(feature_columns)).T
    df.columns = feature_columns
    return df, features_length


def build_dataset(df):
    features = torch.Tensor(df.values)

    dataset = TensorDataset(features)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def predict_with_model(dataloader, features_length):
    data = next(iter(dataloader))[0]
    data = data.view([batch_size, -1, features_length]).to(device)
    return model(data).cpu()


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = request.json['date']
        print(date)

        features_df, features_length = get_features_df(date)
        dataloader = build_dataset(features_df)

        output = predict_with_model(dataloader, features_length)

        output_dicts = [dict(zip(parking_data_labels, row)) for row in output.tolist()]
        return jsonify(output_dicts)


if __name__ == '__main__':
    app.run()
