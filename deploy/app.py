import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from flask import Flask, jsonify, request

app = Flask(__name__)

model_path = "model_scripted.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.eval()

parking_data_labels = ["P24", "P44", "P42", "P33", "P23", "P25", "P21", "P31", "P53", "P32", "P22", "P52", "P51",
                       "P43"]  # TODO get these from metadata file
ignored_columns = ["datetime", "date", "year", "month", "day", "weekdayname", "weekday", "time", "hour", "minute"]


def load_features_labels():
    df = pd.read_csv("../data/preprocessing/02_pp_sg_train_features.csv", sep=";")

    y = df[parking_data_labels]
    X = df.drop(columns=parking_data_labels)
    X = X.drop(columns=ignored_columns)

    input_dim = len(X.columns)
    output_dim = len(y.columns)

    print(f"Input dimension: {input_dim}, columns: {X.columns}")
    print(f"Output dimension: {output_dim}, columns: {y.columns}")

    return X, y, input_dim, output_dim


def build_dataset(batch_size, X):
    features = torch.Tensor(X.values)

    dataset = TensorDataset(features)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = request.json['date']
        print(date)

        X, y, input_dim, output_dim = load_features_labels()

        batch_size = 1
        index_to_predict = 10000

        X_to_predict = X[index_to_predict:index_to_predict + batch_size]

        dataloader = build_dataset(batch_size, X_to_predict)

        data = next(iter(dataloader))[0]
        data = data.view([batch_size, -1, input_dim]).to(device)
        output = model(data).cpu()
        output_dicts = [dict(zip(parking_data_labels, row)) for row in output.tolist()]
        return jsonify(output_dicts)


if __name__ == '__main__':
    app.run()
