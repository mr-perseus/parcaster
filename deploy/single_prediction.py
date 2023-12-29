import torch
import json
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from model.scaler import Scaler
from data.preprocessing.single_prediction_features import SinglePredictionFeatures
from data.metadata.metadata import parking_data_labels

batch_size = 1  # Required for model input


class SinglePrediction:
    def __init__(self, model_path, X_scaler_path, y_scaler_path, raw_features_path, metadata_path):
        self.metadata = json.load(open(metadata_path))
        self.labels_readable = [self.metadata["parking_sg"]["fields"][field]["label"] for field in parking_data_labels]
        self.max_capacity = [self.metadata["parking_sg"]["fields"][field]["max_cap"] for field in parking_data_labels]
        self.X_scaler = Scaler.load(X_scaler_path)
        self.y_scaler = Scaler.load(y_scaler_path)
        self.single_prediction_features = SinglePredictionFeatures(raw_features_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def build_dataset(self, df):
        features = torch.Tensor(df)

        dataset = TensorDataset(features)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def predict_with_model(self, dataloader, features_length):
        data = next(iter(dataloader))[0]
        data = data.view([batch_size, -1, features_length]).to(self.device)
        return self.model(data).detach().cpu().numpy()

    def pretty_prediction(self, output_scaled_back):
        output_list = output_scaled_back.tolist()[0]

        # TODO find cleaner solution.
        rounded_list = [round(entry) for entry in output_list]
        list_capped_min = [0 if entry < 0 else entry for entry in rounded_list]
        list_capped_max = [self.max_capacity[i] if entry > self.max_capacity[i] else entry for i, entry in
                           enumerate(list_capped_min)]

        return list_capped_max

    def predict_for_date(self, date):
        features_df, features_length = self.single_prediction_features.build_dataframe(date)

        scaled_features = self.X_scaler.fit_transform(features_df)

        dataloader = self.build_dataset(scaled_features)

        output = self.predict_with_model(dataloader, features_length)

        output_scaled_back = self.y_scaler.inverse_transform(pd.DataFrame(output, columns=parking_data_labels))

        return {
            "predictions": self.pretty_prediction(output_scaled_back),
            "labels": parking_data_labels,
            "labels_readable": self.labels_readable,
            "max_capacity": self.max_capacity
        }


if __name__ == "__main__":
    predict = SinglePrediction("../model_scripted.pt", "../X_scaler.pkl", "../y_scaler",
                               "../data/preprocessing/raw_features_2024.csv",
                               "../data/metadata/metadata.json")
    print(predict.predict_for_date("2023-12-08 08:00"))
    print(predict.predict_for_date("2023-12-10 18:00"))
    print(predict.predict_for_date("2023-12-12 12:00"))
