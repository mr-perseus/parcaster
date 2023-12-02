import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from model.scaler import Scaler
from data.preprocessing.single_prediction_features import SinglePredictionFeatures
from data.metadata.metadata import parking_data_labels

batch_size = 1  # Required for model input


class SinglePrediction:
    def __init__(self, model_path, scaler_path, raw_features_path):
        self.scaler = Scaler.load(scaler_path)
        self.single_prediction_features = SinglePredictionFeatures(raw_features_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def build_dataset(self, df):
        features = torch.Tensor(df.values)

        dataset = TensorDataset(features)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def predict_with_model(self, dataloader, features_length):
        data = next(iter(dataloader))[0]
        data = data.view([batch_size, -1, features_length]).to(self.device)
        return self.model(data).detach().cpu().numpy()

    def predict_for_date(self, date):
        features_df, features_length = self.single_prediction_features.build_dataframe(date)
        dataloader = self.build_dataset(features_df)

        output = self.predict_with_model(dataloader, features_length)

        output_scaled_back = self.scaler.inverse_transform(pd.DataFrame(output, columns=parking_data_labels))

        return [dict(zip(parking_data_labels, row)) for row in output_scaled_back.tolist()]


if __name__ == "__main__":
    predict = SinglePrediction("model_scripted.pt", "scaler.pkl", "../data/preprocessing/raw_features_2024.csv")
    print(predict.predict_for_date("2023-12-08 08:00"))
    print(predict.predict_for_date("2023-12-10 18:00"))
    print(predict.predict_for_date("2023-12-12 12:00"))
