import torch
from torch.utils.data import TensorDataset, DataLoader
from data.preprocessing.get_features_for_prediction import build_dataframe

batch_size = 1  # Required for model input
parking_data_labels = ["P24", "P44", "P42", "P33", "P23", "P25", "P21", "P31", "P53", "P32", "P22", "P52", "P51",
                       "P43"]  # TODO get these from metadata file


class Predict:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @staticmethod
    def build_dataset(df):
        features = torch.Tensor(df.values)

        dataset = TensorDataset(features)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def predict_with_model(self, dataloader, features_length):
        data = next(iter(dataloader))[0]
        data = data.view([batch_size, -1, features_length]).to(device)
        return self.model(data).cpu()

    def predict_for_date(self, date):
        features_df, features_length = build_dataframe(date)
        dataloader = self.build_dataset(features_df)

        output = self.predict_with_model(dataloader, features_length)

        return [dict(zip(parking_data_labels, row)) for row in output.tolist()]
