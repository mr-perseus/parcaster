import os
import urllib.request
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from scaler import Scaler
from data.metadata.metadata import parking_data_labels
from data.preprocessing.preprocess_features import PreprocessFeatures

output_path = "test_predictions"
os.mkdir(output_path) if not os.path.exists(output_path) else None

url_model = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/f8x6m452/model_scripted.pt"
model_path = "model_scripted.pt"
X_url_scaler = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/f8x6m452/X_scaler.pkl"
X_scaler_path = "X_scaler.pkl"
y_url_scaler = "https://api.wandb.ai/files/parcaster/pp-sg-lstm/f8x6m452/y_scaler.pkl"
y_scaler_path = "y_scaler.pkl"

urllib.request.urlretrieve(url_model, model_path)
urllib.request.urlretrieve(X_url_scaler, X_scaler_path)
urllib.request.urlretrieve(y_url_scaler, y_scaler_path)

test_data_path = "../data/preprocessing/01_pp_sg_test_cleaned.csv"

def build_dataset(batch_size, X, y):
    features = torch.Tensor(X)
    targets = torch.Tensor(y)

    dataset = TensorDataset(features, targets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def load_features_labels(csv_path):
    df = pd.read_csv(csv_path, sep=";")

    preprocess_features = PreprocessFeatures(df)

    y = df[parking_data_labels]
    X, input_dim = preprocess_features.get_features_for_model()

    output_dim = len(y.columns)

    print(f"Input dimension: {input_dim}, columns: {X.columns}")
    print(f"Output dimension: {output_dim}, columns: {y.columns}")

    return X, y, input_dim, output_dim

X_test, y_test, input_dim, output_dim = load_features_labels(test_data_path)

X_scaler = Scaler.load(X_scaler_path)
y_scaler = Scaler.load(y_scaler_path)

X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test)
test_loader = build_dataset(1, X_test_scaled, y_test_scaled)

def calculate_loss(output, target, y_scaler):
    output_inverted = y_scaler.inverse_transform(output.detach().cpu().numpy())
    target_inverted = y_scaler.inverse_transform(target.detach().cpu().numpy())
    loss = mean_squared_error(output_inverted, target_inverted, squared=False)
    return loss, output_inverted, target_inverted

def plot_test_prediction(outputs, targets):
    for i, (output, target) in enumerate(zip(outputs, targets)):
        df_output = pd.DataFrame(output, columns=parking_data_labels)
        df_target = pd.DataFrame(target, columns=parking_data_labels)

        n_features = len(df_output.columns)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Setting the positions of the bars
        ind = np.arange(n_features)  # the x locations for the groups
        width = 0.35  # the width of the bars

        # Plotting bars for each row
        bars1 = ax.bar(ind - width / 2, df_output.iloc[0], width, label='Prediction (from model)')
        bars2 = ax.bar(ind + width / 2, df_target.iloc[0], width, label='Target (form dataset)')

        # Adding some text for labels, title, and custom x-axis tick labels
        ax.set_xlabel('Parking garages')
        ax.set_ylabel('Free parking spots')
        ax.set_title(f'Comparison of Two Rows in a Bar Chart {i}')
        ax.set_xticks(ind)
        ax.set_xticklabels(df_output.columns)
        ax.legend()

        plt.savefig(f"{output_path}/test_prediction_{i}.png")


def test_network(network, loader, batch_size, input_dim, y_scaler):
    outputs = []
    targets = []
    losses = []
    with torch.no_grad():
        network.eval()
        for _, (data, target) in enumerate(loader):
            data, target = data.view([batch_size, -1, input_dim]).to(device), target.to(device)
            output = network(data)
            loss, output_inverted, target_inverted = calculate_loss(output, target, y_scaler)
            losses.append(loss)
            outputs.append(output_inverted)
            targets.append(target_inverted)

    return np.mean(losses), outputs, targets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.jit.load(model_path, map_location=device)
avg_test_loss, test_outputs, test_targets = test_network(model, test_loader, 1, input_dim, y_scaler)
plot_test_prediction(test_outputs, test_targets)