import numpy as np
import pandas as pd
import glob
import json

# Input source: https://daten.stadt.sg.ch/explore/dataset/freie-parkplatze-in-der-stadt-stgallen-pls
input_json_path = 'data/*.json'
input_values_property_name = 'results'
input_parking_id_column_name = 'phid'
input_time_column_name = 'zeitpunkt'
input_free_parking_spots_column_name = 'shortfree'

output_time_column_name = 'date'
output_csv_file_name = 'data.csv'


def load_json_data():
    parking_data_dict = {}

    json_files = glob.glob(input_json_path)

    for json_file_path in json_files:
        with open(json_file_path) as file:
            data = json.load(file)

        for row in data[input_values_property_name]:
            parking_id = row[input_parking_id_column_name]
            if parking_id not in parking_data_dict:
                parking_data_dict[parking_id] = []
            parking_data_dict[parking_id].append(row)

    return parking_data_dict


def to_pandas_data_frame_dict(json_data):
    data_frame_dict = {}
    for parking_id in json_data:
        df = pd.DataFrame(json_data[parking_id])
        df[output_time_column_name] = pd.to_datetime(df[input_time_column_name])
        df.set_index(output_time_column_name, inplace=True)
        sorted_df = df.sort_index()
        data_frame_dict[parking_id] = sorted_df

    return data_frame_dict


def clear_df_columns(data_frames):
    cleared_data_frames = {}
    for key, df in data_frames.items():
        df[key] = df[input_free_parking_spots_column_name]  # Set free parking spots to the parking id
        cleared_data_frames[key] = df.loc[:, [key]]  # Only keep the parking id column

    return cleared_data_frames


def combine_df_dict_to_single_df(data_frames):
    combined_data_frame = data_frames[list(data_frames.keys())[0]].copy()  # Clone first df
    print(f"Length (before removing duplicates: {len(combined_data_frame)}")
    combined_data_frame = combined_data_frame[~combined_data_frame.index.duplicated(keep='first')]
    print(f"Length (after removing duplicates: {len(combined_data_frame)}")

    for key, df in data_frames.items():
        for column in df.columns:
            combined_data_frame[column] = np.nan

        print(f"Adding {key} to combined df. (cols)", combined_data_frame.columns, "Length of combined df",
              len(combined_data_frame))

        for index, row in df.iterrows():
            if index not in combined_data_frame.index:
                raise Exception(f"Index {index} not in combined df. (cols)", combined_data_frame.columns,
                                "Length of combined df", len(combined_data_frame))

            combined_data_frame.loc[index, key] = row[key]

    combined_data_frame = combined_data_frame.sort_index()

    return combined_data_frame


def write_csv_file(data_frame):
    data_frame.to_csv(output_csv_file_name)


def jsons_to_csv_pipeline():
    result = load_json_data()
    result = to_pandas_data_frame_dict(result)
    result = clear_df_columns(result)
    result = combine_df_dict_to_single_df(result)

    print("result.columns")
    print(result.columns)
    print("result.head()")
    print(result.head())
    print(f"Amount Null / NaN: {result.isnull().sum().sum()} / {result.isna().sum().sum()}")

    write_csv_file(result)


if __name__ == "__main__":
    jsons_to_csv_pipeline()
