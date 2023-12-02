import pandas as pd
import openmeteo_requests
import requests_cache
from datetime import datetime, date, timedelta
from retry_requests import retry
from data.preprocessing.preprocess_features import PreprocessFeatures

weather_api_url = "https://api.open-meteo.com/v1/forecast"  # URL API


class SinglePredictionFeatures:
    def __init__(self, raw_features_path):
        # Import Features
        self.calendar_features = pd.read_csv(raw_features_path, sep=",")
        self.calendar_features['date'] = pd.to_datetime(self.calendar_features['date'],
                                                        format='%Y-%m-%d')  # Extract Date

        # Get Weather-Data
        # Set up Client and define Params
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def get_weather_forecast(self, timestamp):
        params = {
            "latitude": 47.4239,
            "longitude": 9.3748,
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "snowfall_sum"],
            "start_date": timestamp.strftime("%Y-%m-%d"),
            "end_date": timestamp.strftime("%Y-%m-%d")
        }

        responses = self.openmeteo.weather_api(weather_api_url, params=params)
        response = responses[0]

        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(3).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ), "temperature_2m_max": daily_temperature_2m_max, "temperature_2m_min": daily_temperature_2m_min,
            "rain_sum": daily_rain_sum, "snowfall_sum": daily_snowfall_sum}

        daily_dataframe = pd.DataFrame(data=daily_data)
        return daily_dataframe

    # def get_parking_data_df(self):
    #     df_parking_data = pd.read_csv("temp-for-lagged.csv", sep=";")
    #     df_parking_data["datetime"] = pd.to_datetime(df_parking_data["datetime"], format='%d.%m.%Y %H:%M')
    #
    #     df_parking_data.set_index("datetime", inplace=True)
    #
    #     # df = pd.merge(df, df_parking_data, on="datetime", how="outer")
    #
    #     print("df_parking_data")
    #     print(df_parking_data.head())
    #
    #     return df_parking_data

    def build_dataframe(self, input_date):
        timestamp = datetime.strptime(input_date, '%Y-%m-%d %H:%M')

        # Get Data
        df_weather = self.get_weather_forecast(timestamp)

        # Merge Weather with other Features
        df = pd.merge(df_weather, self.calendar_features, on="date", how="left")
        df["datetime"] = timestamp
        # df["datetime"] = pd.to_datetime(timestamp, format='%d.%m.%Y %H:%M')

        # parking_df = self.get_parking_data_df()

        return PreprocessFeatures(df).get_features_for_model()


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    date_today = date.today()
    date_tomorrow = date_today + timedelta(days=1)
    single_prediction_features = SinglePredictionFeatures("raw_features_2024.csv")
    # print(date_tomorrow.strftime("%Y-%m-%d %H:%M"))
    # print(date_today.strftime("%Y-%m-%d %H:%M"))
    df_demo, features_length = single_prediction_features.build_dataframe(date_tomorrow.strftime("%Y-%m-%d %H:%M"))
    print(df_demo.head())
    print(df_demo.columns)
    print(features_length)
