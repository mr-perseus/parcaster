import pandas as pd
import openmeteo_requests
import requests_cache
from datetime import datetime, date, timedelta
from retry_requests import retry
from prepare_time_features import prepare_time_features

# Import Features
calendar_features = pd.read_csv("raw_features_2024.csv", sep=",")
calendar_features['date'] = pd.to_datetime(calendar_features['date'], format='%Y-%m-%d')  # Extract Date

# Get Weather-Data
# Set up Client and define Params
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"  # URL API


# Define Function
def get_weather_forecast(params):
    responses = openmeteo.weather_api(url, params=params)
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


def build_dataframe(input_date):
    timestamp = datetime.strptime(input_date, '%Y-%m-%d %H:%M')

    params = {
        "latitude": 47.4239,
        "longitude": 9.3748,
        "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "snowfall_sum"],
        "start_date": timestamp.strftime("%Y-%m-%d"),
        "end_date": timestamp.strftime("%Y-%m-%d")
    }

    # Get Data
    df_weather = get_weather_forecast(params)

    # Merge Weather with other Features
    df = pd.merge(df_weather, calendar_features, on="date", how="left")
    df["datetime"] = timestamp

    # Add time-features
    df = prepare_time_features(df)

    return df


if __name__ == "__main__":
    date_today = date.today()
    date_tomorrow = date_today + timedelta(days=1)
    df_demo = build_dataframe(date_tomorrow.strftime("%Y-%m-%d %H:%M"))
    print(df_demo.head())
