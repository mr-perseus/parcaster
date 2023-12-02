import numpy as np
import pandas as pd
from data.metadata.metadata import feature_columns, parking_data_labels


class PreprocessFeatures:
    def __init__(self, df):
        self.df = df

    def get_features_for_model(self):
        self.append_time_features()
        # self.get_lagged_features()

        return self.df[feature_columns], len(feature_columns)

    def append_time_features(self):
        # Prepare time-features
        ## Extract Time Components
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%d.%m.%Y %H:%M')  # Make Object to datetime
        self.df['date'] = self.df['datetime'].dt.date  # Extract Date
        self.df['year'] = self.df['datetime'].dt.year  # Extract Year
        self.df['month'] = self.df['datetime'].dt.month  # Extract Month
        self.df['day'] = self.df['datetime'].dt.day  # Extract Day
        self.df['weekdayname'] = self.df['datetime'].dt.day_name()
        self.df['weekday'] = self.df['datetime'].dt.dayofweek  # Extract Weekday
        self.df['time'] = self.df['datetime'].dt.strftime('%H:%M')  # Extract Time
        self.df['hour'] = self.df['datetime'].dt.hour  # Extract Hour
        self.df['minute'] = self.df['datetime'].dt.minute  # Extract Minute

        ## Decompose Time-Features in sine and cosine component
        ### Inspired by https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820
        ### (vgl. https://github.com/nok-halfspace/Transformer-Time-Series-Forecasting/blob/main/Preprocessing.py)

        minutes_in_hour = 60
        hours_in_day = 24
        days_in_week = 7
        days_in_month = 30
        month_in_year = 12

        self.df['sin_minute'] = np.sin(2 * np.pi * self.df['minute'] / minutes_in_hour)
        self.df['cos_minute'] = np.cos(2 * np.pi * self.df['minute'] / minutes_in_hour)
        self.df['sin_hour'] = np.sin(2 * np.pi * self.df['hour'] / hours_in_day)
        self.df['cos_hour'] = np.cos(2 * np.pi * self.df['hour'] / hours_in_day)
        self.df['sin_weekday'] = np.sin(2 * np.pi * self.df['weekday'] / days_in_week)
        self.df['cos_weekday'] = np.cos(2 * np.pi * self.df['weekday'] / days_in_week)
        self.df['sin_day'] = np.sin(2 * np.pi * self.df['day'] / days_in_month)
        self.df['cos_day'] = np.cos(2 * np.pi * self.df['day'] / days_in_month)
        self.df['sin_month'] = np.sin(2 * np.pi * self.df['month'] / month_in_year)
        self.df['cos_month'] = np.cos(2 * np.pi * self.df['month'] / month_in_year)

    # def get_lagged_features(self, period=24):
    #     for label in parking_data_labels:
    #         self.df[label + '_lagged_' + str(period)] = self.labels_df[label].shift(periods=period)
