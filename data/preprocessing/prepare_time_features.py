# Prepare time-features
## Extract Time Components
df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M') # Make Object to datetime
df['date'] = df['datetime'].dt.date # Extract Date
df['year'] = df['datetime'].dt.year # Extract Year
df['month'] = df['datetime'].dt.month # Extract Month
df['day'] = df['datetime'].dt.day # Extract Day
df['weekdayname'] = df['datetime'].dt.day_name()
df['weekday'] = df['datetime'].dt.dayofweek # Extract Weekday
df['time'] = df['datetime'].dt.strftime('%H:%M') # Extract Time
df['hour'] = df['datetime'].dt.hour # Extract Hour
df['minute'] = df['datetime'].dt.minute # Extract Minute

## Decompose Time-Features in sine and cosine component
### Inspired by https://medium.com/mlearning-ai/transformer-implementation-for-time-series-forecasting-a9db2db5c820 
### (vgl. https://github.com/nok-halfspace/Transformer-Time-Series-Forecasting/blob/main/Preprocessing.py) 

minutes_in_hour = 60
hours_in_day = 24
days_in_week = 7
days_in_month = 30
month_in_year = 12

df['sin_minute'] = np.sin(2*np.pi*df['minute']/minutes_in_hour)
df['cos_minute'] = np.cos(2*np.pi*df['minute']/minutes_in_hour)
df['sin_hour'] = np.sin(2*np.pi*df['hour']/hours_in_day)
df['cos_hour'] = np.cos(2*np.pi*df['hour']/hours_in_day)
df['sin_weekday'] = np.sin(2*np.pi*df['weekday']/days_in_week)
df['cos_weekday'] = np.cos(2*np.pi*df['weekday']/days_in_week)
df['sin_day'] = np.sin(2*np.pi*df['day']/days_in_month)
df['cos_day'] = np.cos(2*np.pi*df['day']/days_in_month)
df['sin_month'] = np.sin(2*np.pi*df['month']/month_in_year)
df['cos_month'] = np.cos(2*np.pi*df['month']/month_in_year)