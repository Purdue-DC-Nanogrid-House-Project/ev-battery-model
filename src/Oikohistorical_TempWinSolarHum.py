from datetime import timedelta, datetime
import pytz
import requests
import json
import pandas as pd

today = datetime.today() - timedelta(days = -0)
date_old = datetime.today() - timedelta(days=120)

today = today.astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")
date_old = date_old.astimezone(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d")

r = requests.get('https://api.oikolab.com/weather',
                     params = {'param': ['temperature','wind_speed','surface_solar_radiation', 'surface_diffuse_solar_radiation','direct_normal_solar_radiation'], #'relative_humidity'],
                             'start': date_old,
                             'end': today,
                             'lat': '40.430930',
                             'lon': '-86.911617',
                             'api-key': '86237827ec964dfc975d742ba45a33b8'})

print(r.text)
weather_data = json.loads(r.json()['data'])
df = pd.DataFrame(index=pd.to_datetime(weather_data['index'],
                                           unit='s').tz_localize('UTC')
                        .tz_convert('US/Eastern')
                        .strftime("%m/%d/%Y %H:%M:%S"),
                    data=weather_data['data'],
                    columns=weather_data['columns'])

df.to_csv('HistoricalWeather.csv')
