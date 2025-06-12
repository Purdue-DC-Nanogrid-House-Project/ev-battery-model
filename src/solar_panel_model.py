import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, temperature
from pvlib.location import Location
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime,timedelta

class SolarPanelModel:
    def __init__(self,dt,day,pdc0,v_mp,i_mp,v_oc,i_sc,alpha_sc,beta_oc,gamma_pdc,latitude,longitude,i):
        # Location Paramaters
        self.latitude = latitude # Latitude Co-ordinate of Solar Panel Location (deg)
        self.longitude = longitude # Longitude Co-ordinate of Solar Panel Location (deg)
        self.day = day
        self.dt = dt
        self.i = i

        # Weather Parameters
        weather_path = 'data/HistoricalWeather.csv'
        self.start_time,self.end_time = self.date_format()
        # self.start_time = '2024-06-15 00:00:00'
        # self.end_time = '2024-06-16 00:00:00'
        self.weather = self.load_weather_data(weather_path,self.start_time,self.end_time)

        # Define module parameters for Panasonic VBHN325KA03
        self.module_parameters = {
            'pdc0': pdc0,  # DC power at standard test conditions (W)
            'v_mp': v_mp, # Maximum power voltage (V)
            'i_mp': i_mp, # Maximum power current (A)
            'v_oc': v_oc, # Open circuit voltage (V)
            'i_sc': i_sc,  # Short circuit current (A)
            'alpha_sc':  alpha_sc,  # Temperature coefficient of Isc (A/C)
            'beta_oc': beta_oc,  # Temperature coefficient of Voc (V/C)
            'gamma_pdc':gamma_pdc  # Power temperature coefficient (1/C)
        }

        # Temperature model parameters
        self.temperature_model_parameters = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # Define inverter parameters (assuming total system power per module)
        self.inverter_parameters = {
            'pdc0': 325  # DC input power rating (W) per module
        }

        # Define each segment
        self.segments = [
        {'tilt': 32, 'azimuth': 90, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 50, 'azimuth': 180, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 32, 'azimuth': 90, 'modules': 6, 'pdc0': 1.95e3},
        {'tilt': 30, 'azimuth': 270, 'modules': 30, 'pdc0': 9.75e3}
        ]

        # Define the location
        self.site = Location(self.latitude, self.longitude)

        # Create solar panel model
        self.dc_power_total =  sum(self.model_segment(segment).dc for segment in self.segments)
        self.ac_power_total =  sum(self.model_segment(segment).ac for segment in self.segments)
    # Function to model each segment
    def model_segment(self,segment):
        system = pvsystem.PVSystem(
            surface_tilt=segment['tilt'],
            surface_azimuth=segment['azimuth'],
            module_parameters=self.module_parameters,
            temperature_model_parameters=self.temperature_model_parameters,
            modules_per_string=1,
            strings_per_inverter=segment['modules'],
            inverter_parameters=self.inverter_parameters
        )
        # Create the model chain
        mc = modelchain.ModelChain(system, self.site, dc_model='pvwatts', ac_model='pvwatts', aoi_model='physical', spectral_model='no_loss')
        # Debug: print the PVSystem parameters
        #print(f"Modeling segment with parameters: {segment}")

        # Run the model and print intermediate steps
        try:
            mc.run_model(self.weather)

            # Debug: Check intermediate results
            '''
            print("MC Results:", mc.results)
            print("DC Power:", mc.results.dc)
            print("AC Power:", mc.results.ac)
            '''
            # Check if model chain has run successfully
            if mc.results.ac is None:
                print(f"ModelChain did not produce 'ac' for segment: {segment}")
                return pd.Series(0, index=self.weather.index)  # return zero power if error

            # Return AC power
            return mc.results
        except Exception as e:
            print(f"Error modeling segment {segment}: {e}")
            return pd.Series(0, index=self.weather.index)  # return zero power if error
        
    # Function to load and filter weather data
    def load_weather_data(self, file_path, start_time, end_time):
        # Load and label the columns
        weather_data = pd.read_csv(file_path)
        weather_data.columns = [
            'timestamp', 'coordinates', 'model', 'elevation', 'utc_offset',
            'temperature', 'wind_speed', 'ghi', 'dhi', 'dni'
        ]
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])  # Convert to datetime
        weather_data.set_index('timestamp', inplace=True)
        weather_data = weather_data[~weather_data.index.duplicated(keep='first')]

        freq = f'{int(self.dt*60)}min'  # e.g. '15min' or '60min' depending on self.dt

        # Separate numeric and non-numeric columns
        numeric_cols = weather_data.select_dtypes(include='number').columns
        non_numeric_cols = weather_data.select_dtypes(exclude='number').columns

        # Resample and interpolate numeric columns only
        numeric_data = weather_data[numeric_cols].resample(freq).interpolate(method='time')

        # Forward fill non-numeric columns after resampling
        non_numeric_data = weather_data[non_numeric_cols].resample(freq).ffill()

        # Combine numeric and non-numeric columns back together
        weather_data = pd.concat([non_numeric_data, numeric_data], axis=1)

        # Now slice by start_time and end_time
        weather_data = weather_data.loc[start_time:end_time]
        weather_data = weather_data.iloc[:-1]

        return weather_data
            
    def date_format(self):
        day_str = datetime.strptime(self.day, '%m/%d/%Y')
        offset = self.i * self.dt * 60  # total minutes to shift

        start_time_dt = datetime.combine(day_str.date(), datetime.min.time()) + timedelta(minutes=offset)
        end_time_dt = datetime.combine((day_str + timedelta(days=1)).date(), datetime.min.time()) + timedelta(minutes=offset)

        # Format with time reflecting the offset (not fixed to 00:00:00)
        start_time = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_time = end_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        return start_time,end_time
    
if __name__ == "__main__":
    dt = 5/60  # 5-minute timestep
    day = "6/15/2024"
    i = 0  # MPC iteration step

    # Parameter values from argparse config
    pdc0 = 325.6         # DC power at STC (W)
    v_mp = 59.2          # Max power voltage (V)
    i_mp = 5.50          # Max power current (A)
    v_oc = 70.9          # Open circuit voltage (V)
    i_sc = 5.94          # Short circuit current (A)
    alpha_sc = 3.27e-3   # Temp coefficient of Isc (A/C)
    beta_oc = -0.17      # Temp coefficient of Voc (V/C)
    gamma_pdc = -0.00258 # Power temp coefficient (1/C)
    latitude = 40.43093  # Latitude
    longitude = -86.911617  # Longitude

    # Initialize the SolarPanelModel
    solar_model = SolarPanelModel(
        dt, day, pdc0, v_mp, i_mp, v_oc, i_sc,
        alpha_sc, beta_oc, gamma_pdc, latitude, longitude,i
    )

    # Output test information
    print("Start time:", solar_model.start_time)
    print("End time:", solar_model.end_time)
    print("data length:", len(solar_model.weather))
    print("Resulting Data")
    print(solar_model.dc_power_total)