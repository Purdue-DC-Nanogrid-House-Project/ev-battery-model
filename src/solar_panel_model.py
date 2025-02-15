import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, temperature
from pvlib.location import Location
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

class SolarPanelModel:
    def __init__(self,dt,pdc0,v_mp,i_mp,v_oc,i_sc,alpha_sc,beta_oc,gamma_pdc,latitude,longitude):
        # Location Paramaters
        self.latitude = latitude # Latitude Co-ordinate of Solar Panel Location (deg)
        self.longitude = longitude # Longitude Co-ordinate of Solar Panel Location (deg)

        # Weather Parameters
        weather_path = 'data/HistoricalWeather.csv'
        start_time = '2023-06-19 20:00:00'
        end_time = '2024-06-19 19:00:00'
        self.weather = self.load_weather_data(weather_path,start_time,end_time)

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
        self.segment = [
        {'tilt': 32, 'azimuth': 90, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 50, 'azimuth': 180, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 32, 'azimuth': 90, 'modules': 6, 'pdc0': 1.95e3},
        {'tilt': 30, 'azimuth': 270, 'modules': 30, 'pdc0': 9.75e3}
        ]

        # Define the location
        self.site = Location(self.latitude, self.longitude)

        # Create solar panel model
        self.mc = self.model_segment()

    # Function to model each segment
    def model_segment(self):
        system = pvsystem.PVSystem(
            surface_tilt=self.segment['tilt'],
            surface_azimuth=self.segment['azimuth'],
            module_parameters=self.module_parameters,
            temperature_model_parameters=self.temperature_model_parameters,
            modules_per_string=1,
            strings_per_inverter=self.segment['modules'],
            inverter_parameters=self.inverter_parameters
        )
        # Create the model chain
        mc = modelchain.ModelChain(system, self.site, dc_model='pvwatts', ac_model='pvwatts', aoi_model='physical', spectral_model='no_loss')
        
        return mc
    
    # Function to load and filter weather data
    def load_weather_data(self,file_path, start_time, end_time):
        weather_data = pd.read_csv(file_path, parse_dates=[0])
        weather_data.columns = [
            'timestamp', 'coordinates', 'model', 'elevation', 'utc_offset',
            'temperature', 'wind_speed', 'ghi', 'dhi', 'dni'
        ]
        weather_data = weather_data.set_index('timestamp')
        weather_data = weather_data[start_time:end_time]
        return weather_data
        
if __name__ == "__main__":
    dt = 1.0