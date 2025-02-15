import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, temperature
from pvlib.location import Location
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

class SolarPanelModel:
    def __init__(self,dt,pdc0,v_mp,i_mp,v_oc,i_sc,alpha_sc,beta_oc,gamma_pdc,latitude,longitude):
        self.dt=dt
        self.pdc0 =pdc0  # DC power at standard test conditions (W)
        self.v_mp = v_mp # Maximum power voltage (V)
        self.i_mp=i_mp # Maximum power current (A)
        self.v_oc = v_oc # Open circuit voltage (V)
        self.i_sc = i_sc  # Short circuit current (A)
        self.alpha_sc = alpha_sc,  # Temperature coefficient of Isc (A/C)
        self.beta_oc = beta_oc,  # Temperature coefficient of Voc (V/C)
        self.gamma_pdc = gamma_pdc  # Power temperature coefficient (1/C)
        self.latitude = latitude # Latitude Co-ordinate of Solar Panel Location (deg)
        self.longitude = longitude # Longitude Co-ordinate of Solar Panel Location (deg)

        # Temperature model parameters
        self.temperature_model_parameters = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

        # Define each segment
        self.segments = [
        {'tilt': 32, 'azimuth': 90, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 50, 'azimuth': 180, 'modules': 3, 'pdc0': 0.975e3},
        {'tilt': 32, 'azimuth': 90, 'modules': 6, 'pdc0': 1.95e3},
        {'tilt': 30, 'azimuth': 270, 'modules': 30, 'pdc0': 9.75e3}
        ]

        # Define the location
        self.site = Location(self.latitude, self.longitude)
        
if __name__ == "__main__":
    dt = 1.0