import pandas as pd
import pvlib
from pvlib import pvsystem, modelchain, temperature
from pvlib.location import Location
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

class SolarPanelModel:
    def __init__(self,dt):
        self.dt=dt
        self.pdc0 =25.6  # DC power at standard test conditions (W)
        self.v_mp = 59.2 # Maximum power voltage (V)
        self.i_mp=5.50 # Maximum power current (A)
        self.v_oc = 70.9 # Open circuit voltage (V)
        self.i_sc = 5.94  # Short circuit current (A)
        self.alpha_sc = 3.27e-3,  # Temperature coefficient of Isc (A/C)
        self.beta_oc = -0.17,  # Temperature coefficient of Voc (V/C)
        self.gamma_pdc = -0.00258  # Power temperature coefficient (1/C)

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
        latitude, longitude = 40.43093, -86.911617
        self.site = Location(latitude, longitude)
        
if __name__ == "__main__":
    dt = 1.0