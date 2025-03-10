import logging
import pandas as pd
import args_handler

from ev_model import EVModel  # Import the EVModel class
from battery_model import BatteryModel  # Import the BatteryModel class
from utility_model import UtilityModel #Import UtilityModel class
from home_model import HomeModel # Import UtilityModel Class
from solar_panel_model import SolarPanelModel # Import SolarPanelModel Class
from optimizer  import Optimizer #Import Optimizer Class

# Import the test function
from test_models import test_ev_charging,test_ev_charging_v2,test_solar_model,test_ev_charging_v3,evbm_optimization_v1,plot_results 

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse arguments for the battery model
    model_args = args_handler.args_handler()

    if model_args.check_params:
        # Log the parsed battery parameters
        logging.info("Battery Parameters:")
        logging.info(f"  tau_b: {model_args.tau_b} h")
        logging.info(f"  eta_c_b: {model_args.eta_c_b} %")
        logging.info(f"  eta_d_b: {model_args.eta_d_b} %")
        logging.info(f"  x_bar_b: {model_args.x_bar_b} kWh")
        logging.info(f"  p_c_bar_b: {model_args.p_c_bar_b} kW")
        logging.info(f"  p_d_bar_b: {model_args.p_d_bar_b} kW")
        logging.info(f"  V_nom_b: {model_args.V_nom_b} V")
        logging.info(f"  P_rated_b: {model_args.P_rated_b} kW")
        print(f" ")

        # Log the parsed electric vehicle parameters
        logging.info("Electric Vehicle Parameters:")
        logging.info(f"  tau_ev: {model_args.tau_ev} h")
        logging.info(f"  eta_c_ev: {model_args.eta_c_ev} %")
        logging.info(f"  eta_d_ev: {model_args.eta_d_ev} %")
        logging.info(f"  x_bar_ev: {model_args.x_bar_ev} kWh")
        logging.info(f"  p_c_bar_ev: {model_args.p_c_bar_ev} kW")
        logging.info(f"  p_d_bar_ev: {model_args.p_d_bar_ev} kW")
        logging.info(f"  V_nom_ev: {model_args.V_nom_ev} V")
        logging.info(f"  P_rated_ev: {model_args.P_rated_ev} kW")
        logging.info(f"  alpha_ev: {model_args.alpha_ev} kWh/km")
        logging.info(f"  temperature_ev: {model_args.temperature_ev} F")
        logging.info(f"  distance: {model_args.distance} km")
        print(f" ")

        # Log the parsed solar panel parameters
        logging.info("Solar Panel Parameters:")
        logging.info(f"  pdc0: {model_args.pdc0} W")
        logging.info(f"  v_mp: {model_args.v_mp} V")
        logging.info(f"  i_mp: {model_args.i_mp} A")
        logging.info(f"  v_oc: {model_args.v_oc} V")
        logging.info(f"  i_sc: {model_args.i_sc} A")
        logging.info(f"  alpha_sc: {model_args.alpha_sc} A/C")
        logging.info(f"  beta_oc: {model_args.beta_oc} V/C")
        logging.info(f"  gamma_pdc: {model_args.gamma_pdc} 1/C")
        logging.info(f"  latitude: {model_args.latitude} deg")
        logging.info(f"  longitude: {model_args.longitude} deg")
        print(f" ")
        
    # Create instances of the models
    dt = 1/60 # Example time step in hours

    # Create BatteryModel instance
    charger_model = BatteryModel(
        dt=dt,
        tau_b=model_args.tau_b,
        eta_c_b=model_args.eta_c_b,
        eta_d_b=model_args.eta_d_b,
        x_bar_b=model_args.x_bar_b,
        p_c_bar_b=model_args.p_c_bar_b,
        p_d_bar_b=model_args.p_d_bar_b,
        V_nom_b=model_args.V_nom_b,
        P_rated_b=model_args.P_rated_b
    )

    # Create EVModel instance
    ev_model = EVModel(
        dt=dt,
        tau_ev=model_args.tau_ev,
        eta_c_ev=model_args.eta_c_ev,
        eta_d_ev=model_args.eta_d_ev,
        x_bar_ev=model_args.x_bar_ev,
        p_c_bar_ev=model_args.p_c_bar_ev,
        p_d_bar_ev=model_args.p_d_bar_ev,
        V_nom_ev=model_args.V_nom_ev,
        P_rated_ev=model_args.P_rated_ev,
        alpha_ev=model_args.alpha_ev,
        Temperature_ev=model_args.temperature_ev
    )

    # Create Utility Model Instance
    utility_model = UtilityModel(
        dt = dt,
        utility = 0
    )

    # Create Home Model Instance
    home_model = HomeModel(
        dt = dt,
        demand = 0
    )

    # Create Solar Panel Model Instance
    solar_model = SolarPanelModel(
        dt = dt,
        pdc0 = model_args.pdc0, 
        v_mp = model_args.v_mp, 
        i_mp= model_args.i_mp,
        v_oc = model_args.v_oc,
        i_sc = model_args.i_sc ,
        alpha_sc = model_args.alpha_sc, 
        beta_oc = model_args.beta_oc,
        gamma_pdc = model_args.gamma_pdc, 
        latitude = model_args.latitude,
        longitude = model_args.longitude
    )

    # Create Optimizer Instance
    optimizer = Optimizer(
        dt = dt,
        battery_model = charger_model,
        home_model= home_model,
        x0 = 0.5
    )

    # Print Solar results for example
    # time_range = pd.to_datetime(solar_model.dc_power_total[0:-1].index)
    # time_range = (time_range).strftime('%H:%M')
    # power = (solar_model.dc_power_total[0:-1].values)/1000
    # # Creating a DataFrame with 'Time (hours)' and 'Power'
    # df_solar_results = pd.DataFrame({
    #     'Time (hours)': time_range,
    #     'Solar Power (kW)': power
    # })

    # # Run basic model test
    # test_ev_charging(ev_model, charger_model, initial_charge=0.5, target_charge=0.9)

    # # Run test with Utility and Home
    # # ## Case 1
    # home_model.demand = 15 #(kW), EV call for 4.0 (kW)
    # test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.7, target_charge=0.8, ev_call = 4)

    # ## Case 2
    # home_model.demand = 8 #(kW), EV call for 5.0 (kW) [Max]
    # test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.5, target_charge=0.9, ev_call = ev_model.p_c_bar_ev)

    #Plot Solar Output on chosen day
    # test_solar_model(solar_model,dt)

    # Run test case with Utility, Home, and Solar
    #home_model.demand = 15 #(kW), EV call for 5.0 (kW) [Max]
    #test_ev_charging_v3(ev_model,charger_model,home_model,utility_model,solar_model,initial_charge_pre=0.8, initial_charge_post=0.6, target_charge= 1.0)

    # Run test case of optimizer with just Utility
    home_model.demand = 15 #(kW)
    [x_b, P_bat, P_util] = evbm_optimization_v1(optimizer)
    plot_results(x_b, P_bat, P_util,dt)
    

if __name__ == "__main__":
    main()