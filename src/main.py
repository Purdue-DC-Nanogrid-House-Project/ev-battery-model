import logging
import args_handler

from ev_model import EVModel  # Import the EVModel class
from battery_model import BatteryModel  # Import the BatteryModel class
from utility_model import UtilityModel #Import UtilityModel class
from home_model import HomeModel # Import UtilityModel Class


from test_models import test_ev_charging,test_ev_charging_v2 # Import the test function

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

        # Create instances of the models
        dt = 1/60  # Example time step in hours

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
        home_demand = 15 #KW
        home_model = HomeModel(
            dt = dt,
            demand = home_demand
        )

        # # Run basic model test
        # test_ev_charging(ev_model, charger_model, initial_charge=0.5, target_charge=0.9)

        # Run test with Utility and Home
        ## Case 1
        home_model.demand = 15 #(kW), EV call for 6(kW)
        test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.7, target_charge=0.8, ev_call = 6)

        ## Case 2
        home_model.demand = 8 #(kW), EV call for 13.5 (kW)
        test_ev_charging_v2(ev_model,charger_model,home_model,utility_model,initial_charge=0.7, target_charge=0.8, ev_call = ev_model.p_c_bar_ev)

if __name__ == "__main__":
    main()