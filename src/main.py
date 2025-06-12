import logging
import pandas as pd
import args_handler

# Import the test function
from utils import initialize_models,evbm_optimization_v3,plot_optimizer_results
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

    #Single optimizer check
    i = 24
    dt = 5/60 # Example time step in hours - 5 mins
    optimizer = initialize_models(model_args, dt,i)
    [x_b,x_ev,P_bat,P_ev,P_util, P_sol,P_dem] = evbm_optimization_v3(optimizer,8000,i)
    plot_optimizer_results(x_b,x_ev,P_bat,P_ev,P_util,P_sol,P_dem,dt,model_args.day,8000)

    #MPC - iterative optimizer

if __name__ == "__main__":
    main()