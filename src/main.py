import logging
import args_handler
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse arguments for the battery model
    model_args = args_handler.args_handler()

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

if __name__ == "__main__":
    main()