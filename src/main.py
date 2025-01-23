import logging
import args_battery
import args_ev

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse arguments for the battery model
    battery_args = args_battery.parse_battery_args()
    
    # Parse arguments for the electric vehicle model
    ev_args = args_ev.parse_ev_args()

    # Log the parsed battery parameters
    logging.info("Battery Parameters:")
    logging.info(f"  tau_b: {battery_args.tau_b}")
    logging.info(f"  eta_c_b: {battery_args.eta_c_b}")
    logging.info(f"  eta_d_b: {battery_args.eta_d_b}")
    logging.info(f"  x_bar_b: {battery_args.x_bar_b}")
    logging.info(f"  p_c_bar_b: {battery_args.p_c_bar_b}")
    logging.info(f"  p_d_bar_b: {battery_args.p_d_bar_b}")
    logging.info(f"  V_nom_b: {battery_args.V_nom_b}")
    logging.info(f"  P_rated_b: {battery_args.P_rated_b}")

    # Log the parsed electric vehicle parameters
    logging.info("Electric Vehicle Parameters:")
    logging.info(f"  tau_ev: {ev_args.tau_ev}")
    logging.info(f"  eta_c_ev: {ev_args.eta_c_ev}")
    logging.info(f"  eta_d_ev: {ev_args.eta_d_ev}")
    logging.info(f"  x_bar_ev: {ev_args.x_bar_ev}")
    logging.info(f"  p_c_bar_ev: {ev_args.p_c_bar_ev}")
    logging.info(f"  p_d_bar_ev: {ev_args.p_d_bar_ev}")
    logging.info(f"  V_nom_ev: {ev_args.V_nom_ev}")
    logging.info(f"  P_rated_ev: {ev_args.P_rated_ev}")
    logging.info(f"  alpha_ev: {ev_args.alpha_ev}")
    logging.info(f"  Temperature_ev: {ev_args.Temperature_ev}")

if __name__ == "__main__":
    main()