import args_battery
import args_ev
import logging 

def main():
    # Parse arguments for the battery model
    battery_args = args_battery.parse_battery_args()
    
    # Parse arguments for the electric vehicle model
    ev_args = args_ev.parse_ev_args()

    # Now you can use battery_args and ev_args in your model
    # For example, you might want to log or print the parameters
    # Here, we can just log the parameters for demonstration
    logging.info("Battery Parameters:")
    logging.info(f"  tau: {battery_args.tau}")
    logging.info(f"  eta_c: {battery_args.eta_c}")
    logging.info(f"  eta_d: {battery_args.eta_d}")
    logging.info(f"  x_bar: {battery_args.x_bar}")
    logging.info(f"  p_c_bar: {battery_args.p_c_bar}")
    logging.info(f"  p_d_bar: {battery_args.p_d_bar}")

    logging.info("Electric Vehicle Parameters:")
    logging.info(f"  tau: {ev_args.tau}")
    logging.info(f"  eta_c: {ev_args.eta_c}")
    logging.info(f"  eta_d: {ev_args.eta_d}")
    logging.info(f"  x_bar: {ev_args.x_bar}")
    logging.info(f"  p_c_bar: {ev_args.p_c_bar}")
    logging.info(f"  p_d_bar: {ev_args.p_d_bar}")
    logging.info(f"  alpha: {ev_args.alpha}")
    logging.info(f"  Temperature: {ev_args.Temperature}")

if __name__ == "__main__":
    main()