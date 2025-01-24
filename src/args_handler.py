import argparse
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_eta(value):
    """Check if the value is between 0 and 1"""
    f_value = float(value)  # Convert the input to float
    if f_value < 0 or f_value > 1:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1 (expected a value between 0 and 1)")
    return f_value

def check_pos(value):
    """Check if the value is greater than or equal to 0"""
    f_value = float(value)  # Convert the input to float
    if f_value < 0:
        raise argparse.ArgumentTypeError(f"{value} is less than 0 (expected a value >= 0)")
    return f_value

def args_handler():
    """Parse command line arguments for battery model parameters."""
    parser = argparse.ArgumentParser(description='Charger Battery Model (b) and Electric Vehicle(ev) Parameters')

    # Charger Battery
        ## Efficiency Parameters
    parser.add_argument('--tau_b', type=check_pos, default=(800 + 2400) / 2, help='Dissipation Time Constant (h) [typically between 800-2400 h]')
    parser.add_argument('--eta_c_b', type=check_eta, default=0.95, help='Charging Efficiency (fraction, e.g., 0.95 for 95%%)')
    parser.add_argument('--eta_d_b', type=check_eta, default=0.95, help='Discharging Efficiency (fraction, e.g., 0.95 for 95%%)')
        ## Capacity Parameters 
    parser.add_argument('--x_bar_b', type=check_pos, default=13.5, help='Chemical Energy Capacity (kWh) [e.g., 15, 13.5]')
    parser.add_argument('--p_c_bar_b', type=check_pos, default=5, help='Electrical Charging Capacity (kW) [e.g., 5, 6]')
    parser.add_argument('--p_d_bar_b', type=check_pos, default=5, help='Electrical Discharging Capacity (kW) [e.g., 5, 6]')
        ##Power rating and resistance parameters
    parser.add_argument('--V_nom_b', type=check_pos, default=384, help='Nominal Voltage (V) [e.g., 380, 384]')
    parser.add_argument('--P_rated_b', type=check_pos, default=12.5, help='Power Rating (kWh) [e.g., 12.5,13.5]')

    # Electric Vehicle
        ## Efficiency Parameters
    parser.add_argument('--tau_ev', type=check_pos, default=(800 + 2400) / 2, help='Dissipation Time Constant (h) [typically between 800-2400 h]')
    parser.add_argument('--eta_c_ev', type=check_eta, default=0.95, help='Charging Efficiency (fraction, e.g., 0.95 for 95%%)')
    parser.add_argument('--eta_d_ev', type=check_eta, default=0.95, help='Discharging Efficiency (fraction, e.g., 0.95 for 95%%)')
        ## Capacity Parameters 
    parser.add_argument('--x_bar_ev', type=check_pos, default=13.5, help='Chemical Energy Capacity (kWh) [e.g., 15, 13.5]')
    parser.add_argument('--p_c_bar_ev', type=check_pos, default=5, help='Electrical Charging Capacity (kW) [e.g., 5, 6]')
    parser.add_argument('--p_d_bar_ev', type=check_pos, default=5, help='Electrical Discharging Capacity (kW) [e.g., 5, 6]')
        ## Power rating and resistance parameters
    parser.add_argument('--V_nom_ev', type=check_pos, default=384, help='Nominal Voltage (V) [e.g., 380, 384]')
    parser.add_argument('--P_rated_ev', type=check_pos, default=12.5, help='Power Rating (kWh) [e.g., 12.5,13.5]')
        ## Intensity Parameters
    parser.add_argument('--alpha_ev', type=check_pos, default=(0.15 + 0.4) / 2, help='Energy Intensity (kWh/km) [e.g., 0.15,0.4]')
    parser.add_argument('--temperature_ev', type=float, default=-1, help='Temperature (F) [e.g.68,20]')
        ## Use Paramaters
    parser.add_argument('--distance',type=check_pos, default=10, help='Distance Driven (km) [e.g. 30, 50]')



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args_handler()