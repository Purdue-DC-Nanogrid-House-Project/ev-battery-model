import argparse

parser = argparse.ArgumentParser(description='Electriv Vehicle Model Parameters')

# Default to Tesla Powerwall Values
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

# Efficiency Parameters
parser.add_argument('--tau', type=check_pos, default=(800 + 2400) / 2, help='Dissipation Time Constant (h) [typically between 800-2400 h]')
parser.add_argument('--eta_c', type=check_eta, default=0.95, help='Charging Efficiency (fraction, e.g., 0.95 for 95%%)')
parser.add_argument('--eta_d', type=check_eta, default=0.95, help='Discharging Efficiency (fraction, e.g., 0.95 for 95%%)')

# Capacity Parameters 
parser.add_argument('--x_bar', type=check_pos, default=13.5, help='Chemical Energy Capacity (kWh) [e.g., 15, 13.5]')
parser.add_argument('--p_c_bar', type=check_pos, default=5, help='Electrical Charging Capacity (kW) [e.g., 5, 6]')
parser.add_argument('--p_d_bar', type=check_pos, default=5, help='Electrical Discharging Capacity (kW) [e.g., 5, 6]')

# Power rating and resistance parameters
parser.add_argument('--V_nom', type=check_pos, default=384, help='Nominal Voltage (V) [e.g., 380, 384]')
parser.add_argument('--P_rated', type=check_pos, default=12.5, help='Power Rating (kWh) [e.g., 12.5,13.5]')

# Intensity Paramaters
parser.add_argument('--alpha', type=check_pos, default=(0.15+0.4)/2, help='Energy Intensity (kWh/km) [e.g., 0.15,0.4]')
parser.add_argument('--Temperature', type=float, default=-1, help='Temperature (F) [e.g.68,20]')




args = parser.parse_args()


# Print the parsed arguments for verification
print(f"tau: {args.tau}")
print(f"eta_c: {args.eta_c}")
print(f"eta_d: {args.eta_d}")
print(f"x_bar: {args.x_bar}")
print(f"p_c_bar: {args.p_c_bar}")
print(f"p_d_bar: {args.p_d_bar}")