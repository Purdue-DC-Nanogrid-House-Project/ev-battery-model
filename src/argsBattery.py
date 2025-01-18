import argparse
parser = argparse.ArgumentParser(description='Battery Model Parameters')

#Function to test whether efficincies are between 0.00 and 1.00
def check_eta(value):
    """Check if the value is between 0 and 1"""
    f_value = float(value)  # Convert the input to float
    if f_value < 0 or f_value > 1:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
    return f_value

#Function to test whether capacities and time constant are positive
def check_pos(value):
    """Check if the value is less than 0"""
    f_value = float(value)  # Convert the input to float
    if f_value < 0:
        raise argparse.ArgumentTypeError(f"{value} is less than 0")
    return f_value



# Efficiency Parameters
parser.add_argument('--tau', type=check_pos, default=(800+2400)/2, help='Dissipation Time Constant (h) [typically between 800-2400 h]')
parser.add_argument('--eta_c', type=check_eta, default=0.95, help='Charging Efficiency (%))[eg. 0.95,0.90]')
parser.add_argument('--eta_d', type=check_eta, default=0.95, help='Discharging Efficiency (%)[eg. 0.95,0.90]')

# Capacity Parameters 
parser.add_argument('--x_bar', type=check_pos, default=13.5, help='Chemical Energy Capacity (kWh)[eg. 15,13.5]')
parser.add_argument('--p_c_bar', type=check_pos, default=5, help='Electrical Charging Capacity (kW)[eg. 5,6]')
parser.add_argument('--p_d_bar', type=check_pos, default=5, help='Electrical Discharging Capacity (kW)[eg. 5,6]')

args = parser.parse_args()

# Print the parsed arguments for verification
print(f"Battery Model Parmaters:")
print(f" ")
print(f"tau: {args.tau}")
print(f"eta_c: {args.eta_c}")
print(f"eta_d: {args.eta_d}")
print(f"x_bar: {args.x_bar}")
print(f"p_c_bar: {args.p_c_bar}")
print(f"p_d_bar: {args.p_d_bar}")


