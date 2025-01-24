import numpy as np
from scipy.signal import cont2discrete

class BatteryModel:
    def __init__(self, dt, tau_b, eta_c_b, eta_d_b, x_bar_b, p_c_bar_b, p_d_bar_b, V_nom_b, P_rated_b):
        self.dt = dt  # time step
        self.tau_b = tau_b
        self.eta_c_b = eta_c_b
        self.eta_d_b = eta_d_b

        self.x_bar_b = x_bar_b
        self.p_c_bar_b = p_c_bar_b
        self.p_d_bar_b = p_d_bar_b

        self.V_nom_b = V_nom_b
        self.P_rated_b = P_rated_b
        self.R_in = self.V_nom_b**2 / self.P_rated_b  # ohms

        # State-space representation
        self.Ad, self.Bd = self.battery_model()

    def battery_model(self):
        A_bat = 0
        B_bat = self.eta_c_b / self.x_bar_b
        C_bat = 1

        # Continuous-time state-space representation
        sys_continuous = (A_bat, B_bat, C_bat, 0)

        # Convert to discrete-time state-space representation
        sys_discrete = cont2discrete(sys_continuous, self.dt, method='zoh')
        Ad, Bd, _, _ = sys_discrete

        return Ad, Bd

# Example usage
if __name__ == "__main__":
    dt = 1.0  # Example time step in hours
    # Example parameters (these should be replaced with actual values from command line arguments)
    tau_b = 800  # Example value
    eta_c_b = 0.95
    eta_d_b = 0.95
    x_bar_b = 13.5
    p_c_bar_b = 5
    p_d_bar_b = 5
    V_nom_b = 384
    P_rated_b = 12.5
    alpha_b = 0.275  # Example value
    Temperature_b = 20  # Example value

    battery_model = BatteryModel(dt, tau_b, eta_c_b, eta_d_b, x_bar_b, p_c_bar_b, p_d_bar_b, V_nom_b, P_rated_b)
    print(f"Ad: {battery_model.Ad}")
    print(f"Bd: {battery_model.Bd}")
