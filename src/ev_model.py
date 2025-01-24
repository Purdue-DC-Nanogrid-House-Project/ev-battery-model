import numpy as np
from scipy.signal import cont2discrete

class EVModel:
    def __init__(self, dt, tau_ev, eta_c_ev, eta_d_ev, x_bar_ev, p_c_bar_ev, p_d_bar_ev, V_nom_ev, P_rated_ev, alpha_ev, Temperature_ev):
        self.dt = dt  # time step
        self.tau_ev = tau_ev
        self.eta_c_ev = eta_c_ev
        self.eta_d_ev = eta_d_ev

        self.x_bar_ev = x_bar_ev
        self.p_c_bar_ev = p_c_bar_ev
        self.p_d_bar_ev = p_d_bar_ev

        self.V_nom_ev = V_nom_ev
        self.P_rated_ev = P_rated_ev
        self.R_in = self.V_nom_ev**2 / self.P_rated_ev  # ohms

        self.alpha_ev = alpha_ev
        self.Temperature_ev = Temperature_ev

        # State-space representation
        self.Ad, self.Bd = self.ev_model()

    def ev_model(self):
        A_bat = 0
        B_bat = self.eta_c_ev / self.x_bar_ev
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
    tau_ev = 800  # Example value
    eta_c_ev = 0.95
    eta_d_ev = 0.95
    x_bar_ev = 13.5
    p_c_bar_ev = 5
    p_d_bar_ev = 5
    V_nom_ev = 384
    P_rated_ev = 12.5
    alpha_ev = 0.275  # Example value
    Temperature_ev = 20  # Example value

    ev_model = EVModel(dt, tau_ev, eta_c_ev, eta_d_ev, x_bar_ev, p_c_bar_ev, p_d_bar_ev, V_nom_ev, P_rated_ev, alpha_ev, Temperature_ev)
    print(f"Ad: {ev_model.Ad}")
    print(f"Bd: {ev_model.Bd}")
