import numpy as np
import control

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
        self.sys_d = self.ev_model()

    def ev_model(self):
        A_ev = np.array([[0]])  # State matrix (1x1 matrix)
        B_ev = np.array([[self.eta_c_ev / self.x_bar_ev]])  # Input matrix (1x1 matrix)
        C_ev = np.array([[1]])  # Output matrix (1x1 matrix)
        D_ev = np.array([[0]])  # Feedforward matrix (1x1 matrix)

        # Continuous-time state-space representation
        sys_continuous = control.ss(A_ev, B_ev, C_ev, D_ev)

        # Discrete-time stat space representation

        sys_discrete = control.sample_system(sys_continuous,dt, method='zoh')

        return sys_discrete
    
# Example usage
if __name__ == "__main__":
    dt = 1.0  # Example time step in hours
    # Example parameters (these should be replaced with actual values from command line arguments)
    tau_ev = 1600  # Example value
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
    print(f"sys_d: {ev_model.sys_d}")
