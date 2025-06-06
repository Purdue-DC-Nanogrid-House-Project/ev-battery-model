import numpy as np
import control

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
        self.sys_d = self.battery_model_v2()
        # self.sys_d2 = self.battery_model_v2()
        # print(f"Old Model:",self.sys_d)
        # print(f"New Model:",self.sys_d)

    def battery_model(self):
        A_bat = np.array([[0]])  # State matrix (1x1 matrix)
        B_bat = np.array([[1]])  # Input matrix (1x1 matrix)
        C_bat = np.array([[1]])  # Output matrix (1x1 matrix)
        D_bat = np.array([[0]])  # Feedforward matrix (1x1 matrix)

        # Continuous-time state-space representation
        sys_continuous = control.ss(A_bat, B_bat, C_bat, D_bat)

        # Convert to discrete-time state-space representation
        sys_discrete = control.sample_system(sys_continuous,self.dt, method='zoh')

        return sys_discrete
    
    def battery_model_v2(self):
        A_bat = np.array([[-1/self.tau_b]])  # State matrix (1x1 matrix)
        B_bat = np.array([[1]])  # Input matrix (1x1 matrix)
        C_bat = np.array([[1]])  # Output matrix (1x1 matrix)
        D_bat = np.array([[0]])  # Feedforward matrix (1x1 matrix)

        # Continuous-time state-space representation
        sys_continuous = control.ss(A_bat, B_bat, C_bat, D_bat)

        # Convert to discrete-time state-space representation
        sys_discrete = control.sample_system(sys_continuous,self.dt, method='zoh')

        return sys_discrete

# Example usage
if __name__ == "__main__":
    dt = 1.0  # Example time step in hours
    # Example parameters (these should be replaced with actual values from command line arguments)
    tau_b = 1600  # Example value
    eta_c_b = 0.95
    eta_d_b = 0.95
    x_bar_b = 13.5
    p_c_bar_b = 5
    p_d_bar_b = 5
    V_nom_b = 384
    P_rated_b = 12.5

    battery_model = BatteryModel(dt, tau_b, eta_c_b, eta_d_b, x_bar_b, p_c_bar_b, p_d_bar_b, V_nom_b, P_rated_b)
    print(f"sys_d: {battery_model.sys_d}")
    
