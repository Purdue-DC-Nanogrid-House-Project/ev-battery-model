import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class HVAC:
    def __init__(self, dt, T_0, T_set,theta, i):
        self.dt = dt
        self.T_0 = T_0
        self.T_set = T_set
        self.theta = theta
        self.i = i
        
        # Heat pump & tank constants
        self.Rout = 2.0424 * 1.15
        self.Rm = 1.0616
        self.C = 6.5 * 1.1
        self.db = 1.0
        self.Tm = 20.61

        # PI-controller state
        self.prev_I = 0.0
        self.aux1_timer = 0
        self.aux2_timer = 0
        self.hp_timer = 0

        # PI gains
        self.P_gain = 0.4 * 20.0
        self.I_gain = 1.0 * 0.0004 * (5*60)/2.0

        # Power and thresholds
        self.Q_aux1 = 9.6  # kW
        self.Q_aux2 = 0.0  # kW
        self.pf_hp = 0.8
        self.pf_aux = 1.0
        self.P_nominal = 1.0

        self.n_hp = 5 * (5/60) / self.dt
        self.n_aux1 = 12 * (5/60) / self.dt
        self.l = 100 * (5*60)/2.0

    def compute_eta(self, theta):
        eta = np.ones_like(theta)
        above = theta >= -24
        eta[above] = 0.0008966 * theta[above]**2 + 0.1074 * theta[above] + 3.098
        return eta

    def HP_emulator(self, T, Tset, theta):
        theta = float(theta)
        eta = self.compute_eta(np.array([theta]))[0]
        PmaxHP = min(-0.0149 * theta + 4.0934, 4.0)
        Q_hp = eta * PmaxHP

        # Effective resistance and a-factor
        R = (self.Rout * self.Rm) / (self.Rout + self.Rm)
        R *= 1.15
        a = np.exp(-self.dt / (R * self.C))
        theta_cor = (self.Rm * theta + self.Rout * self.Tm) / (self.Rout + self.Rm)

        # PI Control
        e = max(Tset - T, 0)
        e_I = np.clip(self.prev_I + e * self.dt, -self.l, self.l)

        if T > Tset + self.db/2:
            e = 0
            e_I = 0
            self.prev_I = 0

        PI_output = self.P_gain * e + self.I_gain * e_I
        PI_output_dim = PI_output / self.P_nominal

        Q_total = 0.0
        P_total = 0.0
        I_total = 0.0

        if PI_output_dim > 0:
            Q_total = Q_hp
            P_total = Q_hp / eta
            I_total = (1 / self.pf_hp) * (Q_hp / eta)

            if PI_output_dim > 2:
                self.hp_timer += 1
            elif PI_output_dim < 1:
                self.hp_timer = 0
                self.aux1_timer = 0

            if (PI_output_dim > 2 and self.hp_timer > self.n_hp) or \
                (self.aux1_timer > 0 and self.aux2_timer == 0):
                if self.aux1_timer <= self.n_aux1:
                    self.aux1_timer += 1
                    Q_total += self.Q_aux1
                    P_total += self.Q_aux1
                    I_total += (1 / self.pf_aux) * self.Q_aux1
                else:
                    self.aux1_timer = 0
                    self.hp_timer = 0

            if PI_output_dim > 4 and self.hp_timer > self.n_hp:
                Q_total += self.Q_aux2
                P_total += self.Q_aux2
                I_total += (1 / self.pf_aux) * self.Q_aux2

        T_next = a * T + (1 - a) * (theta_cor + R * (Q_total + 2))
        self.prev_I = e_I
        I_total = 1000 * I_total / 240  # convert to A

        return T_next, self.prev_I, Q_total, self.aux1_timer, self.aux2_timer, self.hp_timer, I_total
    
if __name__ == "__main__":
    # Simulation parameters
    dt_fast = 5 / 60  # timestep in hours (5 minutes)
    hours = 24
    n_steps = int(hours / dt_fast)

    # Initial conditions
    T = 18.0              # Initial room temperature (°C)
    Tset = 22.0           # Setpoint temperature (°C)
    prev_I = 0.0
    aux1_timer = 0
    aux2_timer = 0
    hp_timer = 0

    # Time and ambient temp
    time_hours = np.arange(n_steps) * dt_fast
    theta_array = np.full(n_steps, -5.0)

    # Preallocate result arrays
    T_array = np.zeros(n_steps)
    Tset_array = np.full(n_steps, Tset)
    Q_total_array = np.zeros(n_steps)
    I_total_array = np.zeros(n_steps)

    # Initialize HVAC object
    hvac = HVAC(dt_fast, T, Tset, theta_array[0], i=0)

    # Run simulation
    for k in range(n_steps):
        theta_k = theta_array[k]
        T_next, prev_I, Q_total, aux1_timer, aux2_timer, hp_timer, I_total = hvac.HP_emulator(
            T, Tset, theta_k
        )

        # Store results
        T_array[k] = T
        Q_total_array[k] = Q_total
        I_total_array[k] = I_total

        # Update for next time step
        T = T_next

    # Plotting
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_hours, T_array, 'b-', label='Room Temp', linewidth=1.5)
    plt.plot(time_hours, Tset_array, 'r--', label='Setpoint', linewidth=1.2)
    plt.ylabel("Room Temp (°C)")
    plt.title("Room Temperature Profile")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_hours, theta_array, 'k-', linewidth=1.5)
    plt.ylabel("Ambient Temp (°C)")
    plt.title("Ambient Temperature (Theta)")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time_hours, Q_total_array, 'm-', label='Power Draw (kW)', linewidth=1.5)
    plt.ylabel("Power Draw (kW)")
    plt.grid(True)
    plt.twinx()
    plt.plot(time_hours, I_total_array, 'c--', label='Current (A)', linewidth=1.5)
    plt.ylabel("Current (A)")
    plt.title("Heating Power and Current Draw")
    plt.xlabel("Time (hours)")

    plt.tight_layout()
    plt.show()
