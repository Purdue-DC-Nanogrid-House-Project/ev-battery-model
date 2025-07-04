import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class HVAC:
    def __init__(self, dt, T_0, i):
        self.dt = dt
        self.T_0 = T_0
        self.i = i

        # Time series inputs
        self.T_set = self.get_T_set(i)
        self.theta = self.get_theta(i)

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
        self.I_gain = 0.0004 * (5 * 60) / 2.0

        # Power characteristics
        self.Q_aux1 = 9.6  # kW
        self.Q_aux2 = 0.0  # kW (disabled here)
        self.pf_hp = 0.8
        self.pf_aux = 1.0
        self.P_nominal = 1.0

        # Timing thresholds
        self.n_hp = 5 * (5 / 60) / self.dt
        self.n_aux1 = 12 * (5 / 60) / self.dt
        self.l = 100 * (5 * 60) / 2.0

    def compute_eta(self, theta):
        eta = np.ones_like(theta)
        above = theta >= -24
        eta[above] = 0.0008966 * theta[above]**2 + 0.1074 * theta[above] + 3.098
        return eta

    def get_T_set(self, i):
        n_steps = int(24 / self.dt)
        T_set_full = np.full(2 * n_steps, 22)
        return T_set_full[i:n_steps + i]

    def get_theta(self, i):
        n_steps = int(24 / self.dt)
        t = np.linspace(0, 48, 2 * n_steps)
        theta_full = 18 + 5 * np.sin(2 * np.pi * (t - 14) / 24)
        return theta_full[i:n_steps + i]

    def HP_emulator(self, T, Tset, theta):
        theta = float(theta)
        eta = self.compute_eta(np.array([theta]))[0]
        PmaxHP = min(-0.0149 * theta + 4.0934, 4.0)
        Q_hp = eta * PmaxHP

        # Dynamics
        R = (self.Rout * self.Rm) / (self.Rout + self.Rm)
        R *= 1.15
        a = np.exp(-self.dt / (R * self.C))
        theta_cor = (self.Rm * theta + self.Rout * self.Tm) / (self.Rout + self.Rm)

        # PI Control
        e = max(Tset - T, 0)
        e_I = np.clip(self.prev_I + e * self.dt, -self.l, self.l)

        if T > Tset + self.db / 2:
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

            if (PI_output_dim > 2 and self.hp_timer > self.n_hp) or (self.aux1_timer > 0 and self.aux2_timer == 0):
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
        I_total = 1000 * I_total / 240  # Convert to A

        return T_next, Q_total, I_total

    def loop_emulator(self):
        n_steps = int(24 / self.dt)
        T = self.T_0

        T_array = np.zeros(n_steps)
        Q_total_array = np.zeros(n_steps)
        I_total_array = np.zeros(n_steps)

        for k in range(n_steps):
            T_next, Q_total, I_total = self.HP_emulator(T, self.T_set[k], self.theta[k])
            T_array[k] = T
            Q_total_array[k] = Q_total
            I_total_array[k] = I_total
            T = T_next

        return T_array.reshape(-1, 1), Q_total_array.reshape(-1, 1), I_total_array.reshape(-1, 1)
if __name__ == "__main__":
    # Parameters
    dt_fast = 5 / 60  # 5 minutes in hours
    hours = 24
    n_steps = int(hours / dt_fast)
    T_init = 18.0

    # Time vector for plotting
    time_hours = np.arange(n_steps) * dt_fast

    # Run HVAC simulation
    hvac = HVAC(dt=dt_fast, T_0=T_init, i=0)
    T_array, Q_total_array, I_total_array = hvac.loop_emulator()

    # Plotting
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(time_hours, T_array, 'b-', label='Room Temp')
    plt.plot(time_hours, hvac.T_set, 'r--', label='Setpoint')
    plt.ylabel("Room Temp (°C)")
    plt.title("Room Temperature Profile")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_hours, hvac.theta, 'k-', label='Ambient Temp')
    plt.ylabel("Ambient Temp (°C)")
    plt.title("Ambient Temperature (Theta)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_hours, Q_total_array, 'm-', label='Power Draw (kW)')
    plt.ylabel("Power (kW)")
    plt.grid(True)
    plt.twinx()
    plt.plot(time_hours, I_total_array, 'c--', label='Current (A)')
    plt.ylabel("Current (A)")
    plt.title("Heating Power and Current Draw")
    plt.xlabel("Time (hours)")

    plt.tight_layout()
    plt.show()