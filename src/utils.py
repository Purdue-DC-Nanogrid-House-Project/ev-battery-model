from ev_model import EVModel  # Import the EVModel class
from battery_model import BatteryModel  # Import the BatteryModel class
from utility_model import UtilityModel #Import UtilityModel class
from home_model import HomeModel # Import UtilityModel Class
from solar_panel_model import SolarPanelModel # Import SolarPanelModel Class
from optimizer  import Optimizer #Import Optimizer Class

import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import numpy as np
import os
import re

def initialize_models(model_args, dt,i):
    charger_model = BatteryModel(
        dt=dt,
        tau_b=model_args.tau_b,
        eta_c_b=model_args.eta_c_b,
        eta_d_b=model_args.eta_d_b,
        x_bar_b=model_args.x_bar_b,
        p_c_bar_b=model_args.p_c_bar_b,
        p_d_bar_b=model_args.p_d_bar_b,
        V_nom_b=model_args.V_nom_b,
        P_rated_b=model_args.P_rated_b
    )

    ev_model = EVModel(
        dt=dt,
        tau_ev=model_args.tau_ev,
        eta_c_ev=model_args.eta_c_ev,
        eta_d_ev=model_args.eta_d_ev,
        x_bar_ev=model_args.x_bar_ev,
        p_c_bar_ev=model_args.p_c_bar_ev,
        p_d_bar_ev=model_args.p_d_bar_ev,
        V_nom_ev=model_args.V_nom_ev,
        P_rated_ev=model_args.P_rated_ev,
        alpha_ev=model_args.alpha_ev,
        Temperature_ev=model_args.temperature_ev,
        time_leave=model_args.time_leave,
        time_arrive=model_args.time_arrive,
        distance=model_args.distance
    )

    utility_model = UtilityModel(
        dt=dt,
        utility=0
    )

    home_model = HomeModel(
        dt=dt,
        day=model_args.day,
        i=i
    )

    solar_model = SolarPanelModel(
        dt=dt,
        day=model_args.day,
        pdc0=model_args.pdc0,
        v_mp=model_args.v_mp,
        i_mp=model_args.i_mp,
        v_oc=model_args.v_oc,
        i_sc=model_args.i_sc,
        alpha_sc=model_args.alpha_sc,
        beta_oc=model_args.beta_oc,
        gamma_pdc=model_args.gamma_pdc,
        latitude=model_args.latitude,
        longitude=model_args.longitude,
        i = i
    )

    optimizer = Optimizer(
        dt=dt,
        battery_model=charger_model,
        ev_model=ev_model,
        home_model=home_model,
        solar_model=solar_model,
        x0_b=0.5,
        x0_ev=0.5
    )

    return optimizer

def evbm_optimization_v2(optimizer,weight):
    # Define variables
    x_b = cp.Variable((1, optimizer.K+1))  # Battery SOC
    x_ev = cp.Variable((1, optimizer.K+1))  # EV SOC
    P_bat_c = cp.Variable((optimizer.K, 1))  # Battery charging
    P_bat_d = cp.Variable((optimizer.K, 1))  # Battery discharging
    P_ev_c = cp.Variable((optimizer.K, 1))   # EV charging
    P_ev_d = cp.Variable((optimizer.K, 1))   # EV discharging
    P_bat = cp.Variable((optimizer.K, 1))  
    P_ev = cp.Variable((optimizer.K, 1))  
    P_util = cp.Variable((optimizer.K, 1))   # Grid power
    t_leave = optimizer.ev_model.time_leave
    t_arrive = optimizer.ev_model.time_arrive
    K_leave = int(t_leave / optimizer.dt)
    K_arrive = int(t_arrive / optimizer.dt)
    # print("Index Leave",{K_leave})
    # print("Index arrive",{K_arrive})

    # Known data
    time_range = np.arange(0, 24, optimizer.dt)

    solar_power = (optimizer.solar_model.dc_power_total[0:-1].values)/1000
    P_sol = np.interp(time_range, np.linspace(0, 23, len(solar_power)), solar_power).reshape(-1, 1)

    c_elec = np.where((time_range >= 14) & (time_range <= 20), 0.2573, 0.0825).reshape(-1, 1)
    P_dem = optimizer.home_model.demand.to_numpy().reshape(-1, 1)
    ev_plugged = np.ones_like(P_sol)
    ev_plugged[K_leave:K_arrive] = 0


    # Constraints
    constraints = [
        # Grid power balance
        P_bat == optimizer.battery_model.eta_c_b * P_bat_c - (1 / optimizer.battery_model.eta_d_b) * P_bat_d,
        P_ev == optimizer.ev_model.eta_c_ev * P_ev_c - (1 / optimizer.ev_model.eta_d_ev) * P_ev_d,
        P_util == P_dem + P_bat + cp.multiply(P_ev, ev_plugged[:optimizer.K]) - P_sol[:optimizer.K],

        # EV SOC dynamics
        x_ev[0, 0] == optimizer.x0_ev,
        x_ev[0,K_leave] == 0.8,# move to soft constraints
        x_ev[0,K_arrive] == 0.2,
        # P_ev[K_leave:K_arrive] == 0, #caused a shit ton of problems 
        x_ev[:, 1:optimizer.K+1] == optimizer.ev_model.sys_d.A @ x_ev[:, :optimizer.K] +
                            optimizer.ev_model.sys_d.B @ P_ev.T,

        P_ev <= optimizer.ev_model.p_c_bar_ev,
        P_ev >= -optimizer.ev_model.p_d_bar_ev,

        x_ev[0, optimizer.K] == x_ev[0, 0],
        x_ev >= 0.1,
        x_ev <= 0.9,

        # Battery SOC dynamics
        x_b[0, 0] == optimizer.x0_b,
        x_b[:, 1:optimizer.K+1] == optimizer.battery_model.sys_d.A @ x_b[:, :optimizer.K] +
                           optimizer.battery_model.sys_d.B @ P_bat.T,
        P_bat <= optimizer.battery_model.p_c_bar_b,
        P_bat >= -optimizer.battery_model.p_d_bar_b,
        x_b[0, optimizer.K] == x_b[0, 0],
        x_b >= 0.1,
        x_b <= 0.9
    ]

    # Objective function 
    objective = cp.Minimize(
        cp.sum(                
                weight * cp.norm(optimizer.dt * P_util, 2) +                        # moderate penalty on total grid use
                # 5 * cp.norm(optimizer.dt * P_bat, 2) +                            # low penalty on battery use
                # 5 * cp.norm(optimizer.dt * P_ev, 2)+                              # low penalty on EV use

                50 * cp.norm(P_bat[:, 1:] - P_bat[:, :-1], 2) +                     # penalize sudden changes in Battery power
                50 * cp.norm(P_ev[:, 1:] - P_ev[:, :-1], 2) +                       # penalize sudden changes in EV power

                10 * optimizer.dt * cp.maximum(0, cp.multiply(c_elec, P_util)) +    # keep energy cost awareness

                10 *optimizer.dt * cp.maximum(0, x_b[:, :optimizer.K] - 0.8) +      # soft constraints on Battery SOC
                10 *optimizer.dt * cp.maximum(0, 0.2 - x_b[:, :optimizer.K]) +
                
                10 * optimizer.dt * cp.maximum(0, x_ev[:, :optimizer.K] - 0.8) +    # soft constraints on EV SOC
                10 * optimizer.dt * cp.maximum(0, 0.2 - x_ev[:, :optimizer.K])
        )
    )

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=True)

    return x_b.value,x_ev.value,P_bat.value, P_ev.value*ev_plugged[:optimizer.K], P_util.value, P_sol, P_dem

def evbm_optimization_v3(optimizer,weight,i):
# Define variables
    x_b = cp.Variable((1, optimizer.K+1))  # Battery SOC
    x_ev = cp.Variable((1, optimizer.K+1))  # EV SOC
    P_bat_c = cp.Variable((optimizer.K, 1))  # Battery charging
    P_bat_d = cp.Variable((optimizer.K, 1))  # Battery discharging
    P_ev_c = cp.Variable((optimizer.K, 1))   # EV charging
    P_ev_d = cp.Variable((optimizer.K, 1))   # EV discharging
    P_bat = cp.Variable((optimizer.K, 1))  
    P_ev = cp.Variable((optimizer.K, 1))  
    P_util = cp.Variable((optimizer.K, 1))   # Grid power

    # Known data
    time_range = np.arange(0, 24, optimizer.dt)

    solar_power = (optimizer.solar_model.dc_power_total.values)/1000
    P_sol = solar_power.reshape(-1, 1)

    c_elec = np.where((time_range >= 14) & (time_range <= 20), 0.2573, 0.0825).reshape(-1, 1)
    P_dem = optimizer.home_model.demand.to_numpy().reshape(-1, 1)

    #Determining when car leaves ad arrives
    t_leave = optimizer.ev_model.time_leave
    t_arrive = optimizer.ev_model.time_arrive
    K_leave_real = int(t_leave / optimizer.dt)
    K_arrive_real = int(t_arrive / optimizer.dt)
    ev_plugged_real = np.ones(2 * len(P_sol), dtype=P_sol.dtype)
    ev_plugged_real[K_leave_real:K_arrive_real] = 0
    ev_plugged_real[K_leave_real+int((24/optimizer.dt)):K_arrive_real+int((24/optimizer.dt))] = 0

    # Detect transitions in ev_plugged_real
    transitions_real = np.diff(ev_plugged_real)
    change_indices_real = np.where(transitions_real != 0)[0] + 1
    # Create ev_plugged slice (already in your code)
    ev_plugged = ev_plugged_real[i:int(i + (24 / optimizer.dt))].reshape(-1, 1)
    # Plotting
    plt.figure(figsize=(14, 5))
    # Plot ev_plugged_real full series
    plt.plot(ev_plugged_real, drawstyle='steps-post', label='ev_plugged_real', alpha=0.4)
    # Highlight transitions in ev_plugged_real
    for idx in change_indices_real:
        plt.axvline(x=idx, color='red', linestyle='--', alpha=0.4)
        plt.text(idx, 1.05, str(idx), rotation=90, verticalalignment='bottom', horizontalalignment='center', fontsize=7)
    # Plot ev_plugged window (offset to match its position in ev_plugged_real)
    offset = int(i)
    ev_plugged_flat = ev_plugged.flatten()
    x_vals = np.arange(offset, offset + len(ev_plugged_flat))
    plt.plot(x_vals, ev_plugged_flat, drawstyle='steps-post', color='blue', linewidth=2, label='ev_plugged window')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Index')
    plt.ylabel('Plugged Status')
    plt.title('EV Plugged Status: Full vs. Windowed')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ev_plugged_flat = ev_plugged.flatten()
    transitions = np.diff(ev_plugged_flat)
    leave_idxs = np.where(transitions == -1)[0] + 1
    arrive_idxs = np.where(transitions == 1)[0] + 1
    SOC_leave = 0.8
    SOC_arrive = 0.2

    X0_EV = optimizer.x0_ev   
    X0_B = optimizer.x0_b 
    # Handle leave
    if leave_idxs.size > 0:
        K_leave = int(leave_idxs[0])
    else:
        K_leave = 0
        X0_EV = SOC_leave
    # Handle arrive
    if arrive_idxs.size > 0:
        K_arrive = int(arrive_idxs[0])
    else:
        K_arrive = 0
        X0_EV = SOC_arrive

    # Constraints
    constraints = [
        # Grid power balance
        P_bat == optimizer.battery_model.eta_c_b * P_bat_c - (1 / optimizer.battery_model.eta_d_b) * P_bat_d,
        P_ev == optimizer.ev_model.eta_c_ev * P_ev_c - (1 / optimizer.ev_model.eta_d_ev) * P_ev_d,
        P_util == P_dem + P_bat + cp.multiply(P_ev, ev_plugged[:optimizer.K]) - P_sol[:optimizer.K],

        # EV SOC dynamics
        x_ev[:, 1:optimizer.K+1] == optimizer.ev_model.sys_d.A @ x_ev[:, :optimizer.K] +
                            optimizer.ev_model.sys_d.B @ P_ev.T,

        P_ev <= optimizer.ev_model.p_c_bar_ev,
        P_ev >= -optimizer.ev_model.p_d_bar_ev,

        x_ev[0, 0] == X0_EV,
        x_ev[0,K_leave] == SOC_leave,
        x_ev[0,K_arrive] == SOC_arrive,
        x_ev[0, optimizer.K] == x_ev[0, 0],
        x_ev >= 0.1,
        x_ev <= 0.9,

        # Battery SOC dynamics
        x_b[:, 1:optimizer.K+1] == optimizer.battery_model.sys_d.A @ x_b[:, :optimizer.K] +
                           optimizer.battery_model.sys_d.B @ P_bat.T,
        P_bat <= optimizer.battery_model.p_c_bar_b,
        P_bat >= -optimizer.battery_model.p_d_bar_b,

        x_b[0, 0] == X0_B,
        x_b[0, optimizer.K] == x_b[0, 0],
        x_b >= 0.1,
        x_b <= 0.9
    ]

    # Objective function 
    objective = cp.Minimize(
        cp.sum(                
                weight * cp.norm(optimizer.dt * P_util, 2) +                        # moderate penalty on total grid use
                # 5 * cp.norm(optimizer.dt * P_bat, 2) +                            # low penalty on battery use
                # 5 * cp.norm(optimizer.dt * P_ev, 2)+                              # low penalty on EV use

                50 * cp.norm(P_bat[:, 1:] - P_bat[:, :-1], 2) +                     # penalize sudden changes in Battery power
                50 * cp.norm(P_ev[:, 1:] - P_ev[:, :-1], 2) +                       # penalize sudden changes in EV power

                10 * optimizer.dt * cp.maximum(0, cp.multiply(c_elec, P_util)) +    # keep energy cost awareness

                10 *optimizer.dt * cp.maximum(0, x_b[:, :optimizer.K] - 0.8) +      # soft constraints on Battery SOC
                10 *optimizer.dt * cp.maximum(0, 0.2 - x_b[:, :optimizer.K]) +
                
                10 * optimizer.dt * cp.maximum(0, x_ev[:, :optimizer.K] - 0.8) +    # soft constraints on EV SOC
                10 * optimizer.dt * cp.maximum(0, 0.2 - x_ev[:, :optimizer.K])
        )
    )

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)

    return x_b.value,x_ev.value,P_bat.value, P_ev.value*ev_plugged[:optimizer.K], P_util.value, P_sol, P_dem

def mpc_v1(model_args,dt,Horizon):
    N_steps = int(Horizon/dt)
    for i in range(N_steps):
        optimizer = initialize_models(model_args,dt,i)
    
    return None

def plot_optimizer_results(x_b, x_ev, P_bat, P_ev, P_util, P_sol, P_dem, dt, day, weight,i,run_id=None):
    safe_day = day.replace("/", "-")
    if run_id is None:
        run_id = safe_day

    # Base folder name pattern
    base_dir = "plots"
    base_name = f"run_{run_id}"
    version = 1
    folder_name = f"{base_name}_v{version}"
    folder_path = os.path.join(base_dir, folder_name)

    # Scan existing folders to find max version
    existing_folders = [name for name in os.listdir(base_dir) if re.match(rf"{re.escape(base_name)}_v\d+$", name)]
    if existing_folders:
        # Extract version numbers
        versions = [int(re.search(r"_v(\d+)$", name).group(1)) for name in existing_folders]
        version = max(versions) + 1
        folder_name = f"{base_name}_v{version}"
        folder_path = os.path.join(base_dir, folder_name)

    os.makedirs(folder_path)

    # Convert arrays to 1D and align dimensions
    P_util = np.squeeze(P_util)
    x_b = np.squeeze(x_b)[:-1]
    x_ev = np.squeeze(x_ev)[:-1]
    P_bat = np.squeeze(P_bat)
    P_ev = np.squeeze(P_ev)
    P_sol = np.squeeze(P_sol)
    P_dem = np.squeeze(P_dem)
    
    time = np.arange(0, len(P_util) * dt, dt) 
    time = time + dt*i
    P_tot = -P_util + P_dem + P_bat - P_sol + P_ev  # Power conservation check

    # Energy metrics
    E_grid_to_home = np.sum(P_util[P_util > 0]) * dt
    E_home_demand = np.sum(P_dem) * dt
    E_solar_generated = np.sum(P_sol) * dt
    E_fed_to_grid = -np.sum(P_util[P_util < 0]) * dt
    # save_metrics(weight,E_grid_to_home,E_fed_to_grid)

    # Save energy summary to text file
    energy_summary = (
        f"=== Energy Flow Summary for {day} ===\n"
        f"1. Grid supplied to Home/Battery: {E_grid_to_home:.2f} kWh\n"
        f"2. Total Home Demand:             {E_home_demand:.2f} kWh\n"
        f"3. Solar Energy Produced:         {E_solar_generated:.2f} kWh\n"
        f"4. Energy Fed Back to Grid:       {E_fed_to_grid:.2f} kWh\n"
    )
    print(energy_summary)
    with open(os.path.join(folder_path, "energy_summary.txt"), "w") as f:
        f.write(energy_summary)

    # Plot SOC
    plt.figure(figsize=(10, 6))
    plt.plot(time, x_b, label="SOC Battery", color="r", linewidth=2)
    plt.plot(time, x_ev, label="SOC EV", color="grey", linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("State of Charge (%)")
    plt.title(f"SOC of Battery and EV on {day}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "soc_plot.png"))
    plt.close()

    # Plot Power Flows
    plt.figure(figsize=(10, 6))
    plt.plot(time, P_util, label="Utility Power (P_util)", color="b", linewidth=2)
    plt.plot(time, P_bat, label="Battery Power (P_bat)", color="g", linewidth=2)
    plt.plot(time, P_ev, label="EV Power (P_ev)", color="black", linewidth=2)
    plt.plot(time, P_sol, label="Solar Power (P_sol)", color="orange", linewidth=2)
    plt.plot(time, P_dem, label="Demand", color="purple", linewidth=2)
    plt.plot(time, P_tot, label="Power Conservation (P_tot)", color="red", linestyle='--', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (kW)")
    plt.title(f"Power Flows and Conservation Check on {day}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "power_flow_plot.png"))
    plt.close()

    ## Verify Power conservation
    # fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # # Battery SOC
    # axs1[0].plot(time, x_b, label="State of Charge (SOC) Battery", color="r", linewidth=2)
    # axs1[0].set_ylabel("SOC (%)")
    # axs1[0].set_title("Battery: State of Charge (SOC) and Power")
    # axs1[0].grid(True)
    # axs1[0].legend(loc="best")
    # # Battery Power
    # axs1[1].plot(time, P_bat, label="Battery Power (P_bat)", color="g", linewidth=2)
    # axs1[1].set_xlabel("Time (hours)")
    # axs1[1].set_ylabel("Power (kW)")
    # axs1[1].grid(True)
    # axs1[1].legend(loc="best")
    # plt.tight_layout()

    # fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # # EV SOC
    # axs2[0].plot(time, x_ev, label="State of Charge (SOC) EV", color="grey", linewidth=2)
    # axs2[0].set_ylabel("SOC (%)")
    # axs2[0].set_title("EV: State of Charge (SOC) and Power")
    # axs2[0].grid(True)
    # axs2[0].legend(loc="best")
    # # EV Power
    # axs2[1].plot(time, P_ev, label="EV Power (P_ev)", color="black", linewidth=2)
    # axs2[1].set_xlabel("Time (hours)")
    # axs2[1].set_ylabel("Power (kW)")
    # axs2[1].grid(True)
    # axs2[1].legend(loc="best")
    # plt.tight_layout()

def plot_obj_functions(x_b,x_ev, P_bat,P_ev,P_util, P_sol, P_dem, dt):
    # Convert arrays to 1D
    P_util = np.squeeze(P_util)
    x_b = np.squeeze(x_b)[:-1]  # Trim last value of x_b to match other arrays
    x_ev = np.squeeze(x_ev)[:-1]
    P_bat = np.squeeze(P_bat)
    P_ev = np.squeeze(P_ev)
    P_sol = np.squeeze(P_sol)
    P_dem = np.squeeze(P_dem)

    # Create time vector
    time = np.arange(0, len(P_util) * dt, dt)
    # Create cost vector
    c_elec = np.where((time >= 14) & (time <= 20), 0.2573, 0.0825)

    # Compute each term
    obj_grid = 1000 * (dt * P_util) ** 2
    obj_bat = 0.01 * (dt * P_bat) ** 2
    obj_ev = 0.01 * (dt * P_ev) ** 2
    obj_cost = 10 * dt * np.maximum(0, c_elec * P_util)

    obj_soc_b_high = 10 * dt * np.maximum(0, x_b - 0.8)
    obj_soc_b_low = 10 * dt * np.maximum(0, 0.2 - x_b)
    obj_soc_ev_high = 10 * dt * np.maximum(0, x_ev - 0.8)
    obj_soc_ev_low = 10 * dt * np.maximum(0, 0.2 - x_ev)

    # Plotting
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 2, 1)
    plt.plot(time, obj_grid, label="Grid Use Penalty")
    plt.ylabel("Penalty")
    plt.title("Grid Use")
    plt.grid()

    plt.subplot(4, 2, 2)
    plt.plot(time, obj_cost, label="Electricity Cost Penalty", color='orange')
    plt.title("Electricity Cost")
    plt.grid()

    plt.subplot(4, 2, 3)
    plt.plot(time, obj_bat, label="Battery Use Penalty", color='green')
    plt.title("Battery Use")
    plt.grid()

    plt.subplot(4, 2, 4)
    plt.plot(time, obj_ev, label="EV Use Penalty", color='purple')
    plt.title("EV Use")
    plt.grid()

    plt.subplot(4, 2, 5)
    plt.plot(time, obj_soc_b_high, label="Battery SOC > 0.8", color='red')
    plt.plot(time, obj_soc_b_low, label="Battery SOC < 0.2", color='blue')
    plt.title("Battery SOC Penalties")
    plt.legend()
    plt.grid()

    plt.subplot(4, 2, 6)
    plt.plot(time, obj_soc_ev_high, label="EV SOC > 0.8", color='red')
    plt.plot(time, obj_soc_ev_low, label="EV SOC < 0.2", color='blue')
    plt.title("EV SOC Penalties")
    plt.legend()
    plt.grid()

    # Optional: total objective value per time step
    total_obj = (obj_grid + obj_bat + obj_ev + obj_cost +
                 obj_soc_b_high + obj_soc_b_low +
                 obj_soc_ev_high + obj_soc_ev_low)

    plt.subplot(4, 1, 4)
    plt.plot(time, total_obj, label="Total Objective", color='black')
    plt.title("Total Objective Value")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_inputs(P_sol, P_dem, dt,day):
    # Convert arrays to 1D
    P_sol = np.squeeze(P_sol)
    P_dem = np.squeeze(P_dem)

    time = np.arange(0, len(P_sol) * dt, dt)
    # Plot Power Data (Utility, Battery, Solar, Demand, and Conservation)
    plt.figure(figsize=(10, 6))
    plt.plot(time, P_sol, label="Solar Power (P_sol)", color="orange", linestyle='-', linewidth=2)
    plt.plot(time, P_dem, label="Demand", color="purple", linestyle='-', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (kW)")
    plt.title(f"Inputs on {day}")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.tight_layout()
    plt.show()

def save_metrics(weight, E_grid_home, E_back_feed):
    filename = os.path.join('data', 'metrics.csv')
    
    new_data = pd.DataFrame([{
        'Weight': weight,
        'E_grid_home': E_grid_home,
        'E_back_feed': E_back_feed
    }])

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(filename, index=False)