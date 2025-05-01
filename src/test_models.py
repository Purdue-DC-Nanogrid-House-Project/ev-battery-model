import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cvxpy as cp
import numpy as np

def test_ev_charging(ev_model, battery_model, initial_charge, target_charge):
    # Convert percentages to state of charge (SoC) in kWh for ev
    initial_soc_ev = initial_charge * ev_model.x_bar_ev
    target_soc_ev = target_charge * ev_model.x_bar_ev

    # Convert percentage to state of charge (SoC) in Kwh for charger
    initial_soc_b = 1.00 * battery_model.x_bar_b #assumes full charge

    time_range = np.arange(0, 24, ev_model.dt)  # Time range for 24 hours

    # Create DataFrame for EV
    df_ev = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'Charger_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for charger power values
    })

    # Create DataFrame for Battery
    df_b = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'EV_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    #Assign Initial states
    df_ev.iloc[0,1] = initial_soc_ev
    df_b.iloc[0,1] = initial_soc_b

    print(f"Initial ev SOC: {df_ev.iloc[0,1] } kWh")
    print(f"Initial b SOC: {df_b.iloc[0,1] } kWh")

    df_ev.iloc[0,2] = 0
    df_b.iloc[0,2] = 0


    # print(ev_model.sys_d.A,ev_model.sys_d.B)
    # print(battery_model.sys_d.A,battery_model.sys_d.B)
    

    for i in range(len(df_ev)-1):
            # Check if the current time is greater than 2 hours
            if df_ev.iloc[i,0] > 2:
                # Check if the current SoC is less than the target SoC
                if df_ev.iloc[i,1] < target_soc_ev:
                    #Determining EV charge rate & Updating EV SOC
                    U = ev_model.p_c_bar_ev
                    df_ev.iloc[i+1,1] = df_ev.iloc[i,1]*ev_model.sys_d.A + (ev_model.sys_d.B)*U

                    #Solve for Battery discharging rate & Updating Battery SOC
                    P_discharge = U / battery_model.eta_d_b
                    df_b.iloc[i+1,1] = df_b.iloc[i,1]*battery_model.sys_d.A - (battery_model.sys_d.B)*P_discharge

                    #Setting Power
                    df_ev.iloc[i+1,2] = ev_model.p_c_bar_ev * ev_model.eta_c_ev
                    df_b.iloc[i+1,2] = -P_discharge* battery_model.eta_d_b
                
                else:
                    df_ev.iloc[i+1, 1] =  df_ev.iloc[i, 1] #Battery Reached desired state
                    df_b.iloc[i+1, 1] =  df_b.iloc[i, 1]

                    df_ev.iloc[i+1, 2] = 0  # No charging power
                    df_b.iloc[i+1, 2] = 0  # No battery power delivered
            else:
                df_ev.iloc[i+1, 1] =  df_ev.iloc[i, 1] #Battery Reached desired state
                df_b.iloc[i+1, 1] =  df_b.iloc[i, 1]


                df_ev.iloc[i+1, 2] = 0  # No charging power
                df_b.iloc[i+1, 2] = 0  # No battery power delivered
                 

    # Plotting SoC for EV and Battery
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 1], label='EV SoC (kWh)', color='blue')  # Time vs EV SoC
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 1], label='Battery SoC (kWh)', color='orange')  # Time vs Battery SoC
    plt.title('State of Charge (SoC) Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('SoC (kWh)')
    plt.ylim(0,np.max([np.max(df_ev.iloc[:, 1]), np.max(df_b.iloc[:, 1])])+2)
    plt.legend()
    plt.grid()

    # Plotting Charger Power and Battery Power on the same graph
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 2], label='EV Power (kW)', color='green')  # Time vs Charger Power
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 2], label='Charger Power (kW)', color='purple')  # Time vs Battery Power
    plt.title('Power Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.ylim(np.min([np.min(df_ev.iloc[:, 2]), np.min(df_b.iloc[:, 2])])-2,np.max([np.max(df_ev.iloc[:, 2]), np.max(df_b.iloc[:, 2])])+2)
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def test_ev_charging_v2(ev_model,battery_model,home_model,utility_model,initial_charge,target_charge,ev_call):
    # Convert percentages to state of charge (SoC) in kWh for ev
    initial_soc_ev = initial_charge * ev_model.x_bar_ev
    target_soc_ev = target_charge * ev_model.x_bar_ev

    # Convert percentage to state of charge (SoC) in Kwh for charger
    initial_soc_b = 1.00 * battery_model.x_bar_b #assumes full charge

    time_range = np.arange(0, 24, ev_model.dt)  # Time range for 24 hours

    # Create DataFrame for EV
    df_ev = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'EV_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for charger power values
    })

    # Create DataFrame for Battery
    df_b = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'Charger_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    # Create DataFrame for Utility
    df_u = pd.DataFrame({
        'Time (hours)': time_range,
        'Utility Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    # Create DataFrame for Utility
    df_h = pd.DataFrame({
        'Time (hours)': time_range,
        'Utility Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })


    #Set Initial states SOC's
    df_ev.iloc[0,1] = initial_soc_ev
    df_b.iloc[0,1] = initial_soc_b

    print(f"Initial ev SOC: {df_ev.iloc[0,1] } kWh")
    print(f"Initial b SOC: {df_b.iloc[0,1] } kWh")

    #Set Initial Powers
    df_ev.iloc[0,2] = 0
    df_b.iloc[0,2] = 0
    df_u.iloc[0,1] = 0

    #Home demand is constant
    U_HOME = home_model.demand
    df_h.iloc[:,1] = U_HOME

    for i in range(len(df_ev)-1):
        #Charging only if the current_soc < target_soc and plugged in at 2 AM
        if (df_ev.iloc[i,0] >2) and df_ev.iloc[i,1] < target_soc_ev:
            U_EV = ev_call
        else:
            U_EV = 0

        U_TOTAL = U_HOME + U_EV #Will also incporate the battery when its pulling energy (U_CHARGER)

        #Discharging charger battery only if SOC_b>0
        if df_b.iloc[i,1] > 0:
            #Output maximum if U_TOTAL is greater than max discharge of battery
            if U_TOTAL >= battery_model.p_d_bar_b:
                P_discharge = battery_model.p_d_bar_b/battery_model.eta_d_b
            else:
                P_discharge = U_TOTAL/battery_model.eta_d_b

        else:
            P_discharge = 0
        
        utility_model.utility = U_TOTAL - P_discharge

        #Updating Battery SOC's
        df_ev.iloc[i+1,1] = df_ev.iloc[i,1]*ev_model.sys_d.A + (ev_model.sys_d.B)*U_EV
        #In the future it would charge
        df_b.iloc[i+1,1] = df_b.iloc[i,1]*battery_model.sys_d.A - (battery_model.sys_d.B)*P_discharge

        #Updating Powers
        df_ev.iloc[i+1,2] = U_EV
        df_b.iloc[i+1,2] = -P_discharge
        df_u.iloc[i+1,1] = -utility_model.utility

    # Plotting SoC for EV and Battery
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 1], label='EV SoC (kWh)', color='blue')  # Time vs EV SoC
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 1], label='Battery SoC (kWh)', color='orange')  # Time vs Battery SoC
    plt.title('State of Charge (SoC) Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('SoC (kWh)')
    plt.ylim(np.min([np.min(df_ev.iloc[:, 1]), np.min(df_b.iloc[:, 1])])-2,np.max([np.max(df_ev.iloc[:, 1]), np.max(df_b.iloc[:, 1])])+2)
    plt.legend()
    plt.grid()

    # Plotting Charger Power and Battery Power on the same graph
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 2], label='EV Power (kW)', color='green')  # Time vs Charger Power (kW)
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 2], label='Charger Power (kW)', color='purple')  # Time vs Battery Power (kW)
    plt.plot(df_u.iloc[:,0],df_u.iloc[:,1],label='Utility Power (kW)', color='black') # Time vs Utility Power (kW)
    plt.plot(df_h.iloc[:,0],df_h.iloc[:,1],label='Home Power (kW)', color='red') # Time vs Home Power (kW)
    plt.title('Power Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.ylim(np.min([np.min(df_ev.iloc[:, 2]), np.min(df_b.iloc[:, 2]),np.min(df_u.iloc[:, 1]),np.min(df_h.iloc[:, 1])])-2,np.max([np.max(df_ev.iloc[:, 2]), np.max(df_b.iloc[:, 2]),np.max(df_u.iloc[:, 1]),np.max(df_h.iloc[:, 1])])+2)
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def test_solar_model(solar_model,dt):
    time_range = np.arange(0, 24, dt)  # Time range for 24 hours
    solar_power = (solar_model.dc_power_total[0:-1].values)/1000
    solar_power_interp = np.interp(time_range, np.linspace(0, 23, len(solar_power)), solar_power)
    print(solar_power_interp.shape)
    df_s = pd.DataFrame({
        'Time (hours)': time_range,
        'Solar Power (kW)': solar_power_interp  # Placeholder for battery power values
    })
    plt.plot(df_s['Time (hours)'], df_s['Solar Power (kW)'])
    plt.xlabel('Time')
    plt.ylabel('Total DC Power (kW)')
    plt.title(f'Total Solar PV Output on {solar_model.start_time} ')
    plt.show()

def test_ev_charging_v3(ev_model,battery_model,home_model,utility_model,solar_model,initial_charge_pre, initial_charge_post,target_charge):
    # Convert percentages to state of charge (SoC) in kWh for ev
    initial_soc_ev_pre = initial_charge_pre * ev_model.x_bar_ev
    initial_soc_ev_post = initial_charge_post * ev_model.x_bar_ev
    target_soc_ev = target_charge * ev_model.x_bar_ev

    # Convert percentage to state of charge (SoC) in Kwh for charger
    initial_soc_b = 1.00 * battery_model.x_bar_b #assumes full charge
    target_soc_b = target_charge * battery_model.x_bar_b

    time_range = np.arange(0, 24, ev_model.dt)  # Time range for 24 hours

    # Create DataFrame for EV
    df_ev = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'EV_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for charger power values
    })

    # Create DataFrame for Battery
    df_b = pd.DataFrame({
        'Time (hours)': time_range,
        'SoC (kWh)': np.zeros(len(time_range)),  # Placeholder for SoC values
        'Charger_Battery Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    # Create DataFrame for Utility
    df_u = pd.DataFrame({
        'Time (hours)': time_range,
        'Utility Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    # Create DataFrame for Utility
    df_h = pd.DataFrame({
        'Time (hours)': time_range,
        'Utility Power (kW)': np.zeros(len(time_range))  # Placeholder for battery power values
    })

    # Create DataFrame for Solar
    solar_power = (solar_model.dc_power_total[0:-1].values)/1000
    solar_power_interp = np.interp(time_range, np.linspace(0, 23, len(solar_power)), solar_power)
    df_s = pd.DataFrame({
        'Time (hours)': time_range,
        'Solar Power (kW)': solar_power_interp  # Placeholder for battery power values
    })

    #Set Initial states SOC's
    df_ev.iloc[0,1] = initial_soc_ev_pre
    df_b.iloc[0,1] = initial_soc_b

    print(f"Initial ev SOC: {df_ev.iloc[0,1] } kWh")
    print(f"Initial b SOC: {df_b.iloc[0,1] } kWh")

    #Set Initial Powers
    df_ev.iloc[0,2] = 0
    df_b.iloc[0,2] = 0
    df_u.iloc[0,1] = 0

    #Home demand is constant
    U_HOME = home_model.demand
    df_h.iloc[:,1] = U_HOME
    U_CHARGER = 0

    for i in range(len(df_ev)-1):

        ev_model.plugged = True
        battery_charging = False
        # Charging home battery if car is unplugged (between 9:00 AM and 4:00 PM)
        if 9 <= df_ev.iloc[i,0] < 16 :
            ev_model.plugged = False
            battery_charging = True
            if df_b.iloc[i,1] <= target_soc_b:
                if df_s.iloc[i,1] >= battery_model.p_c_bar_b:
                    U_CHARGER = battery_model.p_c_bar_b
                else:
                    U_CHARGER = df_s.iloc[i,1]
            else:
                U_CHARGER = 0
        
        if df_ev.iloc[i,0] == 16:
            df_ev.iloc[i,1] = initial_soc_ev_post
        
        # Charging car battery if car battery is plugged in
        if ev_model.plugged == False or df_ev.iloc[i,1]>= target_soc_ev:
            U_EV = 0
        else:
            U_EV = ev_model.p_c_bar_ev/ev_model.eta_c_ev

        U_TOTAL = U_HOME + U_EV + U_CHARGER

        #Discharging charger battery only if SOC_b>0
        if df_b.iloc[i,1] > 0 and battery_charging == False:
            #Output maximum if U_TOTAL is greater than max discharge of battery
            if U_TOTAL >= battery_model.p_d_bar_b:
                P_discharge = battery_model.p_d_bar_b/battery_model.eta_d_b
            else:
                P_discharge = U_TOTAL/battery_model.eta_d_b

        else:
            P_discharge = 0
        

        # Determining remainder solar power if any after charging battery
        P_solar =  df_s.iloc[i,1]

        # Determining total utility needed for the house (minimize this)
        utility_model.utility = U_TOTAL - P_discharge - P_solar

        #Updating Battery Dataframe
        df_ev.iloc[i+1,1] = df_ev.iloc[i,1]*ev_model.sys_d.A + (ev_model.sys_d.B)*U_EV
        df_ev.iloc[i+1,2] = U_EV

        #Updating Charger SOC
        if battery_charging == True:
            # print(f"{df_ev.iloc[i,0]} : charging") #Debug to see when its charging
            df_b.iloc[i+1,1] = df_b.iloc[i,1]*battery_model.sys_d.A + (battery_model.sys_d.B)*U_CHARGER
            df_b.iloc[i+1,2] = U_CHARGER
        else:
            # print(f"{df_ev.iloc[i,0]} : discharging") #Debug to see when its discharging
            df_b.iloc[i+1,1] = df_b.iloc[i,1]*battery_model.sys_d.A - (battery_model.sys_d.B)*P_discharge
            df_b.iloc[i+1,2] = -P_discharge

        # Updating Utility
        df_u.iloc[i+1,1] = -utility_model.utility

    # Plotting SoC for EV and Battery
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 1], label='EV SoC (kWh)', color='blue')  # Time vs EV SoC
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 1], label='Battery SoC (kWh)', color='orange')  # Time vs Battery SoC
    plt.title(f'State of Charge (SoC) Over Time on {solar_model.start_time}')
    plt.xlabel('Time (hours)')
    plt.ylabel('SoC (kWh)')
    plt.ylim(np.min([np.min(df_ev.iloc[:, 1]), np.min(df_b.iloc[:, 1])])-2,np.max([np.max(df_ev.iloc[:, 1]), np.max(df_b.iloc[:, 1])])+2)
    plt.legend()
    plt.grid()


    # Plotting Charger Power and Battery Power on the same graph
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 1st subplot
    plt.plot(df_ev.iloc[:, 0], df_ev.iloc[:, 2], label='EV Power (kW)', color='green')  # Time vs Charger Power (kW)
    plt.plot(df_b.iloc[:, 0], df_b.iloc[:, 2], label='Charger Power (kW)', color='purple')  # Time vs Battery Power (kW)
    plt.plot(df_u.iloc[:,0],df_u.iloc[:,1],label='Utility Power (kW)', color='black') # Time vs Utility Power (kW)
    plt.plot(df_h.iloc[:,0],df_h.iloc[:,1],label='Home Power (kW)', color='red') # Time vs Home Power (kW)
    plt.plot(df_s.iloc[:,0],df_s.iloc[:,1],label='Solar Power (kW)', color='orange') # Time vs Home Power (kW)
    plt.title(f'Power Over Time on {solar_model.start_time}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.ylim(np.min([np.min(df_ev.iloc[:, 2]), np.min(df_b.iloc[:, 2]),np.min(df_u.iloc[:, 1]),np.min(df_h.iloc[:, 1])])-2,np.max([np.max(df_ev.iloc[:, 2]), np.max(df_b.iloc[:, 2]),np.max(df_u.iloc[:, 1]),np.max(df_h.iloc[:, 1])])+2)
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Adjust layout to prevent overlap
    
    plt.show()

def evbm_optimization_v1(optimizer):
    # Define variables
    x_b = cp.Variable((1, optimizer.K+1))  # Battery state of charge
    x_ev = cp.Variable((1, optimizer.K+1))
    P_bat = cp.Variable((optimizer.K, 1))  # Battery power
    P_ev = cp.Variable((optimizer.K, 1))
    P_util = cp.Variable((optimizer.K, 1))  # Utility power

    # Define knowns
    time_range = np.arange(0, 24, optimizer.dt)  # Time range for 24 hours
    solar_power = (optimizer.solar_model.dc_power_total[0:-1].values)/1000
    P_sol = np.interp(time_range, np.linspace(0, 23, len(solar_power)), solar_power).reshape(-1, 1)
    c_elec = np.where((time_range >= 14) & (time_range <= 20), 0.2573, 0.0825).reshape(-1, 1)
    P_dem = optimizer.home_model.demand.to_numpy().reshape(-1, 1)

    # Check the shapes after squeezing
    # print("x_b shape:", x_b.shape)
    # print("P_util shape:", P_util.shape)
    # print("P_bat shape:", P_bat.shape)
    # print("P_sol shape:", P_sol.shape)
    # print("P_dem shape:",P_dem.shape)

    # Constraints
    constraints = [
        P_bat == cp.maximum(P_bat/optimizer.battery_model.eta_c_b,P_bat*optimizer.battery_model.eta_c_b),
        P_ev == cp.maximum(P_bat/optimizer.ev_model.eta_c_ev,P_bat*optimizer.ev_model.eta_c_ev),

        P_util == P_dem + P_bat + P_ev - P_sol[:optimizer.K],
        
        #EV Constrains
        x_ev[0, 0] == optimizer.x0_ev,
        #add EV leaving and arrival constraint here
        #calculate what SOC the battery comes back at and add constrant for this as well and what the battery leaves at
        #Piece Wise functioN
        #P_bat_eff = np.max()
        x_ev[:, 1:optimizer.K+1] == cp.multiply(x_ev[:, :optimizer.K],optimizer.ev_model.sys_d.A) + cp.multiply(P_ev.T,optimizer.ev_model.sys_d.B),
            #Power Constraints
        P_ev <= cp.multiply(optimizer.ev_model.p_c_bar_ev,optimizer.ev_model.eta_c_ev),
        P_ev >= cp.multiply(-optimizer.ev_model.p_d_bar_ev,1/optimizer.ev_model.eta_d_ev),

        x_ev[0, optimizer.K] == x_ev[0, 0],
            #Physical Limits
        x_ev >= 0.1,
        x_ev <= 0.9,

        #Charger Constraints
        x_b[0, 0] == optimizer.x0_b,
        x_b[:, 1:optimizer.K+1] == cp.multiply(x_b[:, :optimizer.K],optimizer.battery_model.sys_d.A) + cp.multiply(P_bat.T,optimizer.battery_model.sys_d.B),
            #Power Constraints
        P_bat <= cp.multiply(optimizer.battery_model.p_c_bar_b,optimizer.battery_model.eta_c_b),
        P_bat >= cp.multiply(-optimizer.battery_model.p_d_bar_b,1/optimizer.battery_model.eta_d_b),
        x_b[0, optimizer.K] == x_b[0, 0],
            #Physical Limits
        x_b >= 0.1,
        x_b <= 0.9


    ]

    # Objective function
    objective = cp.Minimize(
        cp.sum(
            # Uses L2 euclidean smoothing to ensure the are no sudden variations in Utility pull
            10 * cp.norm(optimizer.dt * (0-P_util), 2) +   # (1) Minimizing utility power usage variations

            # Uses L2 euclidean smoothing to ensure no sudden variations in battery/ev pull
            cp.norm(optimizer.dt * (0-P_bat), 2)+
            cp.norm(optimizer.dt * (0-P_ev), 2)+

            # Ensures to minimize negative feed back
            1000*cp.maximum(0,-P_util)+

            # Ensures to mimize positive feed back
            1000*cp.maximum(0,P_util)+

            # Minimizing electricity cost by minimizing P_util
            optimizer.dt * cp.maximum(0, cp.multiply(c_elec, P_util)) +  # (2) Cost minimization    

            #Soft Preferences
            optimizer.dt * cp.maximum(0, x_b[:, :optimizer.K] - 0.8) + # (3) Penalizing exceeding max SOC (80%
            optimizer.dt * cp.maximum(0, 0.2 - x_b[:, :optimizer.K])   # (4) Penalizing dropping below min SOC (20%)
        )
    )

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose =False)

    return x_b.value,x_ev.value,P_bat.value,P_ev.value,P_util.value, P_sol,P_dem

def evbm_optimization_v2(optimizer):
    # Define variables
    x_b = cp.Variable((1, optimizer.K+1))  # Battery SOC
    x_ev = cp.Variable((1, optimizer.K+1))  # EV SOC
    P_bat_c = cp.Variable((optimizer.K, 1))  # Battery charging
    P_bat_d = cp.Variable((optimizer.K, 1))  # Battery discharging
    P_ev_c = cp.Variable((optimizer.K, 1))   # EV charging
    P_ev_d = cp.Variable((optimizer.K, 1))   # EV discharging
    P_util = cp.Variable((optimizer.K, 1))   # Grid power
    t_leave = optimizer.ev_model.time_leave
    t_arrive = optimizer.ev_model.time_arrive
    K_leave = int(t_leave / optimizer.dt)
    K_arrive = int(t_arrive / optimizer.dt)
    print("Index Leave",{K_leave})
    print("Index arrive",{K_arrive})

    # Known data
    time_range = np.arange(0, 24, optimizer.dt)
    solar_power = (optimizer.solar_model.dc_power_total[0:-1].values)/1000
    
    P_sol = np.interp(time_range, np.linspace(0, 23, len(solar_power)), solar_power).reshape(-1, 1)
    c_elec = np.where((time_range >= 14) & (time_range <= 20), 0.2573, 0.0825).reshape(-1, 1)
    P_dem = optimizer.home_model.demand.to_numpy().reshape(-1, 1)
    ev_plugged = np.ones_like(P_sol)
    ev_plugged[K_leave:K_arrive] = 0

    # Total power = charging - discharging
    P_bat = optimizer.battery_model.eta_c_b * P_bat_c - (1 / optimizer.battery_model.eta_d_b) * P_bat_d
    P_ev = optimizer.ev_model.eta_c_ev * P_ev_c - (1 / optimizer.ev_model.eta_d_ev) * P_ev_d

    # Constraints
    constraints = [
        # Grid power balance

        P_util == P_dem + P_bat + cp.multiply(P_ev, ev_plugged[:optimizer.K]) - P_sol[:optimizer.K],

        # EV SOC dynamics
        x_ev[0, 0] == optimizer.x0_ev,
        x_ev[0,K_leave] == 0.8,
        x_ev[0,K_arrive] == 0.2,
        P_ev[K_leave:K_arrive] == 0,
        # x_ev[0,K_arrive] == x_ev[0,K_leave-1],
        x_ev[:, 1:optimizer.K+1] == cp.multiply(x_ev[:, :optimizer.K], optimizer.ev_model.sys_d.A) +
                                   cp.multiply(optimizer.ev_model.sys_d.B,
                                               optimizer.ev_model.eta_c_ev * P_ev_c.T -
                                               (1/optimizer.ev_model.eta_d_ev) * P_ev_d.T),

        P_ev_c >= 0,
        P_ev_c <= optimizer.ev_model.p_c_bar_ev,
        P_ev_d >= 0,
        P_ev_d <= optimizer.ev_model.p_d_bar_ev,

        x_ev[0, optimizer.K] == x_ev[0, 0],
        x_ev >= 0.1,
        x_ev <= 0.9,

        # Battery SOC dynamics
        x_b[0, 0] == optimizer.x0_b,
        x_b[:, 1:optimizer.K+1] == cp.multiply(x_b[:, :optimizer.K], optimizer.battery_model.sys_d.A) +
                                  cp.multiply(optimizer.battery_model.sys_d.B,
                                              optimizer.battery_model.eta_c_b * P_bat_c.T -
                                              (1/optimizer.battery_model.eta_d_b) * P_bat_d.T),
        P_bat_c >= 0,
        P_bat_c <= optimizer.battery_model.p_c_bar_b,
        P_bat_d >= 0,
        P_bat_d <= optimizer.battery_model.p_d_bar_b,
        x_b[0, optimizer.K] == x_b[0, 0],
        x_b >= 0.1,
        x_b <= 0.9
    ]

    # Objective function 
    objective = cp.Minimize(
        cp.sum(
            2000 * cp.norm(optimizer.dt * (0 - P_util), 2) +
            100*cp.norm(optimizer.dt * (0 - P_bat), 2) +
            100*cp.norm(optimizer.dt * (0 - P_ev), 2) +
            1000 * cp.maximum(0, -P_util) +
            4000 * cp.maximum(0, P_util) +
            optimizer.dt * cp.maximum(0, cp.multiply(c_elec, P_util)) +
            optimizer.dt * cp.maximum(0, x_b[:, :optimizer.K] - 0.8) +
            optimizer.dt * cp.maximum(0, 0.2 - x_b[:, :optimizer.K])
        )
    )

    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    return x_b.value, x_ev.value, P_bat.value, P_ev.value, P_util.value, P_sol, P_dem

def plot_results(x_b,x_ev, P_bat,P_ev,P_util, P_sol, P_dem, dt):
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
    # Compute Power Conservation Variable
    P_tot = -P_util + P_dem + P_bat - P_sol + P_ev  # Power balance check

    # Check shapes
    print("Time shape:", time.shape)
    print("x_b shape:", x_b.shape)
    print("P_util shape:", P_util.shape)
    print("P_bat shape:", P_bat.shape)
    print("P_ev shape:", P_ev.shape)
    print("P_sol shape:", P_sol.shape)
    print("P_dem shape:", P_dem.shape)
    print("P_tot shape:", P_tot.shape)  # Ensure shape consistency

    # Compute cumulative energy (kWh)
    E_util = np.cumsum(P_util) * dt
    E_bat = np.cumsum(P_bat) * dt
    E_ev = np.cumsum(P_ev) * dt
    E_sol = np.cumsum(P_sol) * dt
    E_dem = np.cumsum(P_dem) * dt

    # Plot Battery State of Charge (SOC)
    plt.figure(figsize=(10, 6))
    plt.plot(time, x_b, label="State of Charge (SOC Battery", color="r", linestyle='-', linewidth=2)
    plt.plot(time, x_ev, label="State of Charge (SOC) EV", color="grey", linestyle='-', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("State of Charge (%)")
    plt.title("Battery and EV State of Charge (SOC) over Time")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    # Plot Power Data (Utility, Battery, Solar, Demand, and Conservation)
    plt.figure(figsize=(10, 6))
    plt.plot(time, P_util, label="Utility Power (P_util)", color="b", linestyle='-', linewidth=2)
    plt.plot(time, P_bat, label="Battery Power (P_bat)", color="g", linestyle='-', linewidth=2)
    plt.plot(time, P_ev, label="EV Power (P_ev)", color="grey", linestyle='-', linewidth=2)
    plt.plot(time, P_sol, label="Solar Power (P_sol)", color="orange", linestyle='-', linewidth=2)
    plt.plot(time, P_dem, label="Demand", color="purple", linestyle='-', linewidth=2)
    plt.plot(time, P_tot, label="Power Conservation (P_tot)", color="red", linestyle='-', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (kW)")
    plt.title("Utility Power, Battery Power,EV_Power,Solar Power, Demand, and Power Conservation")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    E_grid_to_home = np.sum(P_util[P_util > 0]) * dt
    E_home_demand = np.sum(P_dem) * dt
    E_solar_generated = np.sum(P_sol) * dt
    E_fed_to_grid = -np.sum(P_util[P_util < 0]) * dt  # negative values, so negate

    print("\n=== Energy Flow Summary (kWh) ===")
    print(f"1. Grid supplied to Home/Battery: {E_grid_to_home:.2f} kWh")
    print(f"2. Total Home Demand:             {E_home_demand:.2f} kWh")
    print(f"3. Solar Energy Produced:         {E_solar_generated:.2f} kWh")
    print(f"4. Energy Fed Back to Grid:       {E_fed_to_grid:.2f} kWh")

    #  # Plot Cumulative Energy
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, E_util, label="Cumulative Utility Energy", color="b", linestyle='-')
    # plt.plot(time, E_bat, label="Cumulative Battery Energy", color="g", linestyle='-')
    # plt.plot(time, E_ev, label="Cumulative EV Energy", color="grey", linestyle='-')
    # plt.plot(time, E_sol, label="Cumulative Solar Energy", color="orange", linestyle='-')
    # plt.plot(time, E_dem, label="Cumulative Demand Energy", color="purple", linestyle='-')
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Cumulative Energy (kWh)")
    # plt.title("Cumulative Energy for Each Power Flow")
    # plt.grid(True)
    # plt.legend(loc="best")
    # plt.tight_layout()

    # Show plot
    plt.show()