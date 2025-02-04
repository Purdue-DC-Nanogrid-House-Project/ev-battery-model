import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def test_ev_charging_v2(ev_model,battery_model,home_model,utility_model,initial_charge,target_charge):
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

    df_u = pd.DataFrame({
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

    for i in range(len(df_ev)-1):
        #Charging only if the current_soc < target_soc and plugged in at 2 AM
        if (df_ev.iloc[i,0] >2) and df_ev.iloc[i,1] < target_soc_ev:
            U_EV = ev_model.p_c_bar_ev
        else:
            U_EV = 0

        U_TOTAL = U_HOME + U_EV

        if df_b.iloc[i,1] > 0:
            if U_TOTAL >= battery_model.p_d_bar_b:
                P_discharge = battery_model.p_d_bar_b/battery_model.eta_d_b
            else:
                P_discharge = U_TOTAL/battery_model.eta_d_b

        else:
            P_discharge = 0
        
        utility_model = U_TOTAL - P_discharge

        #Updating Battery SOC's
        df_ev.iloc[i+1,1] = df_ev.iloc[i,1]*ev_model.sys_d.A + (ev_model.sys_d.B)*U_EV
        df_b.iloc[i+1,1] = df_b.iloc[i,1]*battery_model.sys_d.A - (battery_model.sys_d.B)*P_discharge

        #Updating Powers
        df_ev.iloc[i+1,2] = U_EV
        df_b.iloc[i+1,2] = -P_discharge
        df_u.iloc[i+1,1] = -utility_model.utility




        
        

  

if __name__ == "__main__":
    # This block is not needed since we will call the function from main.py
    pass