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

if __name__ == "__main__":
    # This block is not needed since we will call the function from main.py
    pass