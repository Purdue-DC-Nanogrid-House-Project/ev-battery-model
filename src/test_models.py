import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test_ev_charging(ev_model, battery_model, initial_charge, target_charge):
    # Convert percentages to state of charge (SoC) in kWh for ev
    initial_soc_ev = initial_charge * ev_model.x_bar_ev
    target_soc_ev = target_charge * ev_model.x_bar_ev

    # Convert percentage to state of charge (SoC) in Kwh for charger
    initial_soc_b = 1.00 * battery_model.x_bar_b #assumes full charge

    time_range = np.arange(0, 25, ev_model.dt)  # Time range for 25 hours

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
    

    for i, t in enumerate(df_ev.iloc[:, 0]):
            # Check if the current time is greater than 2 hours
            if t > 2:
                # Check if the current SoC is less than the target SoC
                if df_ev.iloc[i,1] < target_soc_ev:
                   break
                else:
                    # If the target SoC is reached, set charger power to 0
                    df_ev.iloc[i, 2] = 0  # No charging power
                    df_b.iloc[i, 2] = 0  # No battery power delivered


if __name__ == "__main__":
    # This block is not needed since we will call the function from main.py
    pass