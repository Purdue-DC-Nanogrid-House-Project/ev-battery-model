import pandas as pd
from datetime import datetime, timedelta

class HomeModel:
     def __init__(self,dt,day,i):
          self.dt = dt
          self.day = day
          self.i = i
          # self.start_time = "6/15/2024 0:00"
          # self.end_time = "6/16/2024 0:00"
          self.start_time, self.end_time = self.date_format()
          self.demand_path = "data/total_home_power_2.csv" 
          self.demand = self.load_demand_data(self.demand_path,self.start_time,self.end_time)

     def load_demand_data(self,file_path, start_time, end_time):
          demand_data = pd.read_csv(file_path)
          demand_data.columns = ['Time', 'Value']
          demand_data["Time"] = pd.to_datetime(demand_data["Time"])  # Convert to datetime

          # Find the index of start_time
          start_index = demand_data[demand_data["Time"] == pd.Timestamp(start_time)].index[0]
          end_index = demand_data[demand_data["Time"] == pd.Timestamp(end_time)].index[0]
          demand_data = demand_data.loc[start_index:end_index-1, ['Time', 'Value']]
          demand_data['Value'] = demand_data['Value'] / 1000

          demand_data.set_index('Time', inplace=True)
          freq = f'{int(self.dt*60)}min'
          demand_data = demand_data.resample(freq).interpolate(method='time')['Value']

          return demand_data
     
     def date_format(self):
          day_str = datetime.strptime(self.day, '%m/%d/%Y')
          offset = self.i * self.dt * 60  # total minutes to shift

          start_time_dt = day_str + timedelta(minutes=offset)
          end_time_dt = (day_str + timedelta(days=1)) + timedelta(minutes=offset)

          start_time = start_time_dt.strftime('%m/%d/%Y %H:%M:%S')
          end_time = end_time_dt.strftime('%m/%d/%Y %H:%M:%S')

          return start_time, end_time
     

if __name__ == "__main__":
    dt = 5/60  # 1 minute (for consistency with your data right now)
    day = "6/15/2024"  # test date that exists in your CSV
    
    home_model = HomeModel(dt, day,1)

    print("Start time:", home_model.start_time)
    print("End time:", home_model.end_time)
    print("Demand length:", len(home_model.demand))
    print("First 5 values:")
    print(home_model.demand.head())