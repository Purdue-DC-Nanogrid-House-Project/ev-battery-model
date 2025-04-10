import pandas as pd

class HomeModel:
     def __init__(self,dt):
          self.dt = dt
          self.start_time = "6/15/2024 0:00"
          self.end_time = "6/16/2024 0:00"
          self.demand_path = "data/total_home_power_2.csv" 
          self.demand = self.load_demand_data(self.demand_path,self.start_time,self.end_time)

     def load_demand_data(self,file_path, start_time, end_time):
          demand_data = pd.read_csv(file_path)
          demand_data.columns = ['Time', 'Value']
          demand_data["Time"] = pd.to_datetime(demand_data["Time"])  # Convert to datetime

          # Find the index of start_time
          start_index = demand_data[demand_data["Time"] == pd.Timestamp(start_time)].index[0]
          end_index = demand_data[demand_data["Time"] == pd.Timestamp(end_time)].index[0]
          demand_data = demand_data.loc[start_index:end_index-1, 'Value']/1000

          return demand_data

if __name__ == "__main__":
     dt = 1.0
     demand = 15 #KW