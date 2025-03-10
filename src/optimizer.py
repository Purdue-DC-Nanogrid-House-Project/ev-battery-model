
class Optimizer:
    def __init__(self, battery_model, home_model,x0,dt):
        self.battery_model = battery_model
        self.home_model = home_model
        self.dt = dt
        self.x0 = x0
        self.K = int(24/dt)

  