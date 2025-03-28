
class Optimizer:
    def __init__(self, battery_model,ev_model,home_model,solar_model,x0_b,x0_ev,dt):
        self.battery_model = battery_model
        self.ev_model = ev_model
        self.solar_model = solar_model
        self.home_model = home_model
        self.dt = dt
        self.x0_b = x0_b
        self.x0_ev = x0_ev
        self.K = int(24/dt)

  