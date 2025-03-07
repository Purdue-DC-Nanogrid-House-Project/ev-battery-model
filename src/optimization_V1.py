import cvxpy as cp
import numpy as np

class Optimizer:
    def __init__(self, battery_model, home_model,x0,dt):
        self.battery_model = battery_model
        self.home_model = home_model
        self.dt = dt
        self.x0 = x0
        self.K = 24/dt # determines number of entries

    def optimization_sub_battery(optimizer):
        # Define variables
        x_b = cp.Variable((1, optimizer.K+1))  # Battery state of charge
        P_bat = cp.Variable((optimizer.K, 1))  # Battery power
        P_util = cp.Variable((optimizer.K, 1))  # Utility power

        # Constraints
        constraints = [
            P_util == optimizer.home_model.demand + P_bat,
            x_b[0, 0] == optimizer.x0,
            x_b[:, 1:optimizer.K+1] == cp.multiply(x_b[:, :optimizer.K],optimizer.battery_model.sys_discrete.A) + cp.multiply(P_bat.T,optimizer.battery_model.sys_discrete.B),
            P_bat <= optimizer.battery_model.p_c_bar_b,
            P_bat >= -optimizer.battery_model.p_d_bar_b,

            #Physical Limits
            x_b >= 0.1*optimizer.x0,
            x_b <= 0.9*optimizer.x0
        ]

        # Objective function
        objective = cp.Minimize(
            cp.sum(
                # Uses L2 euclidean smoothing to ensure the are no sudden variations in Utility pull
                10 * cp.norm(optimizer.dt * (P_util - 0), 2) +   # (1) Minimizing utility power usage variations

                # Minimizing electricity cost by minimizing P_util
                optimizer.dt * cp.abs(P_util) +                  # (2) Minimizing electricity pulled while penalizing feeding bacoptimizer.K to the grid

                #Soft Preferecnes
                optimizer.dt * cp.maximum(0, x_b[:, :optimizer.K] - 0.8) + # (3) Penalizing exceeding max SOC (80%
                optimizer.dt * cp.maximum(0, 0.2 - x_b[:, :optimizer.K])   # (4) Penalizing dropping below min SOC (20%)
            )
        )

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)

        return x_b.value, P_bat.value, P_util.value
