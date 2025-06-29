import numpy as np
from scipy import signal

# Physical constants
GALLONS_PER_M3 = 264.17  # gallons in one cubic meter
CP_KWH_PER_KG_K = 0.001163056  # specific heat of water, kWh per kg per °C
WATER_DENSITY_KG_PER_M3 = 997  # density of water, kg per m^3

class WaterHeater:
   """
   Simulates a water heater with both resistive and heat-pump modes.

   Parameters
   ----------
   dt : float
      Simulation time step in hours.
   heater_type : str
      Type of water heater. One of {'hybrid-hpwh', 'erwh', 'hpowh'} (case-insensitive).

   Attributes
   ----------
   resistance : float
      Resistive heating power in kW.
   compressor : float
      Heat pump heating power in kW.
   C : float
      Thermal capacitance of the tank (kWh per °C).
   R_ul : float
      Thermal resistance between upper and lower layers (°C per kW).
   R_u : float
      Thermal resistance of upper layer to ambient (°C per kW).
   R_l : float
      Thermal resistance of lower layer to ambient (°C per kW).
   height : float
      Fractional height of the upper layer.
   mixing_fraction : float
      Fraction of mass flow that enters the upper layer.
   dt : float
      Simulation time step in hours.
   """

   WATER_HEATER_TYPES = {
      "hybrid-hpwh": {"resistance": 4.5, "compressor": 0.45},
      "erwh": {"resistance": 4.5, "compressor": 0.0},
      "hpowh": {"resistance": 0.0, "compressor": 0.45},
   }

   def __init__(self, dt: float, heater_type: str):
      self.dt = dt
      self.heater_type = heater_type.lower()
      if self.heater_type not in self.WATER_HEATER_TYPES:
         valid = ", ".join(self.WATER_HEATER_TYPES)
         raise ValueError(f"Invalid heater_type '{heater_type}'. Valid types: {valid}")

      props = self.WATER_HEATER_TYPES[self.heater_type]
      self.resistance = props["resistance"]
      self.compressor = props["compressor"]

      # Compute tank properties
      self._compute_tank_properties()

   def _compute_tank_properties(self):
      """Compute and cache physical parameters of the water tank."""
      tank_volume_m3 = 169.901011 / 1000  # convert liters to cubic meters
      self.C = WATER_DENSITY_KG_PER_M3 * CP_KWH_PER_KG_K * tank_volume_m3

      # Geometry
      tank_radius = 0.2286  # m
      self.height = tank_volume_m3 / (np.pi * tank_radius**2)  # m
      tank_area = 2 * tank_volume_m3 / tank_radius  # m^2

      # Thermal resistance calculation (English → metric unit conversion)
      tank_thickness_m = 0.0016256  # m
      tank_conductance_btu = 31  # BTU/(hr·ft·°F)
      insulation_coef_btu = 0.08  # BTU/(hr·ft^2·°F)
      # R_tank in °C·m^2/kWh
      r_tank = (
         (tank_thickness_m / tank_conductance_btu + 1 / insulation_coef_btu)
         * (5/9) * (1/10.8) / (1/3412)
      )

      self.R_ul = 1 / (tank_area * 0.000650 / 0.025)
      self.mixing_fraction = 0.3
      self.R_u = r_tank * 0.5
      self.R_l = r_tank * 0.5

   def discretize(self, W_hat: np.ndarray, T_in: np.ndarray, T_win: np.ndarray):
      """
      Discretize the time-varying continuous model into A, B, w over K steps.

      Parameters
      ----------
      W_hat : array-like
         Mass flow rate at each time step (length K).
      T_in : array-like
         Ambient temperature at each time step (length K).
      T_win : array-like
         Inlet water temperature at each time step (length K).

      Returns
      -------
      A : ndarray, shape (2,2,K)
         Discrete state-to-state matrices.
      B : ndarray, shape (2,K)
         Discrete input-to-state vectors.
      w : ndarray, shape (2,K)
         Discrete disturbance vectors.
      """
      W = np.atleast_1d(W_hat).ravel()
      Tin = np.atleast_1d(T_in).ravel()
      Tw = np.atleast_1d(T_win).ravel()
      K = W.size

      A = np.zeros((2,2,K))
      B = np.zeros((2,K))
      w = np.zeros((2,K))

      # Continuous input matrix (2×1), then normalize by C
      Bt = np.array([
         [self.mixing_fraction / self.height],
         [(1 - self.mixing_fraction) / (1 - self.height)]
      ]) / self.C
      I2 = np.eye(2)

      for k in range(K):
         # Continuous-time state matrix At
         At = np.zeros((2,2))
         At[0,0] = -(1/self.height)*(W[k]*CP_KWH_PER_KG_K + 1/self.R_ul + 1/self.R_u)
         At[0,1] = (1/self.height)*(1/self.R_ul + W[k]*CP_KWH_PER_KG_K)
         At[1,0] = (1/(1-self.height))*(1/self.R_ul)
         At[1,1] = -(1/(1-self.height))*(W[k]*CP_KWH_PER_KG_K + 1/self.R_ul + 1/self.R_l)
         At /= self.C

         # Continuous-time disturbance vector
         wt = np.zeros(2)
         wt[0] = Tin[k]/self.R_u
         wt[1] = Tin[k]/self.R_l + W[k]*CP_KWH_PER_KG_K*Tw[k]
         wt /= self.C

         # Build and discretize, unpacking the tuple output
         dA, dB, _, _, _ = signal.cont2discrete((At, np.hstack((Bt, I2)), I2, np.zeros((2,3))), self.dt, method='zoh')

         A[:,:,k] = dA
         B[:,k] = dB[:,0]
         w[:,k] = dB[:,1:] @ wt

      return A, B, w

   def sim_real_control(self, Tsp: float, TH: float, TL: float, A: np.ndarray, B: np.ndarray, w: np.ndarray, COP: float, maxPwr: float):
      """
      Simulate one time-step of the manufacturer thermostatic controller.

      Parameters
      ----------
      Tsp : float
         Temperature setpoint in °C.
      TH : float
         Current upper-node temperature in °C.
      TL : float
         Current lower-node temperature in °C.
      A : ndarray, shape (2,2,K)
         Discrete state-to-state matrices.
      B : ndarray, shape (2,K)
         Discrete input-to-state vectors.
      w : ndarray, shape (2,K)
         Discrete disturbance vectors.
      COP : float
         Heat pump coefficient of performance.
      maxPwr : float
         Maximum electric power of heat pump in kW.

      Returns
      -------
      q_sim : float
         Applied heating power (thermal) in kW for this step.
      P_sim : float
         Electrical power drawn in kW for this step.
      TH_next : float
         Next upper-node temperature in °C.
      TL_next : float
         Next lower-node temperature in °C.
      """
      # Thermostat deadband half-width
      delta = 2.6
      # Initialize q to zero
      q = 0.0
      # If temperature below lower deadband, turn on at full COP-adjusted power
      if TH < Tsp - delta:
         q = COP * maxPwr
      # If at or above setpoint, turn off
      elif TH >= Tsp:
         q = 0.0
      # State update for current time step index 0
      x = np.array([TH, TL])
      # Use first slice (current step)
      A0 = A[:, :, 0]
      B0 = B[:, 0]
      w0 = w[:, 0]
      x_next = A0 @ x + B0 * q + w0
      TH_next, TL_next = x_next[0], x_next[1]
      # Electrical power
      P = q / COP if COP > 0 else 0.0
      return q, P, TH_next, TL_next

if __name__ == "__main__":
   # Example usage
   wh1 = WaterHeater(dt=1.0, heater_type="hybrid-HPWH")
   print(f"Hybrid HPWH: Resistance={wh1.resistance} kW, Compressor={wh1.compressor} kW")

   K = 5
   W_hat = np.array([10, 12, 11, 9, 13])
   Tin = np.full(K, 20.0)
   Tw = np.array([15, 16, 15.5, 14, 17])

   A, B, w = wh1.discretize(W_hat, Tin, Tw)
   print("A[:,:,0] =", A[:,:,0])
   print("B[:,0] =", B[:,0])
   print("w[:,0] =", w[:,0])

   try:
      WaterHeater(dt=1.0, heater_type="solar")
   except ValueError as e:
      print("Expected error:", e)


