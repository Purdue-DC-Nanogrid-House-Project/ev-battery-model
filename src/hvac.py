import numpy as np
import math

class HVAC:
   def __init__(self):
      #— Heat pump & tank constants (from before) —
      self.Rout   = 2.0424 * 1.15
      self.Rm     = 1.0616
      self.C      = 6.5   * 1.1
      self.db     = 1.0
      self.Tm     = 20.61

      # PI‐controller state
      self.prev_I     = 0.0
      self.aux1_timer = 0
      self.aux2_timer = 0
      self.hp_timer   = 0

      # PI gains
      self.P_gain = 0.4 * 20.0
      self.I_gain = 1.0 * 0.0004 * (5*60)/2.0

   def emulate_heat_pump(self, T, Tset, theta, dt_fast):
       """
       Emulate one timestep of the heat pump:
         T       : current tank temperature (°C)
         Tset    : setpoint temperature (°C)
         theta   : ambient temperature (scalar or np.array, °C)
         dt_fast : simulation timestep (s)
       Returns:
         T_next, Q_total, I_total
       and updates internal timers & integrator
       """
       # 1) Performance map η(θ)
       theta_arr = np.atleast_1d(theta)
       η = np.where(theta_arr >= -24,
                    0.0008966 * theta_arr ** 2 + 0.1074 * theta_arr + 3.098,
                    1.0)

       # 2) Available capacities (kW)
       PmaxHP = np.clip(-0.0149 * theta_arr + 4.0934, None, 4.0)
       Q_hp = η * PmaxHP
       Q_aux1 = 9.6
       Q_aux2 = 0.0
       pf_hp = 0.8
       pf_aux = 1.0

       # 3) Timers thresholds (in timesteps)
       n_hp = int(5 * (5 / 60) / dt_fast)
       n_aux1 = int(12 * (5 / 60) / dt_fast)

       # 4) PI control
       e = max(Tset - T, 0.0)
       limiter = 100 * (5 * 60) / 2.0
       self.prev_I = np.clip(self.prev_I + e * dt_fast, -limiter, limiter)
       # reset if well above setpoint
       if T > Tset + self.db / 2:
           self.prev_I = 0.0
           e = 0.0
       PI_out = self.P_gain * e + self.I_gain * self.prev_I
       PI_dim = PI_out  # since P_nominal=1

       # 5) Heat output scheduling
       Q_total = 0.0
       I_total = 0.0
       if PI_dim > 0:
           # base heat pump
           Q_total = Q_hp
           I_total = (Q_hp / η) / pf_hp
           # timers for staging
           if PI_dim > 2:
               self.hp_timer += 1
           else:
               self.hp_timer = 0
               self.aux1_timer = 0

           # stage 1 aux
           if (PI_dim > 2 and self.hp_timer > n_hp) or (
                   self.aux1_timer > 0 and self.aux2_timer == 0):
               if self.aux1_timer < n_aux1:
                   self.aux1_timer += 1
                   Q_total += Q_aux1
                   I_total += Q_aux1 / pf_aux
               else:
                   self.aux1_timer = 0
                   self.hp_timer = 0

           # stage 2 aux
           if PI_dim > 4 and self.hp_timer > n_hp:
               Q_total += Q_aux2
               I_total += Q_aux2 / pf_aux

       # 6) Tank temperature update (first-order)
       R_eff = (self.Rout * self.Rm) / (self.Rout + self.Rm) * 1.15
       a_factor = math.exp(-dt_fast / (R_eff * self.C))
       theta_cor = (self.Rm * theta_arr + self.Rout * self.Tm) / (
                   self.Rout + self.Rm)
       T_next = theta_cor + (T - theta_cor) * a_factor + (
                   Q_total * R_eff * (1 - a_factor) / self.C)

       return T_next, Q_total, I_total

   def mirror_controls(self, PI_dim):
       """
       Mimic manufacturer control thresholds:
         PI_dim : nondimensional control signal
       Returns a dict of on/off flags for each stage.
       """
       return {
           'compressor_on': PI_dim > 0,
           'aux1_on': PI_dim > 2,
           'aux2_on': PI_dim > 4
       }

   def state_space(self, dt):
      """
      Returns the building's thermal state‐space matrices:
        Continuous:  ẋ = A_c x + Bc [θ_cor; Q_in],  y = C x + D [θ_cor; Q_in]
        Discrete:    xₖ₊₁ = A_d xₖ + B_d [θ_cor; Q_in],  yₖ = C xₖ + D [θ_cor; Q_in]
      where θ_cor = (Rm·θ + Rout·Tm)/(Rout+Rm) is the corrected ambient input.
      """
      # effective single‐resistance
      R_eff = (self.Rout * self.Rm) / (self.Rout + self.Rm)

      # continuous‐time
      A_c = np.array([[-1.0/(R_eff * self.C)]])
      B_c = np.array([[ 1.0/(R_eff * self.C),   # to θ_cor
                        1.0/self.C             # to Q_in (heat pump + base loads)
                     ]])
      C_c = np.array([[1.0]])
      D_c = np.zeros((1,2))

      # discrete‐time (ZOH)
      a = math.exp(-dt/(R_eff * self.C))
      A_d = np.array([[a]])
      # ∫₀ᵈᵗ e^{A_cτ} dτ · B_c = [ (1 − a),  R_eff*(1 − a) ]
      B_d = np.array([[ (1 - a),
                        R_eff*(1 - a)
                     ]])
      C_d = C_c.copy()
      D_d = D_c.copy()

      return {
         'continuous': {
            'A':  A_c,  'B': B_c,
            'C':  C_c,  'D': D_c
         },
         'discrete': {
            'A':  A_d,  'B': B_d,
            'C':  C_d,  'D': D_d
         }
      }

if __name__ == "__main__":
   # Demo usage
   hvac = HVAC()
   dt = 600.0  # 10-minute timestep
   mats = hvac.state_space(dt)
   print("Discrete-time A matrix:", mats['discrete']['A'])
   print("Discrete-time B matrix:", mats['discrete']['B'])

   # Example simulation step
   T    = 20.0
   Tset = 22.0
   theta = np.array([-5.0])
   T_next, Q, I = hvac.emulate_heat_pump(T, Tset, theta, dt)
   # Unwrap scalars if returned as 1-element arrays
   if isinstance(T_next, np.ndarray): T_next = T_next.item()
   if isinstance(Q, np.ndarray):      Q      = Q.item()
   if isinstance(I, np.ndarray):      I      = I.item()

   PI_dim = hvac.P_gain*max(Tset-T,0.0) + hvac.I_gain*hvac.prev_I
   flags  = hvac.mirror_controls(PI_dim)
   print(f"T_next={T_next:.2f}, Heat output={Q:.2f} kW, Controls={flags}")







