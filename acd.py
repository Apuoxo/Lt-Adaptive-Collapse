"""
Lt-Adaptive-Collapse
Unified Variational Model of Adaptive Collapse Dynamics
Author: Locus (Stanislav Usychenko) — 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class AdaptiveCollapseDynamics:
    """
    Numerical solver for the variational collapse theory.
    Implements generalized free energy and approximated Euler-Lagrange equations.
    """
    def __init__(self,
                 m_S: float = 1.0,
                 m_L: float = 0.8,
                 k: float = 1.0,
                 mu: float = 0.35,
                 lambda_c: float = 0.85,
                 G: float = 0.0,
                 noise_level: float = 0.07):
        
        self.m_S = m_S
        self.m_L = m_L
        self.k = k
        self.mu = mu
        self.lambda_c = lambda_c
        self.G = G
        self.noise_level = noise_level
        
        self.history = None

    def free_energy(self, S: float, L: float) -> float:
        """F = V(S-G) + U(L) + C(S,L)"""
        if L < 1e-8:
            return 1e8
        V = 0.5 * self.k * (S - self.G)**2
        U = self.mu * np.log(1.0 / max(L, 1e-8))
        C = self.lambda_c * (S - self.G)**2 / max(L, 1e-8)
        return V + U + C

    def derivatives(self, t: float, y: np.ndarray):
        S, dS_dt, L, dL_dt = y
        L = max(L, 1e-8)
        
        # Partial derivatives of F
        dF_dS = self.k * (S - self.G) + 2 * self.lambda_c * (S - self.G) / L
        dF_dL = -self.mu / L**2 - self.lambda_c * (S - self.G)**2 / L**2
        
        # Accelerations
        d2S_dt2 = (-dF_dS + np.random.normal(0, self.noise_level)) / self.m_S
        d2L_dt2 = (-dF_dL + np.random.normal(0, self.noise_level*0.6)) / self.m_L
        
        return [dS_dt, d2S_dt2, dL_dt, d2L_dt2]

    def simulate(self, t_max: float = 300):
        """Run the simulation"""
        y0 = [0.05, 0.0, 1.0, 0.0]   # [S, dS, L, dL]
        
        sol = solve_ivp(self.derivatives, [0, t_max], y0,
                        method='RK45', rtol=1e-6, atol=1e-8, max_step=0.5)
        
        t = sol.t
        y = sol.y
        
        # Compute observables
        V = (y[2]**2) / (t + 1e-6)
        F = np.array([self.free_energy(y[0,i], y[2,i]) for i in range(len(t))])
        
        self.history = {
            't': t,
            'S': y[0],
            'L': y[2],
            'V': V,
            'F': F
        }
        return self.history

    def print_results(self):
        if not self.history:
            print("Run simulation first!")
            return
        
        t = self.history['t']
        L = self.history['L']
        V = self.history['V']
        F = self.history['F']
        
        print("\n=== SIMULATION RESULTS ===")
        print(f"Duration:          {t[-1]:.1f} time units")
        print(f"Final Stability Margin L: {L[-1]:.4f}")
        print(f"Final Vitality (L²/t):    {V[-1]:.4f}")
        print(f"Final Free Energy F:      {F[-1]:.2f}")
        
        if L[-1] < 0.08:
            print("✅ COLLAPSE OCCURRED — theory prediction confirmed")
        else:
            print("System still surviving at the end of simulation")


# ====================== RUN SIMULATION ======================
if __name__ == "__main__":
    print("🚀 Adaptive Collapse Dynamics — Variational Simulation")
    print("Unified theory by Locus (2026)\n")
    
    model = AdaptiveCollapseDynamics(noise_level=0.075)
    history = model.simulate(t_max=250)
    model.print_results()
    
    # Optional: save plot (if matplotlib works in your environment)
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(2,2,1); plt.plot(history['t'], history['L']); plt.title('L(t)'); plt.grid()
        plt.subplot(2,2,2); plt.plot(history['t'], history['V']); plt.title('Vitality L²/t'); plt.grid()
        plt.subplot(2,2,3); plt.plot(history['t'], history['S']); plt.title('S(t)'); plt.grid()
        plt.subplot(2,2,4); plt.plot(history['t'], history['F']); plt.title('Free Energy F'); plt.grid()
        plt.tight_layout()
        plt.savefig('collapse_dynamics.png', dpi=200)
        print("\nPlot saved as 'collapse_dynamics.png'")
    except:
        pass