"""
Lt-Adaptive-Collapse
Adaptive Collapse Dynamics — Unified Variational Model
Based on "Adaptive Collapse Dynamics" by Locus (2026)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple


class AdaptiveCollapseDynamics:
    """
    Numerical implementation of the variational collapse theory.
    Solves approximated Euler-Lagrange equations from the generalized free energy.
    """
    
    def __init__(self,
                 m_S: float = 1.0,      # inertia of physical state
                 m_L: float = 0.8,      # adaptive inertia
                 k: float = 1.0,        # restoring force coefficient
                 mu: float = 0.3,       # margin cost
                 lambda_c: float = 0.8, # coupling strength
                 G: float = 0.0,        # attractor
                 noise_level: float = 0.05):
        
        self.m_S = m_S
        self.m_L = m_L
        self.k = k
        self.mu = mu
        self.lambda_c = lambda_c
        self.G = G
        self.noise_level = noise_level
        
        self.history = {'t': [], 'S': [], 'L': [], 'V': [], 'F': []}
    
    def free_energy(self, S: float, L: float) -> float:
        """Generalized Free Energy F = V + U + C"""
        if L <= 0:
            return 1e8
        V = 0.5 * self.k * (S - self.G)**2
        U = self.mu * np.log(1.0 / L)
        C = self.lambda_c * (S - self.G)**2 / L
        return V + U + C
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the second-order system (state + velocity)"""
        S, dS, L, dL = y
        
        # Clamp L to prevent singularity
        L = max(L, 1e-6)
        
        # Forces from -∂F/∂S and -∂F/∂L
        dF_dS = self.k * (S - self.G) + 2 * self.lambda_c * (S - self.G) / L
        dF_dL = -self.mu / L**2 - self.lambda_c * (S - self.G)**2 / L**2
        
        # Second derivatives (acceleration)
        d2S = (-dF_dS + np.random.normal(0, self.noise_level)) / self.m_S
        d2L = (-dF_dL + np.random.normal(0, self.noise_level * 0.5)) / self.m_L
        
        return [dS, d2S, dL, d2L]
    
    def simulate(self, t_span=(0, 200), y0=None, method='RK45'):
        """Run numerical integration"""
        if y0 is None:
            y0 = [0.1, 0.0, 1.0, 0.0]  # [S, dS, L, dL]
        
        sol = solve_ivp(self.derivatives, t_span, y0,
                       method=method, rtol=1e-5, atol=1e-8,
                       dense_output=True)
        
        t = sol.t
        y = sol.y
        
        # Compute Free Energy and Vitality
        V = []
        F_vals = []
        for i in range(len(t)):
            F_vals.append(self.free_energy(y[0,i], y[2,i]))
            # Approximate vitality (L²/t regime)
            V.append((y[2,i] ** 2) / (t[i] + 1e-6))
        
        self.history = {
            't': t,
            'S': y[0],
            'L': y[2],
            'V': np.array(V),
            'F': np.array(F_vals)
        }
        
        return self.history
    
    def plot(self):
        """Visualize results"""
        t = self.history['t']
        S = self.history['S']
        L = self.history['L']
        V = self.history['V']
        F = self.history['F']
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(t, L, 'b-', linewidth=2, label='L(t) — Stability Margin')
        plt.axhline(0.05, color='r', linestyle='--', label='Critical L')
        plt.title('Adaptive Stability Margin Degradation')
        plt.ylabel('L(t)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(t, V, 'g-', label='Vitality ≈ L²/t')
        plt.title('Vitality Dynamics')
        plt.ylabel('Vitality')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(t, S, 'purple', label='S(t)')
        plt.title('Physical State S(t)')
        plt.ylabel('S')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(t, F, 'r-', label='Free Energy F')
        plt.title('Generalized Free Energy')
        plt.xlabel('Time')
        plt.ylabel('F')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('adaptive_collapse_dynamics.png', dpi=220, bbox_inches='tight')
        plt.show()
        
        # Final status
        print(f"Final L: {L[-1]:.4f} | Final Vitality: {V[-1]:.4f}")
        if L[-1] < 0.05:
            print("✅ COLLAPSE OCCURRED (as predicted by theory)")


# ====================== DEMO ======================
if __name__ == "__main__":
    print("🚀 Adaptive Collapse Dynamics — Variational Simulation")
    print("Unified theory by Locus (2026)\n")
    
    model = AdaptiveCollapseDynamics(noise_level=0.08, lambda_c=0.9)
    history = model.simulate(t_span=(0, 150))
    model.plot()