import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, Q=None, R=None):
        self.Q = Q or np.diag([30, 3, 300, 30])
        self.R = R or np.array([[0.8]])
        self.K = None

    def compute_gain(self, A, B):
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ B.T @ P
        return self.K

    def get_control(self, x):
        return -self.K @ x

class SwingUpController:
    def __init__(self, cp, amplitude=5.0, frequency=0.5):
        self.cp = cp
        self.amplitude = amplitude  # Control force amplitude
        self.frequency = frequency  # Oscillation frequency
        
    def get_control(self, t, state):
        """Simple oscillation-based swing-up control"""
        theta = state[2] % (2*np.pi)
        

        # Sinusoidal forcing function
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        

class EnergyBasedSwingUpController:
    def __init__(self, cp, k_energy=1.0):
        """
        Energy-based swing-up controller (no damping version)
        Args:
            cp: CartPendulum instance
            k_energy: Energy control gain
        """
        self.cp = cp
        self.k_energy = k_energy
        self.E_desired = 2 * self.cp.m * self.cp.g * self.cp.l  # Energy at upright position

    def get_control(self, t, state):
        """Energy-based control law without damping"""
        _, x_dot, theta, theta_dot = state
        m, l, g, M, d = self.cp.m, self.cp.l, self.cp.g, self.cp.M, self.cp.d
        
        # Current total energy (kinetic + potential)
        E_current = 0.5 * m * (l * theta_dot)**2 +  m * g * l * (1 + np.cos(theta)) + 0.5 * M * x_dot**2
        
        # Energy error
        E_error = E_current - self.E_desired
        
        # Pure energy control law 
        signum = np.tanh(10 * theta_dot * np.cos(theta))
        u = self.k_energy*E_error*signum - d*x_dot
        
        return np.clip(u, -50, 50)