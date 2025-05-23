import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class CartPendulum:
    def __init__(self, m=0.1, M=1.0, l=0.5, g=9.81, d=0.1):
        self.m = m 
        self.M = M
        self.l = l
        self.g = g
        self.d = d
        self.boundary_min = -5.0
        self.boundary_max = 5.0
        self.J = (1/12) * m * (2 * l)**2
        self.solution = None

    def dynamics(self, t, state, u):
        x1, x2, x3, x4 = state
        m, M, l, g, d, J = self.m, self.M, self.l, self.g, self.d, self.J
        
        D = J + m * l**2
        total_mass = m + M
        sin_theta = np.sin(x3)
        cos_theta = np.cos(x3)

        # Calculate angular acceleration
        num = m * g * l * sin_theta - m * l * cos_theta * ((u - d * x2 + m * l * x4**2 * sin_theta) / total_mass)
        denom = D - (m**2 * l**2 * cos_theta**2) / total_mass
        x4_dot = num / denom

        # Calculate linear acceleration
        x2_dot = (u - d * x2 + m * l * (x4**2 * sin_theta - x4_dot * cos_theta)) / total_mass

        return [x2, x2_dot, x4, x4_dot]
    
    def linearize(self):
        m, M, l, g, d, J = self.m, self.M, self.l, self.g, self.d, self.J
        D = J + m * l**2 - (m**2 * l**2) / (m + M)
        
        A = np.zeros((4, 4))
        A[0, 1] = 1
        A[1, 1] = -d / (m + M) - (m**2 * l**2 * d) / ((m + M)**2 * D)
        A[1, 2] = -(m**2 * g * l**2) / ((m + M) * D)
        A[2, 3] = 1
        A[3, 1] = (m * l * d) / ((m + M) * D)
        A[3, 2] = (m * g * l) / D
        
        B = np.zeros((4, 1))
        B[1, 0] = 1 / (m + M) + (m**2 * l**2) / ((m + M)**2 * D)
        B[3, 0] = -(m * l) / ((m + M) * D)
        
        return A, B

    def simulate(self, initial_state, t_span=(0, 10), sampling_time=0.01, control_func=None):
        t_eval = np.arange(t_span[0], t_span[1] + sampling_time, sampling_time)
        
        def dyn(t, y):
            u = control_func(t, y) if control_func else 0.0
            if y[0] <= self.boundary_min or y[0] >= self.boundary_max:
                y[1] *= -0.5
                y[0] = np.clip(y[0], self.boundary_min, self.boundary_max)
            return self.dynamics(t, y, u)
        
        self.solution = solve_ivp(dyn, t_span, initial_state, t_eval=t_eval, method="RK45")
        return self.solution

    def step_by_step_simulation(self, initial_state, t_final=10.0, dt=0.01):
        times = [0.0]
        states = [initial_state]
        
        def step_dynamics(t, y):
            u = 0.0  # No control input
            if y[0] <= self.boundary_min or y[0] >= self.boundary_max:
                y[1] *= -0.5
                y[0] = np.clip(y[0], self.boundary_min, self.boundary_max)
            return self.dynamics(t, y, u)
        
        current_state = initial_state
        current_time = 0.0
        num_steps = int(t_final / dt)
        for _ in range(num_steps):
            sol = solve_ivp(step_dynamics, (current_time, current_time + dt), current_state, method="RK45")
            current_state = sol.y[:, -1]
            current_time += dt
            times.append(current_time)
            states.append(current_state)
        
        times = np.array(times)
        states = np.array(states)
        return times, states
