import numpy as np
from gymnasium import Env, spaces
from cart_pendulum import CartPendulum
import matplotlib.pyplot as plt

class CartPendulumEnv(Env):
    def __init__(self, m=0.2, M=1.0, l=0.5, g=9.81, d=0.05):
        super(CartPendulumEnv, self).__init__()
        
        # Initialize the cart-pendulum system
        self.system = CartPendulum(m=m, M=M, l=l, g=g, d=d)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [x, x_dot, theta, theta_dot]
        high = np.array([
            4.0,   # x position limit
            100.0,  # x velocity limit
            20 *np.pi / 180,  # theta angle limit
            100.0    # theta velocity limit
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 500
        self.state = None
        
        # Accumulation rendering parameters
        self._accumulated_steps = 3  # Render every N steps
        self._step_counter = 0
        
        # Rendering attributes
        self.fig = None
        self.ax = None
        self.line = None
        self.cart_box = None
        self.cart_width = 0.4
        self.cart_height = 0.2
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset to random initial state near the bottom
        self.state = np.array([
            self.np_random.uniform(low=-0.1, high=0.1),  # x
            0.0,                                         # x_dot
            self.np_random.uniform(low=-0.1, high=0.1), # theta 
            0.0                                          # theta_dot
        ], dtype=np.float32)
        
        self.current_step = 0
        self._step_counter = 0
        self.system.solution = None  # Clear previous solution
        
        info = {}
        return self.state, info
    
    def step(self, action):
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        force = action * 5.0
        # Simulate one step (0.01 seconds)
        sol = self.system.simulate(
            self.state, 
            t_span=(0, 0.01), 
            sampling_time=0.01,
            control_func=lambda t, s: force
        )
        
        next_state = sol.y[:, -1].astype(np.float32)
        self.current_step += 1
        self._step_counter += 1
        
        # Calculate reward
        theta = (next_state[2] + np.pi) % (2 * np.pi) - np.pi
        reward = self._calculate_reward(next_state, theta)
        
        # Check termination conditions
        terminated = self._check_done(next_state, theta)
        truncated = self.current_step >= self.max_steps
        
        # Update state
        self.state = next_state
        
        info = {}
        return next_state, reward, terminated, truncated, info
    
    def _calculate_reward(self, state, theta):
        """Custom reward function"""
        reward = -(
            theta**2 + 
            0.01 * state[3]**2 +  # theta_dot
            0.5 * state[0]**2 +   # x position
            0.01 * state[1]**2    # x velocity
        )
        if abs(theta) < 1 * np.pi / 180:
            reward += 2.0 * np.exp(-theta**2 / (0.05**2))
        reward += 1
        return float(reward)
    
    def _check_done(self, state, theta):
        
        # Pendulum fell too far
        if abs(theta) >= 10 * np.pi / 180 and abs(state[3]) >= 2.4:
            return True
        
        return False
    
    def render(self):
            
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            self.ax.set_xlim(self.system.boundary_min - 2, self.system.boundary_max + 2)
            self.ax.set_ylim(-self.system.l*2, self.system.l*2)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title("Cart-Pendulum Environment")

            # Draw static boundaries
            self.ax.axvline(self.system.boundary_min, color='red', linestyle='--')
            self.ax.axvline(self.system.boundary_max, color='red', linestyle='--')

            # Create the cart and pendulum elements
            self.line, = self.ax.plot([], [], 'o-', lw=2, markersize=6)
            self.cart_box = plt.Rectangle(
                (0, 0), 
                self.cart_width, 
                self.cart_height, 
                fc='blue', 
                ec='black', 
                alpha=0.8
            )
            self.ax.add_patch(self.cart_box)

            plt.ion()
            plt.show()
        
        # Only render every N steps
        if self._step_counter >= self._accumulated_steps:
            x, _, theta, _ = self.state
            l = self.system.l

            # Pendulum tip position
            pend_x = x + l * np.sin(theta)
            pend_y = l * np.cos(theta)

            # Update pendulum line
            self.line.set_data([x, pend_x], [0, pend_y])

            # Update cart position
            self.cart_box.set_xy((x - self.cart_width / 2, -self.cart_height / 2))

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self._step_counter = 0
            
        
        return None
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.line = None
            self.cart_box = None