# Cart-Pendulum Stabilization using Reinforcement Learning (PPO)

This project implements a reinforcement learning solution to **stabilize** a cart-pendulum system near the upright equilibrium using **Proximal Policy Optimization (PPO)**. The pendulum starts in a random state close to upright (no swing-up), and the agent learns to keep it balanced.

## Features

- Physics-based `CartPendulum` model
- Custom `Gymnasium` environment (`CartPendulumEnv`)
- PPO-based training, testing, and evaluation routines
- Human control option with keyboard (`A` and `D`)
- Designed for stabilization only (no swing-up phase)
- Optional classical controller (LQR) for comparison

## Requirements

- Python 3.8+
- `gymnasium`
- `stable-baselines3`
- `matplotlib`
- `numpy`
- `keyboard`

## Environment Setup

- Initial state is randomly set **near the upright** position:
- The agent focuses only on **balancing**, not on swinging up.

## Acknowledgments

- Built using `Gymnasium` and `Stable-Baselines3`
- Classical control references: LQR-based balancing
