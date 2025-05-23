import random
import os
from cart_pendulum_env import CartPendulumEnv 

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from gymnasium.utils.env_checker import check_env


# ======= CONFIG =======
mode = "test"  # Options: 'train', 'continue', 'test'
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model14')
# ======================


def test_function_random(env):
    episodes = 1
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        score = 0

        while not terminated and not truncated:
            env.render()
            action = random.choice([-1, 5])
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward

        print(f"[RANDOM] Episode {episode} finished with score {score}")

def test_function_trained(model, env):
    episodes = 1
    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        score = 0
        steps = 0
        steps += 1
        while not terminated and not truncated:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1
        print(f"[TRAINED] Episode {episode} finished with score {score:.2f} after {steps} steps")

def test_env(env):
    check_env(env)

def train_model(save_path):
    vec_env = DummyVecEnv([lambda: CartPendulumEnv()])
    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=100000)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print("Training complete and model saved.")
    vec_env.close()

def continue_training(model_path, more_timesteps=30000):
    env = DummyVecEnv([lambda: CartPendulumEnv()])
    model = PPO.load(model_path, env=env)
    model.learning_rate = 1e-5
    model.learn(total_timesteps=more_timesteps)
    model.save(model_path)
    env.close()
    print(f"Model re-trained with {more_timesteps} more timesteps and saved.")

# Main logic based on selected mode
if __name__ == "__main__":
    if mode == "train":
        print("Starting training...")
        train_model(PPO_path)

    elif mode == "continue":
        print("Continuing training...")
        continue_training(PPO_path, more_timesteps=30000)

    elif mode == "test":
        print("Loading model for evaluation and testing...")
        model = PPO.load(PPO_path)

        eval_env = CartPendulumEnv()
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Evaluation Result -> Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

        print("Testing trained agent...")
        test_function_trained(model, eval_env)

        eval_env.close()

    else:
        print(f"Unknown mode: {mode}. Use 'train', 'continue', or 'test'.")
