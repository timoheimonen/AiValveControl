# train_ai.py
# Licensed under the MIT License.
# Copyright (c) 2024 Timo Heimonen.
# See the LICENSE file in the project root for more details.

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from train_ai_data import TemperatureControlEnv  # Path checked, ensure the file exists

def main():
    # Initialize the environment for temperature control
    env = TemperatureControlEnv()
    
    # Check that the environment complies with Gym standards
    check_env(env)
    
    # Create a PPO model for learning
    model = PPO(
        "MlpPolicy",            # Use a basic multilayer perceptron for the policy
        env,                    # Simulated environment
        verbose=1,              # Display learning logs
        learning_rate=0.0001,   # Learning rate, adjustable as needed
        gamma=0.99,             # Discount factor for long-term decisions
        ent_coef=0.001,         # Entropy coefficient, helps explore alternatives
        clip_range=0.1          # Clipping range for stable policy updates
    )
    
    # Train the PPO model with a high number of steps to ensure learning
    model.learn(total_timesteps=2000000)
    
    # Save the trained model to a file
    model.save("ppo_temperature_control")
    print("Model saved to 'ppo_temperature_control.zip'.")

if __name__ == "__main__":
    main()