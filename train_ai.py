# train_ai.py
# MIT License
# Copyright (c) 2024 Timo Heimonen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
        learning_rate=0.0003,   # Learning rate, adjustable as needed
        gamma=0.99,             # Discount factor for long-term decisions
        ent_coef=0.001,         # Entropy coefficient, helps explore alternatives
        clip_range=0.1          # Clipping range for stable policy updates
    )
    
    # Train the PPO model with a high number of steps to ensure learning
    model.learn(total_timesteps = 2000000)
    
    
    # Save the trained model to a file
    model.save("ppo_temperature_control")
    print("Model saved to 'ppo_temperature_control.zip'.")

if __name__ == "__main__":
    main()