# test_ai.py
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
from train_ai_data import TemperatureControlEnv  # Path to the file that defines the environment

def main():
    # Load the trained PPO model from a file
    model = PPO.load("ppo_temperature_control")
    
    # Initialize the temperature control environment
    env = TemperatureControlEnv()
    
    # Test the model in 10 different episodes
    for episode in range(10):
        # Reset the environment and fetch the initial values
        obs, _ = env.reset()
        initial_temp, setpoint_temp = obs  # Initial and target temperatures
        done = False
        total_reward = 0  # Accumulate rewards during the episode
        
        print(f"Episode {episode + 1} begins:")
        print(f"Starting temperature: {initial_temp:.2f} | Target temperature: {setpoint_temp:.2f}")
        
        # Run the episode until it ends
        while not done:
            # Predict the next action using the trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Perform the action in the environment and update the state
            obs, reward, done, truncated, info = env.step(action)
            
            # Visualize the environment state
            env.render()
            
            # Add the reward to the total score
            total_reward += reward
        
        # Print the episode result
        print(f"Episode {episode + 1} ended, total score: {total_reward:.2f}\n")

if __name__ == "__main__":
    main()