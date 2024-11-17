# train_ai_data
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
from gymnasium import spaces
import numpy as np
from collections import deque

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super(TemperatureControlEnv, self).__init__()
        
        # Initialize valve adjustment tracking
        self.previous_valve_adjust = 50.0  # Middle setting (0-100%)
        self.valve_adjust_queue = deque(maxlen=5)  # Simulate delay, e.g., 5-step length
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [Current temperature, Target temperature]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),  # Minimum temperature limits
            high=np.array([100.0, 100.0], dtype=np.float32),  # Maximum temperature limits
            dtype=np.float32
        )
        
        self.noise_level = 0.5  # Noise level in temperature changes
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(1, 80)  # Random initial temperature
        self.setpoint_temp = int(np.random.uniform(35, 75))  # Random target temperature
        self.previous_valve_adjust = 50.0  # Reset valve adjustment
        self.valve_adjust_queue.clear()  # Clear delay queue
        for _ in range(self.valve_adjust_queue.maxlen):
            self.valve_adjust_queue.append(50.0)  # Initialize delay queue to middle setting
        
        observation = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # Scale action from [-1, 1] → [0, 100%]
        current_valve_adjust = np.clip((action[0] + 1.0) * 50.0, 0, 100)
        
        # Update the valve adjustment delay queue
        self.valve_adjust_queue.append(current_valve_adjust)
        delayed_valve_adjust = self.valve_adjust_queue[0]  # Use delayed adjustment
        
        # Calculate temperature change based on delayed adjustment
        temp_change = (delayed_valve_adjust / 100.0) * 60.0 - 30.0  # Change range: (-15.0, 15.0)
        self.current_temp += temp_change
        self.current_temp += np.random.normal(0, self.noise_level)  # Add some random noise
        
        # Restrict temperature to a reasonable range
        self.current_temp = np.clip(self.current_temp, 0, 100)
        
        # Calculate error between current temperature and target temperature
        error = abs(self.current_temp - self.setpoint_temp)
        
        # Add penalty for large valve adjustments
        valve_change_penalty = abs(current_valve_adjust - self.previous_valve_adjust) * 0.1
        self.previous_valve_adjust = current_valve_adjust  # Update the most recent adjustment

        # Reward and penalty logic
        if error <= 1.0:
            reward = 10.0 - valve_change_penalty  # Maximum reward, but penalize abrupt changes
            done = True  # Episode ends when the target is achieved
        else:
            # Larger penalty when moving further from the tolerance area
            reward = -0.1 * (error ** 2) - valve_change_penalty
            reward = np.clip(reward, -10, 10)
            done = False
       
        # Return state, reward, and additional information
        obs = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return obs, reward, bool(done), False, {"valve_adjust": delayed_valve_adjust}

    def render(self, mode='human'):
        print(f"Current temperature: {self.current_temp:.2f} | Target temperature: {self.setpoint_temp:.2f}")