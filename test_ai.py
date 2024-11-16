# test_ai.py

import gymnasium as gym
from stable_baselines3 import PPO
from train_ai_data import TemperatureControlEnv  # Varmista, että polku on oikein

def main():
    # Lataa koulutettu malli
    model = PPO.load("ppo_temperature_control")
    
    # Luo ympäristö
    env = TemperatureControlEnv()
    
    # Kokeile mallia
    for episode in range(10):  # Suorita 10 episodia
        obs, _ = env.reset()
        initial_temp, setpoint_temp = obs  # Lähtölämpötila ja tavoitelämpötila
        done = False
        total_reward = 0
        print(f"Episodi {episode + 1} alkaa:")
        print(f"Lähtölämpötila: {initial_temp:.2f} | Tavoitelämpötila: {setpoint_temp:.2f}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            total_reward += reward
        
        print(f"Episodi {episode + 1} päättyi, kokonaisscore: {total_reward:.2f}\n")

if __name__ == "__main__":
    main()