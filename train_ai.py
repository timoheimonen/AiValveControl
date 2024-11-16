# train_ai.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from train_ai_data import TemperatureControlEnv  # Varmista, että polku on oikein

def main():
    # Luo ympäristö
    env = TemperatureControlEnv()
    
    # Tarkista ympäristö
    check_env(env)
    
    # Luo PPO-malli
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0001,  # Voit säätää oppimisnopeutta tarpeen mukaan
        gamma=0.99,             # Diskonttaustekijä
        ent_coef=0.001,          # Entropia-kerroin kannustaa eksploraatioon
        clip_range=0.1         # PPO:n klippausalue
    )
    
    # Kouluta mallia
    model.learn(total_timesteps=2000000)
    
    # Tallenna malli
    model.save("ppo_temperature_control")
    print("Malli tallennettu tiedostoon 'ppo_temperature_control.zip'.")

if __name__ == "__main__":
    main()