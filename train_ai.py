# train_ai.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from train_ai_data import TemperatureControlEnv  # Polku tarkistettu, varmista tiedoston olemassaolo

def main():
    # Alustetaan ympäristö lämpötilan säätöä varten
    env = TemperatureControlEnv()
    
    # Tarkistetaan, että ympäristö vastaa Gym-standardia
    check_env(env)
    
    # Luodaan PPO-malli oppimiseen
    model = PPO(
        "MlpPolicy",            # Käytetään perus monikerrosperceptronia politiikkaan
        env,                    # Simuloitu ympäristö
        verbose=1,              # Näytetään oppimisen lokit
        learning_rate=0.0001,   # Oppimisnopeus, voi säätää tilanteen mukaan
        gamma=0.99,             # Diskonttaustekijä pitkän aikavälin päätöksille
        ent_coef=0.001,         # Entropiakerroin, auttaa tutkimaan vaihtoehtoja
        clip_range=0.1          # Klippausalue politiikan päivitysten vakaudelle
    )
    
    # Koulutetaan PPO-mallia, asetettu suuri määrä askeleita oppimisen varmistamiseksi
    model.learn(total_timesteps=2000000)
    
    # Tallennetaan koulutettu malli tiedostoon
    model.save("ppo_temperature_control")
    print("Malli tallennettu tiedostoon 'ppo_temperature_control.zip'.")

if __name__ == "__main__":
    main()