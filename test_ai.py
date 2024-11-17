# test_ai.py

import gymnasium as gym
from stable_baselines3 import PPO
from train_ai_data import TemperatureControlEnv  # Polku tiedostoon, joka sisältää ympäristön määrittelyn

def main():
    # Ladataan koulutettu PPO-malli tiedostosta
    model = PPO.load("ppo_temperature_control")
    
    # Alustetaan lämpötilan säätöympäristö
    env = TemperatureControlEnv()
    
    # Testataan mallia 10 eri episodilla
    for episode in range(10):
        # Resetoi ympäristö ja hae aloitusarvot
        obs, _ = env.reset()
        initial_temp, setpoint_temp = obs  # Alku- ja tavoitelämpötilat
        done = False
        total_reward = 0  # Kerätään palkkioiden summa episodin aikana
        
        print(f"Episodi {episode + 1} alkaa:")
        print(f"Lähtölämpötila: {initial_temp:.2f} | Tavoitelämpötila: {setpoint_temp:.2f}")
        
        # Suorita episodi, kunnes se päättyy
        while not done:
            # Ennustetaan seuraava toiminto koulutetulla mallilla
            action, _states = model.predict(obs, deterministic=True)
            
            # Suoritetaan toiminto ympäristössä ja päivitetään tila
            obs, reward, done, truncated, info = env.step(action)
            
            # Visualisoidaan ympäristön tila
            env.render()
            
            # Lisätään palkkio kokonaispisteisiin
            total_reward += reward
        
        # Tulosta episodin tulos
        print(f"Episodi {episode + 1} päättyi, kokonaisscore: {total_reward:.2f}\n")

if __name__ == "__main__":
    main()