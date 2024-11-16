#train_ai_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super(TemperatureControlEnv, self).__init__()
        
        # Lisää säätömuutoksen seurantamuuttuja
        self.previous_valve_adjust = 50.0  # Aloitetaan puolivälistä (0-100)
        self.valve_adjust_queue = deque(maxlen=5)  # Viiveen simulointi, 5 askeleen pituus
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Havainnointiavaruus: [CurrentTemp, SetpointTemp]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),  # Minimit lämpötiloille
            high=np.array([100.0, 100.0], dtype=np.float32),  # Maksimit lämpötiloille
            dtype=np.float32
        )
        
        self.noise_level = 0.5  # Kohinan taso
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(10, 70)  # Lähempänä setpointia
        self.setpoint_temp = np.random.uniform(35, 75)  # Satunnainen setpoint
        self.previous_valve_adjust = 50.0  # Palautetaan edellisen episodin arvo
        self.valve_adjust_queue.clear()  # Tyhjennetään viivejono
        for _ in range(self.valve_adjust_queue.maxlen):
            self.valve_adjust_queue.append(50.0)  # Alustetaan viivejono keskialueelle
        
        observation = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # Skaalaa toiminta takaisin alkuperäiselle alueelle (0-100%)
        current_valve_adjust = np.clip((action[0] + 1.0) * 50.0, 0, 100)  # [-1, 1] → [0, 100]
        
        # Lisää venttiilin säätö viivejonoon
        self.valve_adjust_queue.append(current_valve_adjust)
        delayed_valve_adjust = self.valve_adjust_queue[0]  # Käytä viivästettyä arvoa
        
        # Lämpötilan muutos viivästetyn venttiilisäädön perusteella
        temp_change = (delayed_valve_adjust / 100.0) * 60.0 - 30.0  # Muutos (-15.0, 15.0)
        self.current_temp += temp_change
        self.current_temp += np.random.normal(0, self.noise_level)  # Lisää kohinaa
        
        # Rajaamme lämpötilan pysymään järkevissä rajoissa
        self.current_temp = np.clip(self.current_temp, 0, 100)
        
        # Lasketaan virhe
        error = abs(self.current_temp - self.setpoint_temp)
        
        # Penaloi nopeita muutoksia venttiilissä
        valve_change_penalty = abs(current_valve_adjust - self.previous_valve_adjust) * 0.1
        self.previous_valve_adjust = current_valve_adjust  # Päivitä edellinen arvo
        
        # Palkinto logiikka
        previous_error = error  # Tallennetaan edellinen virhe seuraavaa askelta varten

        if error <= 0.5:
            reward = 10.0 - valve_change_penalty  # Suurin palkinto, mutta penaloi isoja muutoksia
            done = True  # Episodi päättyy
        else:
            # Laske pieni palkkio, jos virhe pienenee
            error_delta = previous_error - error  # Positiivinen, jos ollaan lähempänä asetusta
            movement_reward = np.clip(error_delta * 0.1, 0, 1.0)  # Skaalataan palkkio
            reward = movement_reward - 0.1 * error - valve_change_penalty  # Lisää liikkumispalkkio
            reward = np.clip(reward, -10, 10)
            done = False  # Episodi jatkuu
                
        # Palauta tila, palkinto ja muut tiedot
        obs = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return obs, reward, bool(done), False, {"valve_adjust": delayed_valve_adjust}

    def render(self, mode='human'):
        print(f"Nykyinen lämpötila: {self.current_temp:.2f} | Lämpötila-asetus: {self.setpoint_temp:.2f}")