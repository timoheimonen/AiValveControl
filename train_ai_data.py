#train_ai_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super(TemperatureControlEnv, self).__init__()
        
        # Alustetaan venttiilin säätötilan seuranta
        self.previous_valve_adjust = 50.0  # Keskiasetus (0-100%)
        self.valve_adjust_queue = deque(maxlen=5)  # Simuloidaan viivettä, esim. 5 askeleen pituus
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Havainnointitila: [Nykyinen lämpötila, Asetuslämpötila]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),  # Lämpötilojen minimirajat
            high=np.array([100.0, 100.0], dtype=np.float32),  # Lämpötilojen maksimirajat
            dtype=np.float32
        )
        
        self.noise_level = 0.5  # Kohinan taso lämpötilamuutoksissa
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(1, 80)  # Satunnainen aloituslämpötila
        self.setpoint_temp = int(np.random.uniform(35, 75)) # Satunnainen asetuslämpötila
        self.previous_valve_adjust = 50.0  # Nollataan venttiilin säätötila
        self.valve_adjust_queue.clear()  # Tyhjennetään viivejono
        for _ in range(self.valve_adjust_queue.maxlen):
            self.valve_adjust_queue.append(50.0)  # Alustetaan viivejono keskitasolle
        
        observation = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return observation, {}

    def step(self, action):
        # Skaalataan toiminta [-1, 1] → [0, 100%]
        current_valve_adjust = np.clip((action[0] + 1.0) * 50.0, 0, 100)
        
        # Päivitetään venttiilin säätötilan viivejono
        self.valve_adjust_queue.append(current_valve_adjust)
        delayed_valve_adjust = self.valve_adjust_queue[0]  # Käytetään viivästettyä säätöä
        
        # Lasketaan lämpötilan muutos viivästetyn säätötilan perusteella
        temp_change = (delayed_valve_adjust / 100.0) * 60.0 - 30.0  # Muutosalue: (-15.0, 15.0)
        self.current_temp += temp_change
        self.current_temp += np.random.normal(0, self.noise_level)  # Lisää hieman satunnaista kohinaa
        
        # Rajoitetaan lämpötila järkevälle alueelle
        self.current_temp = np.clip(self.current_temp, 0, 100)
        
        # Lasketaan virhe nykyisen lämpötilan ja asetuslämpötilan välillä
        error = abs(self.current_temp - self.setpoint_temp)
        
        # Lisätään rangaistus suurista venttiilin muutoksista
        valve_change_penalty = abs(current_valve_adjust - self.previous_valve_adjust) * 0.1
        self.previous_valve_adjust = current_valve_adjust  # Päivitetään viimeisin säätötila
        
        # Palkitsemis- ja rangaistuslogiikka
        if error <= 0.5:
            reward = 10.0 #- valve_change_penalty  # Maksimipalkkio, mutta penalisoidaan äkkinäisiä muutoksia
            done = True  # Episodi päättyy, kun tavoite saavutetaan
        elif error > 0.5 and error <= 2.0:
            # Lievempi rangaistus toleranssialueen lähistöllä
            reward = -1.0 * error - valve_change_penalty
            done = False
        else:
            # Suurempi rangaistus poistuessa kauemmaksi toleranssialueelta
            reward = -2.0 * error - valve_change_penalty
            done = False
        
        # Palautetaan tila, palkkio ja lisätiedot
        obs = np.array([self.current_temp, self.setpoint_temp], dtype=np.float32)
        return obs, reward, bool(done), False, {"valve_adjust": delayed_valve_adjust}

    def render(self, mode='human'):
        print(f"Nykyinen lämpötila: {self.current_temp:.2f} | Lämpötila-asetus: {self.setpoint_temp:.2f}")