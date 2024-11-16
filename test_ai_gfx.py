import pygame
import time
from collections import deque
from stable_baselines3 import PPO
from train_ai_data import TemperatureControlEnv  # Varmista, että polku on oikein

# Pygamen alustaminen
pygame.init()

# Määrittele ikkunan koko ja värit
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)
TEMP_BAR_COLOR = (100, 200, 100)
SETPOINT_COLOR = (200, 100, 100)
VALVE_COLOR = (100, 100, 200)
HISTOGRAM_COLOR = (100, 150, 250)

# Luo ikkunan
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Temperature Control AI")

# Fontit
font = pygame.font.Font(None, 36)

# Histogrammin parametrit
HISTOGRAM_WIDTH = 300
HISTOGRAM_HEIGHT = 200
HISTOGRAM_MAX_POINTS = 50  # Kuinka monta pistettä näytetään
HISTOGRAM_X = WIDTH - HISTOGRAM_WIDTH - 10
HISTOGRAM_Y = 10

# Histogrammin data
temperature_history = deque(maxlen=HISTOGRAM_MAX_POINTS)

def render_environment(screen, current_temp, setpoint_temp, valve_adjust):
    # Tyhjennä ikkuna
    screen.fill(BACKGROUND_COLOR)
    
    # Lämpötila palkki (skaalataan ikkunan kokoon)
    temp_bar_height = int((current_temp + 50) / 150 * HEIGHT)  # Oletetaan -50...100 skaalaksi
    pygame.draw.rect(screen, TEMP_BAR_COLOR, (WIDTH // 4, HEIGHT - temp_bar_height, 100, temp_bar_height))
    
    # Tavoitelämpötilan viiva
    setpoint_y = HEIGHT - int((setpoint_temp + 50) / 150 * HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (WIDTH // 4 - 20, setpoint_y), (WIDTH // 4 + 120, setpoint_y), 3)
    
    # Valve Adjust -prosentti
    valve_text = font.render(f"Valve Adjust: {valve_adjust:.2f}%", True, VALVE_COLOR)
    screen.blit(valve_text, (WIDTH // 2, HEIGHT - 50))
    
    # Näytä nykyinen ja tavoitelämpötila
    temp_text = font.render(f"Temp: {current_temp:.2f}°C | Setpoint: {setpoint_temp:.2f}°C", True, (255, 255, 255))
    screen.blit(temp_text, (50, 50))
    
    # Päivitä näyttö
    pygame.display.flip()

def render_histogram(screen, temperature_history, setpoint_temp):
    # Piirrä histogrammin tausta
    pygame.draw.rect(screen, (50, 50, 50), (HISTOGRAM_X, HISTOGRAM_Y, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT))
    
    # Skaalaa lämpötila-akseli histogrammiin
    max_temp = 100
    min_temp = 0
    scaled_points = [
        ((HISTOGRAM_X + i * (HISTOGRAM_WIDTH // HISTOGRAM_MAX_POINTS)),
         HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT))
        for i, temp in enumerate(temperature_history)
    ]
    
    # Piirrä historialliset lämpötilat
    if len(scaled_points) > 1:
        pygame.draw.lines(screen, HISTOGRAM_COLOR, False, scaled_points, 2)
    
    # Piirrä tavoitelämpötilan viiva histogrammiin
    setpoint_y = HISTOGRAM_Y + HISTOGRAM_HEIGHT - int((setpoint_temp - min_temp) / (max_temp - min_temp) * HISTOGRAM_HEIGHT)
    pygame.draw.line(screen, SETPOINT_COLOR, (HISTOGRAM_X, setpoint_y), (HISTOGRAM_X + HISTOGRAM_WIDTH, setpoint_y), 2)

    # Päivitä näyttö
    pygame.display.flip()

def main():
    # Lataa koulutettu malli
    model = PPO.load("ppo_temperature_control")
    
    # Luo ympäristö
    env = TemperatureControlEnv()
    
    # Suorita Pygame-silmukka
    running = True
    episode = 1  # Episodien laskuri
    try:
        clock = pygame.time.Clock()
        fps = 10  # Aseta päivitystiheys, esim. 10 FPS
        while running:  # Jatka loputtomasti
            obs, _ = env.reset()
            done = False
            total_reward = 0
            active_after_target = 0  # Laskuri kohinan hallintaan

            # Tyhjennä histogrammi uuden episodin alussa
            temperature_history.clear()

            print(f"Episodi {episode} alkaa...")
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                
                # Suorita AI:n ennustus
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Visualisoi nykyinen tila
                current_temp, setpoint_temp = obs
                valve_adjust = info["valve_adjust"]
                render_environment(screen, current_temp, setpoint_temp, valve_adjust)
                
                # Päivitä histogrammin tiedot ja piirrä se
                temperature_history.append(current_temp)
                render_histogram(screen, temperature_history, setpoint_temp)
                
                total_reward += reward
                
                # Tarkista, jatkuuko episodi
                if terminated:
                    if active_after_target < 500:
                        print("Episodi päättyisi, mutta kohinan hallintaa jatketaan.")
                        terminated = False  # Jatka episodia
                    else:
                        done = True  # Päätetään episodi lopullisesti
                elif True:  # Ohitetaan lämpötilatarkistus
                    active_after_target += 1
                    print(f"active_after_target: {active_after_target}")  # DEBUG: Seuraa laskuria
                    if active_after_target >= 50:  # Esimerkki: 500 sykliä (50 sekuntia, jos fps = 10)
                        print("Aika täyttyi, siirrytään seuraavaan episodiin.")
                        done = True  # Päätetään episodi
                else:
                    if active_after_target > 0:
                        print("active_after_target nollattu.")
                    active_after_target = 0
                
                # Odota hetki
                time.sleep(0.2)
            print(f"Episodi {episode} päättyi, kokonaisscore: {total_reward:.2f}")
            terminated = False
            episode += 1
            
            # 1 sekunnin tauko episodien välillä
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nOhjelma keskeytetty käyttäjän toimesta (Ctrl+C).")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()